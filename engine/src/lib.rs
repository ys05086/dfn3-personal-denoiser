//! DeepFilterNet3 on one 48 kHz mono stream, as both the app (through the C API below) and the plugin
//! run it: libDF's tract runtime with the settings of libDF's own C API (local-SNR thresholds -15, 35 and
//! 35 dB, no post filter, the mask reduced by its maximum), so the two sound the same.
//!
//! A model is the ONNX archive DeepFilterNet3 is released as (`DeepFilterNet3_onnx.tar.gz`): the encoder,
//! both decoders and `config.ini`. Fine-tunes of the same network load the same way; the public model is
//! built in.

use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::Array2;

pub const SAMPLE_RATE: usize = 48_000;
/// libDF treats a limit under 0.01 dB as "pass the input through" and skips its delay; 0.01 dB already
/// mixes back 99.9 % of the input, so the output keeps one latency at every setting.
const MIN_REDUCTION_DB: f32 = 0.01;
/// 100 dB or more: no limit.
pub const MAX_REDUCTION_DB: f32 = 100.0;

pub struct Engine {
    df: DfTract,
    input: Array2<f32>,
    output: Array2<f32>,
    reduction_db: f32,
}

// SAFETY: tract keeps intermediate tensors in `Rc`s owned by the model states inside `DfTract`. None of
// them is shared outside it (tract's thread-locals hold only scratch memory), and the engine is never
// cloned, so moving it to another thread moves every reference with it. One thread uses it at a time.
unsafe impl Send for Engine {}

impl Engine {
    /// The public DeepFilterNet3.
    pub fn public() -> Result<Self, String> {
        guarded(|| Self::new(DfParams::default()))
    }

    /// A model archive in memory.
    pub fn from_archive(bytes: &[u8]) -> Result<Self, String> {
        guarded(|| Self::new(DfParams::from_bytes(bytes).map_err(|e| format!("not a model archive: {e:#}"))?))
    }

    /// A model archive on disk, read whole first so the file is not held open.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_archive(&bytes)
    }

    fn new(params: DfParams) -> Result<Self, String> {
        let runtime = RuntimeParams::default_with_ch(1)
            .with_atten_lim(MAX_REDUCTION_DB)
            .with_thresholds(-15.0, 35.0, 35.0)
            .with_post_filter(0.0)
            .with_mask_reduce(ReduceMask::MAX);
        let df = DfTract::new(params, &runtime).map_err(|e| format!("model could not be built: {e:#}"))?;
        if df.sr != SAMPLE_RATE || df.ch != 1 || df.hop_size == 0 || df.fft_size < df.hop_size {
            return Err(format!("unsupported model: {} Hz, {} channel(s), hop {}, FFT {}",
                               df.sr, df.ch, df.hop_size, df.fft_size));
        }
        let hop = df.hop_size;
        Ok(Self { df, input: Array2::zeros((1, hop)), output: Array2::zeros((1, hop)),
                  reduction_db: MAX_REDUCTION_DB })
    }

    /// Samples per call to [`Engine::process`] (480: 10 ms).
    pub fn hop(&self) -> usize {
        self.df.hop_size
    }

    /// The model's own delay in samples: the analysis window beyond one hop plus its lookahead
    /// (1440, 30 ms, for DeepFilterNet3).
    pub fn delay(&self) -> usize {
        self.df.fft_size - self.df.hop_size + self.df.lookahead * self.df.hop_size
    }

    /// The most the noise may be lowered, in dB; 100 or more removes as much as the model decides.
    pub fn set_reduction(&mut self, db: f32) {
        let db = if db.is_finite() { db.clamp(MIN_REDUCTION_DB, MAX_REDUCTION_DB) } else { MAX_REDUCTION_DB };
        if db != self.reduction_db {
            self.df.set_atten_lim(db);
            self.reduction_db = db;
        }
    }

    /// One hop of input in, one hop of output out. Returns the model's local SNR estimate in dB.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32, String> {
        let hop = self.hop();
        if input.len() != hop || output.len() != hop {
            return Err(format!("expected {hop} samples"));
        }
        self.input.as_slice_mut().expect("contiguous").copy_from_slice(input);
        let (df, frame, out) = (&mut self.df, &self.input, &mut self.output);
        let lsnr = guarded(|| df.process(frame.view(), out.view_mut()).map_err(|e| format!("{e:#}")))?;
        output.copy_from_slice(self.output.as_slice().expect("contiguous"));
        Ok(lsnr)
    }
}

/// libDF reports some malformed models by panicking; those become errors here.
fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let reason = panic.downcast_ref::<String>().cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown error".into());
        Err(format!("model error: {reason}"))
    })
}

// ------------------------------------------------------------------ C API (pdn_engine.dll)

fn write_error(message: &str, buffer: *mut c_char, length: usize) {
    if buffer.is_null() || length == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let mut n = bytes.len().min(length - 1);
    while n > 0 && !message.is_char_boundary(n) {
        n -= 1;
    }
    // SAFETY: the caller passes a writable buffer of `length` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), n);
        *buffer.add(n) = 0;
    }
}

/// Opens a model archive (UTF-8 path), or the public model when `path` is null. On failure returns null
/// and writes the reason, NUL-terminated, into `error` (`error_length` bytes; may be null).
///
/// # Safety
/// `path` is null or a NUL-terminated string; `error` is null or writable for `error_length` bytes.
#[no_mangle]
pub unsafe extern "C" fn pdn_open(path: *const c_char, error: *mut c_char, error_length: usize) -> *mut Engine {
    let engine = if path.is_null() {
        Engine::public()
    } else {
        match CStr::from_ptr(path).to_str() {
            Ok(path) => Engine::from_file(Path::new(path)),
            Err(_) => Err("path is not UTF-8".into()),
        }
    };
    match engine {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(message) => {
            write_error(&message, error, error_length);
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `engine` comes from [`pdn_open`] and is not closed.
#[no_mangle]
pub unsafe extern "C" fn pdn_hop(engine: *const Engine) -> usize {
    (*engine).hop()
}

/// # Safety
/// `engine` comes from [`pdn_open`] and is not closed.
#[no_mangle]
pub unsafe extern "C" fn pdn_delay(engine: *const Engine) -> usize {
    (*engine).delay()
}

/// # Safety
/// `engine` comes from [`pdn_open`] and is not closed.
#[no_mangle]
pub unsafe extern "C" fn pdn_set_reduction(engine: *mut Engine, db: f32) {
    (*engine).set_reduction(db)
}

/// Processes one hop (`pdn_hop` samples) from `input` into `output`. Returns the local SNR estimate, or
/// NaN on failure, with `output` silent.
///
/// # Safety
/// `engine` comes from [`pdn_open`]; `input` and `output` hold `pdn_hop(engine)` floats each.
#[no_mangle]
pub unsafe extern "C" fn pdn_process(engine: *mut Engine, input: *const f32, output: *mut f32) -> f32 {
    let engine = &mut *engine;
    let hop = engine.hop();
    let input = std::slice::from_raw_parts(input, hop);
    let output = std::slice::from_raw_parts_mut(output, hop);
    match engine.process(input, output) {
        Ok(lsnr) => lsnr,
        Err(_) => {
            output.fill(0.0);
            f32::NAN
        }
    }
}

/// # Safety
/// `engine` comes from [`pdn_open`] (or is null) and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn pdn_close(engine: *mut Engine) {
    if !engine.is_null() {
        drop(Box::from_raw(engine));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 s of white noise around -39 dBFS.
    fn noise() -> Vec<f32> {
        let mut seed = 1u32;
        (0..2 * SAMPLE_RATE).map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.04
        }).collect()
    }

    fn run(engine: &mut Engine, x: &[f32]) -> Vec<f32> {
        let hop = engine.hop();
        let mut y = vec![0.0; x.len() / hop * hop];
        for (i, o) in x.chunks_exact(hop).zip(y.chunks_exact_mut(hop)) {
            assert!(engine.process(i, o).expect("process").is_finite());
        }
        y
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|&v| (v as f64).powi(2)).sum()
    }

    #[test]
    fn public_model_shape_and_delay() {
        let engine = Engine::public().expect("public model");
        assert_eq!((engine.hop(), engine.delay()), (480, 1440));
    }

    #[test]
    fn removes_noise() {
        let noise = noise();
        let mut engine = Engine::public().unwrap();
        let out = run(&mut engine, &noise);
        // the second half, after the model settles: white noise alone comes out at least 20 dB lower
        let half = noise.len() / 2;
        assert!(energy(&out[half..]) < energy(&noise[half..]) * 0.01);
    }

    #[test]
    fn reduction_limit_keeps_noise() {
        let noise = noise();
        let mut engine = Engine::public().unwrap();
        engine.set_reduction(6.0);
        let out = run(&mut engine, &noise);
        let half = noise.len() / 2;
        let change = 10.0 * (energy(&out[half..]) / energy(&noise[half..])).log10();
        assert!((-7.5..-4.5).contains(&change), "6 dB limit lowered noise by {change:.1} dB");
    }

    #[test]
    fn bad_archives_are_errors() {
        assert!(Engine::from_archive(b"not a model").is_err());
        assert!(Engine::from_file(Path::new("missing/model.tar.gz")).is_err());
        // a valid gzip tar without the model files
        let mut tar = tar_without_model();
        assert!(Engine::from_archive(&tar).is_err());
        tar.truncate(10);
        assert!(Engine::from_archive(&tar).is_err());
    }

    /// An empty tar (two zero blocks) gzipped as one stored deflate block: header, block, CRC-32, size.
    fn tar_without_model() -> Vec<u8> {
        let data = vec![0u8; 1024];
        let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
        out.push(1); // final stored block
        out.extend_from_slice(&(data.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(data.len() as u16)).to_le_bytes());
        out.extend_from_slice(&data);
        out.extend_from_slice(&crc32(&data).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }

    #[test]
    fn c_api() {
        unsafe {
            let mut error = [0 as c_char; 256];
            let missing = c"missing/model.tar.gz";
            assert!(pdn_open(missing.as_ptr(), error.as_mut_ptr(), error.len()).is_null());
            assert!(CStr::from_ptr(error.as_ptr()).to_str().unwrap().contains("missing"));
            let engine = pdn_open(std::ptr::null(), error.as_mut_ptr(), error.len());
            assert!(!engine.is_null());
            assert_eq!(pdn_hop(engine), 480);
            pdn_set_reduction(engine, 100.0);
            let input = [0.01f32; 480];
            let mut output = [1.0f32; 480];
            assert!(pdn_process(engine, input.as_ptr(), output.as_mut_ptr()).is_finite());
            pdn_close(engine);
        }
    }
}
