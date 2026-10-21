//! Personal Denoiser as a VST3 and CLAP plugin: DeepFilterNet3 with the user's own model when there is
//! one (see `models`), the public model otherwise.
//!
//! The audio thread runs the model itself, one 10 ms frame at a time, as the app does: 40 ms of latency
//! in all. Models are built on a loader thread, which also watches for a newly trained model and hands it
//! over; the audio thread only ever trades engines through `try_lock`, so it never waits.

mod editor;
mod models;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use nice_plug::prelude::*;
use nice_plug_egui::EguiEditorState;
use pdn_engine::{Engine, MAX_REDUCTION_DB, SAMPLE_RATE};

use models::{Source, Stamp};

/// DeepFilterNet3's frame and its own delay (10 ms and 30 ms). Every model the app trains shares them; a
/// model that does not is refused.
const HOP: usize = 480;
const MODEL_DELAY: usize = 1440;
/// Input is gathered into whole frames first, one frame more: 40 ms in all.
const LATENCY: usize = HOP + MODEL_DELAY;

#[derive(Params)]
pub(crate) struct DenoiserParams {
    /// libDF's attenuation limit: the most the noise may be lowered.
    #[id = "reduction"]
    pub(crate) reduction: FloatParam,
}

impl Default for DenoiserParams {
    fn default() -> Self {
        Self {
            reduction: FloatParam::new("Max reduction", MAX_REDUCTION_DB,
                                       FloatRange::Linear { min: 0.0, max: MAX_REDUCTION_DB })
                .with_step_size(1.0)
                .with_value_to_string(Arc::new(|db: f32| {
                    if db >= MAX_REDUCTION_DB { "No limit".into() } else { format!("{db:.0} dB") }
                }))
                .with_string_to_value(Arc::new(|text: &str| {
                    let text = text.trim();
                    if text.eq_ignore_ascii_case("no limit") {
                        return Some(MAX_REDUCTION_DB);
                    }
                    text.trim_end_matches("dB").trim().parse().ok()
                })),
        }
    }
}

/// What the editor shows about the model.
#[derive(Clone, Default)]
pub(crate) struct Status {
    /// `None` until the first model is ready.
    pub(crate) source: Option<Source>,
    pub(crate) stamp: Option<Stamp>,
    /// A model file that could not be used, and why.
    pub(crate) error: Option<String>,
}

/// A newly built engine for the audio thread, and the one it replaced, for the loader to free.
#[derive(Default)]
struct Handoff {
    fresh: Option<Box<Engine>>,
    retired: Option<Box<Engine>>,
}

pub(crate) struct Shared {
    handoff: Mutex<Handoff>,
    pub(crate) status: Mutex<Status>,
    /// The host's rate while active; the editor explains the bypass when it is not 48 kHz.
    pub(crate) sample_rate: AtomicU32,
    stop: AtomicBool,
}

struct PersonalDenoiser {
    params: Arc<DenoiserParams>,
    shared: Arc<Shared>,
    editor_state: Arc<EguiEditorState>,
    loader: Option<JoinHandle<()>>,
    engine: Option<Box<Engine>>,
    /// At 48 kHz; at any other rate the audio passes untouched.
    active: bool,
    input: Vec<f32>,
    output: Vec<f32>,
    pos: usize,
    /// Until the first model is ready, the input delayed as the model would delay it.
    dry: Vec<f32>,
    dry_pos: usize,
}

impl Default for PersonalDenoiser {
    fn default() -> Self {
        Self {
            params: Arc::new(DenoiserParams::default()),
            shared: Arc::new(Shared {
                handoff: Mutex::new(Handoff::default()),
                status: Mutex::new(Status::default()),
                sample_rate: AtomicU32::new(0),
                stop: AtomicBool::new(false),
            }),
            editor_state: editor::default_state(),
            loader: None,
            engine: None,
            active: false,
            input: vec![0.0; HOP],
            output: vec![0.0; HOP],
            pos: 0,
            dry: vec![0.0; MODEL_DELAY],
            dry_pos: 0,
        }
    }
}

impl Plugin for PersonalDenoiser {
    const NAME: &'static str = "Personal Denoiser";
    const VENDOR: &'static str = "ys05086";
    const URL: &'static str = "https://github.com/ys05086/dfn3-personal-denoiser";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();
    type Editor = nice_plug_egui::EguiEditor<editor::DenoiserEditor>;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Self::Editor> {
        editor::create(self.params.clone(), self.shared.clone(), self.editor_state.clone())
    }

    fn activate(&mut self, _layout: &AudioIOLayout, config: &BufferConfig,
                context: &mut impl ActivateContext<Self>) -> bool {
        self.shared.sample_rate.store(config.sample_rate.round() as u32, Ordering::Relaxed);
        self.active = (config.sample_rate - SAMPLE_RATE as f32).abs() < 0.5;
        context.set_latency_samples(if self.active { LATENCY as u32 } else { 0 });
        if self.loader.is_none() {
            let shared = self.shared.clone();
            self.loader = std::thread::Builder::new().name("Personal Denoiser model".into())
                .spawn(move || load_models(shared)).ok();
        }
        true
    }

    fn reset(&mut self) {
        self.input.fill(0.0);
        self.output.fill(0.0);
        self.dry.fill(0.0);
        self.pos = 0;
        self.dry_pos = 0;
    }

    fn process(&mut self, buffer: &mut Buffer, _aux: &mut AuxiliaryBuffers,
               _context: &mut impl ProcessContext<Self>) -> ProcessStatus {
        if !self.active {
            return ProcessStatus::Normal;
        }
        self.adopt();
        let channels = buffer.as_slice();
        let Some(samples) = channels.first().map(|c| c.len()) else {
            return ProcessStatus::Normal;
        };
        // a stereo input is mixed to mono, and the result goes to both sides
        let scale = 1.0 / channels.len() as f32;
        for i in 0..samples {
            self.input[self.pos] = channels.iter().map(|c| c[i]).sum::<f32>() * scale;
            let y = self.output[self.pos];
            for channel in channels.iter_mut() {
                channel[i] = y;
            }
            self.pos += 1;
            if self.pos == HOP {
                self.pos = 0;
                self.run_frame();
            }
        }
        ProcessStatus::Normal
    }
}

impl PersonalDenoiser {
    /// Takes a newly loaded engine if one is waiting and the loader has freed the last one replaced.
    fn adopt(&mut self) {
        let Ok(mut handoff) = self.shared.handoff.try_lock() else {
            return;
        };
        if handoff.retired.is_none() {
            if let Some(fresh) = handoff.fresh.take() {
                handoff.retired = self.engine.replace(fresh);
            }
        }
    }

    fn run_frame(&mut self) {
        match self.engine.as_mut() {
            Some(engine) => {
                engine.set_reduction(self.params.reduction.value());
                if engine.process(&self.input, &mut self.output).is_err() {
                    self.output.fill(0.0);
                }
            }
            None => {
                for (x, y) in self.input.iter().zip(self.output.iter_mut()) {
                    *y = std::mem::replace(&mut self.dry[self.dry_pos], *x);
                    self.dry_pos = (self.dry_pos + 1) % MODEL_DELAY;
                }
            }
        }
    }
}

impl Drop for PersonalDenoiser {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(loader) = self.loader.take() {
            let _ = loader.join();
        }
    }
}

fn frames_match(engine: Engine) -> Result<Engine, String> {
    if engine.hop() == HOP && engine.delay() == MODEL_DELAY {
        Ok(engine)
    } else {
        Err(format!("frame {} and delay {} differ from DeepFilterNet3's", engine.hop(), engine.delay()))
    }
}

/// The loader thread: builds the model the plugin should use, then checks every second whether that has
/// changed (a new model trained, or one exported next to the plugin) and builds the new one.
fn load_models(shared: Arc<Shared>) {
    let mut tried: Option<Option<Stamp>> = None;
    let mut serving = false;
    while !shared.stop.load(Ordering::Acquire) {
        let found = models::find();
        let wanted = found.as_ref().map(|(_, stamp)| stamp.clone());
        if tried.as_ref() != Some(&wanted) {
            tried = Some(wanted);
            let built = match &found {
                Some((source, stamp)) => Engine::from_file(&stamp.path).and_then(frames_match)
                    .map(|engine| (engine, *source, Some(stamp.clone()))),
                None => Engine::public().map(|engine| (engine, Source::Public, None)),
            };
            let mut status = shared.status.lock().map(|s| s.clone()).unwrap_or_default();
            match built {
                Ok((engine, source, stamp)) => {
                    hand_over(&shared, engine);
                    serving = true;
                    status = Status { source: Some(source), stamp, error: None };
                }
                Err(error) => {
                    let name = found.as_ref().map(|(_, s)| s.path.display().to_string()).unwrap_or_default();
                    status.error = Some(format!("{name}: {error}"));
                    if !serving {
                        if let Ok(engine) = Engine::public() {
                            hand_over(&shared, engine);
                            serving = true;
                            status.source = Some(Source::Public);
                            status.stamp = None;
                        }
                    }
                }
            }
            if let Ok(mut current) = shared.status.lock() {
                *current = status;
            }
        }
        // free the engine the audio thread let go of, outside the lock
        let retired = shared.handoff.lock().ok().and_then(|mut h| h.retired.take());
        drop(retired);
        for _ in 0..10 {
            if shared.stop.load(Ordering::Acquire) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

fn hand_over(shared: &Shared, engine: Engine) {
    // an engine the audio thread never took (it is not running) is simply replaced
    let replaced = shared.handoff.lock().ok().and_then(|mut h| h.fresh.replace(Box::new(engine)));
    drop(replaced);
}

impl ClapPlugin for PersonalDenoiser {
    const CLAP_ID: &'static str = "io.github.ys05086.personal-denoiser";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("DeepFilterNet3 noise removal with your own fine-tuned model");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] =
        &[ClapFeature::AudioEffect, ClapFeature::Restoration, ClapFeature::Mono, ClapFeature::Stereo];
}

impl Vst3Plugin for PersonalDenoiser {
    // Stable 16-byte identifier; the display name above may contain spaces.
    const VST3_CLASS_ID: [u8; 16] = *b"PersonalDenoiser";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[Vst3SubCategory::Fx, Vst3SubCategory::Restoration];
}

nice_export_clap!(PersonalDenoiser);
nice_export_vst3!(PersonalDenoiser);
