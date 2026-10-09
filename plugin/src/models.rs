//! Where the plugin finds its model, first match wins:
//! 1. one the app exported with it: `Contents/Resources/model.tar.gz` inside the VST3 bundle, or
//!    `<name>.model.tar.gz` next to the `<name>.clap` file;
//! 2. the app's current model, `%APPDATA%\Personal Denoiser\model.tar.gz`, picked up again whenever the
//!    app trains a new one;
//! 3. the public DeepFilterNet3 built into the plugin.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub(crate) const MODEL_FILE: &str = "model.tar.gz";
const APP_FOLDER: &str = "Personal Denoiser";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Exported,
    App,
    Public,
}

/// One version of a model file: training again replaces the file, which changes its size or time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Stamp {
    pub(crate) path: PathBuf,
    pub(crate) len: u64,
    pub(crate) modified: Option<SystemTime>,
}

/// The model file to use, if any; `None` means the public model.
pub(crate) fn find() -> Option<(Source, Stamp)> {
    candidates().into_iter().find_map(|(source, path)| stamp(&path).map(|stamp| (source, stamp)))
}

fn stamp(path: &Path) -> Option<Stamp> {
    let meta = std::fs::metadata(path).ok()?;
    meta.is_file().then(|| Stamp { path: path.to_owned(), len: meta.len(), modified: meta.modified().ok() })
}

fn candidates() -> Vec<(Source, PathBuf)> {
    let mut list = Vec::new();
    if let Some(module) = own_module() {
        let parent = module.parent();
        let contents = parent.and_then(Path::parent);
        // <name>.vst3/Contents/x86_64-win/<name>.vst3
        if parent.and_then(Path::file_name).is_some_and(|n| n.to_string_lossy().ends_with("-win"))
            && contents.and_then(Path::file_name).is_some_and(|n| n == "Contents")
        {
            list.push((Source::Exported, contents.unwrap().join("Resources").join(MODEL_FILE)));
        }
        // <name>.clap
        if module.extension().is_some_and(|e| e.eq_ignore_ascii_case("clap")) {
            if let (Some(dir), Some(stem)) = (parent, module.file_stem()) {
                list.push((Source::Exported, dir.join(format!("{}.{MODEL_FILE}", stem.to_string_lossy()))));
            }
        }
    }
    if let Some(appdata) = std::env::var_os("APPDATA") {
        list.push((Source::App, PathBuf::from(appdata).join(APP_FOLDER).join(MODEL_FILE)));
    }
    list
}

/// The plugin's own DLL.
#[cfg(windows)]
fn own_module() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };
    static MARKER: u8 = 0;
    // SAFETY: an address inside this DLL names it, and the handle is borrowed (not reference counted).
    unsafe {
        let mut module = std::mem::zeroed();
        let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        if GetModuleHandleExW(flags, std::ptr::addr_of!(MARKER).cast(), &mut module) == 0 {
            return None;
        }
        let mut buffer = vec![0u16; 32_768];
        let n = GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) as usize;
        (n > 0 && n < buffer.len()).then(|| PathBuf::from(std::ffi::OsString::from_wide(&buffer[..n])))
    }
}

#[cfg(not(windows))]
fn own_module() -> Option<PathBuf> {
    None
}

/// "5 min ago" and the like, for a model file's time.
pub(crate) fn age(modified: Option<SystemTime>) -> Option<String> {
    let seconds = SystemTime::now().duration_since(modified?).ok()?.as_secs();
    Some(match seconds {
        0..=89 => "just now".into(),
        90..=5_399 => format!("{} min ago", (seconds + 30) / 60),
        5_400..=129_599 => format!("{} h ago", (seconds + 1_800) / 3_600),
        _ => format!("{} days ago", (seconds + 43_200) / 86_400),
    })
}
