# Personal Denoiser

A standalone microphone noise remover for Windows using [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet).
Use the built-in model immediately, or optionally record your voice and room noise to fine-tune it.
The same engine is also available as a VST3 / CLAP plugin.

## Windows desktop app

The Korean interface opens on the daily-use microphone controls:

1. **마이크 노이즈 제거** — choose a physical microphone and virtual cable, toggle processing,
   mute/unmute transmission, adjust reduction while running, and monitor the transmitted level.
2. **비교 · 미리듣기** — compare files, listen through headphones, save processed audio, and import/export models.
3. **학습용 녹음** — record/import voice and noise, play takes, or exclude takes without deleting them.
4. **개인 모델 학습** — install training dependencies, start/cancel fine-tuning, and inspect comparisons.

### Send a clean microphone to Discord, games or OBS

- Select your physical microphone at the top.
- In **보낼 장치**, select **CABLE Input** (WASAPI is preferred when available).
- Use the public model or select your personal model, then click **노이즈 제거 켜기**.
- In the receiving application's microphone/input selector, choose **CABLE Output**.

This follows [VB-CABLE's playback-to-recording routing](https://vb-audio.com/Cable/).
The app lists known VB-CABLE and VoiceMeeter pairs, never selects speakers as a transmission fallback,
and prevents selecting the same cable as both microphone and destination. If the receiving endpoint
is missing from the normal input list, open **Windows 녹음 장치 열기** and check that it is enabled.
Newly installed devices may require restarting the app or Windows. Drivers are not installed by this app.

With **창을 닫아도 트레이에서 노이즈 제거 계속하기** checked, closing the window during transmission
hides it while processing continues. Double-click the tray icon to reopen, or choose **완전히 종료** to
stop and exit. **다음 앱 실행 시 노이즈 제거 자동으로 켜기** is opt-in and does not add Windows startup entries.
Device loss stops processing with an error; the app does not switch to unprocessed microphone audio.

Use headphones when monitoring. The app sends audio to the selected output device; it does not
install a virtual microphone driver. An existing virtual audio cable can be selected for routing to
other applications. Processing runs at 48 kHz; unsupported devices report an error. Imported files
are converted to mono 48 kHz. Model delay and device buffering both contribute to live latency.

### Run

Open `dist/Personal Denoiser/Personal Denoiser.exe` in a local build, or double-click `start_app.bat`.
Keep the complete executable folder, including `_internal`, together. Recording and listening do
not require PyTorch. Training requires Python 3.10 or 3.11 with Python Launcher installed; the
**학습 환경 설치** button installs its dependencies separately. Select CUDA only for a compatible
NVIDIA GPU; otherwise clear that checkbox for CPU training. First installation needs internet and
several GB of disk space. Audio is stored and processed locally.

Record clean speech (including soft speech and whispers) and noise without speech, preferably
1–3 minutes of each. Click **내 모델 학습 시작**. After training, select **내 개인 모델** on the first
tab and compare it against the original and public model. Model promotion in the app requires a
successful load and finite audio from the native engine. The previous model is retained as
`model.previous.tar.gz`. These checks verify execution, not whether the new model sounds better.

Data lives in `%APPDATA%/Personal Denoiser` (takes, model, report, log and settings).
Training dependencies and model download caches live in `%LOCALAPPDATA%/Personal Denoiser`.
Excluded recordings move to `excluded` in the data folder. Exported models can be used by the
VST3/CLAP plugin, which also watches the same app model path.

### Development

Python 3.10/3.11 and Rust with the Windows MSVC build tools are required:

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r app/requirements.txt
cargo build --release -p pdn-engine
.venv/Scripts/python.exe run_app.py
```

Build the portable desktop folder with `./build_app.ps1`. Training dependencies are intentionally
excluded from the desktop package and installed only when needed. `Cargo.lock` pins Rust dependencies.

```powershell
cargo test --release -p pdn-engine
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe run_app.py --smoke-test
```

The tests cover audio conversion, recording stop/save, subprocess failure/cancellation, native
processing, invalid model import, cable selection, mute output, disconnected streams and close-to-tray.
They do not open a real microphone, verify reception inside a third-party app, or assess voice quality.

## Plugin

`cargo xtask bundle personal_denoiser --release` builds the VST3 and CLAP plugins. The plugins
process at 48 kHz only (other host rates bypass), with a declared 40 ms latency. Stereo inputs are
mixed to mono. The desktop app and plugins use the same engine DLL/library implementation.
