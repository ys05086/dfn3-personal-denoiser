"""Fine-tune DeepFilterNet3 on the takes recorded in the app, and export the model the app and the plugin load.

Runs in the training environment the app installs (PyTorch, deepfilternet), as its own process, and prints
one JSON line per progress update: {"stage", "done", "total", ...}. The recipe and the export are those of
github.com/ys05086/dfn3-personal-finetune (finetune.py): clean voice takes mixed with the noise takes at
-5..20 dB SNR, voice levels spread over 30 dB, 15 % noise-only and 10 % voice-only examples; the compressed
spectral loss (gamma 0.3) with removed voice weighted 2x; AdamW lr 1e-4 with cosine decay; BatchNorm
statistics frozen. The last quarter of every take is held out to compare the public and the new model.

The new model replaces <out>/model.tar.gz only after it has loaded and run in the app's engine.

  python train.py --takes <dir with voice/ and noise/> --out <dir> --cache <dir> [--engine pdn_engine.dll]
                  [--steps 2000] [--no-compare]
"""
from __future__ import annotations

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import types
import urllib.request
import zipfile

import numpy as np
import soundfile as sf
from scipy import signal
import torch

RATE = 48000
SEGMENT = 4 * RATE
# The released DeepFilterNet3 checkpoint, pinned to a commit of github.com/Rikorose/DeepFilterNet.
BASE_URL = ("https://github.com/Rikorose/DeepFilterNet/raw/d375b2d8309e0935d165700c91da9de862a99c31/"
            "models/DeepFilterNet3.zip")
BASE_SHA256 = "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2"

# deepfilternet 0.5.6 imports torchaudio.backend.common (only for a type hint); newer torchaudio dropped it.
if "torchaudio.backend.common" not in sys.modules:
    _backend, _common = types.ModuleType("torchaudio.backend"), types.ModuleType("torchaudio.backend.common")
    _common.AudioMetaData = type("AudioMetaData", (), {})
    _backend.common = _common
    sys.modules["torchaudio.backend"], sys.modules["torchaudio.backend.common"] = _backend, _common


def report(stage, done=0, total=0, **extra):
    print(json.dumps({"stage": stage, "done": done, "total": total, **extra}, ensure_ascii=False), flush=True)


def ensure_base(cache: Path) -> Path:
    """The public checkpoint, downloaded and checked once (about 8 MB)."""
    base = cache / "DeepFilterNet3"
    if not (base / "config.ini").is_file():
        report("download")
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "DeepFilterNet3.zip"
        urllib.request.urlretrieve(BASE_URL, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != BASE_SHA256:
            archive.unlink()
            raise RuntimeError(f"the downloaded DeepFilterNet3 checkpoint does not match (sha256 {digest})")
        with zipfile.ZipFile(archive) as z:
            z.extractall(cache)
        archive.unlink()
    return base


def fresh_state():
    """libDF's analysis keeps streaming state, so every batch and every file gets a new one."""
    from libdf import DF
    return DF(sr=RATE, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2)


def load(path) -> np.ndarray:
    """Mono float32 at 48 kHz, with digital silence (from mic gates) trimmed off both ends."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != RATE:
        x = signal.resample_poly(x, RATE, sr).astype(np.float32)
    sound = np.flatnonzero(np.abs(x) > 1e-5)
    return x[sound[0]:sound[-1] + 1] if sound.size else x


def load_folder(folder: Path) -> dict[str, np.ndarray]:
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".wav", ".flac"))
    return {p.name: load(p) for p in files}


def split(x, held_out=0.25):
    cut = int(len(x) * (1 - held_out))
    return x[:cut], x[cut:]


def mixer(clean, noises, seed):
    weights = np.array([len(x) for x in clean], float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)

    def crop(x):
        if len(x) <= SEGMENT:
            return np.pad(x, (0, SEGMENT - len(x)))
        start = int(rng.integers(0, len(x) - SEGMENT + 1))
        return x[start:start + SEGMENT]

    def sample():
        c = crop(clean[rng.choice(len(clean), p=weights)]) * 10 ** (rng.uniform(-25, 5) / 20)
        if rng.random() < 0.15:
            c = np.zeros(SEGMENT, np.float32)  # noise alone must come out silent
        n = crop(noises[rng.integers(len(noises))])
        if rng.random() < 0.1:
            n = np.zeros(SEGMENT, np.float32)  # clean voice must come out untouched
        else:
            active = np.abs(c) > 1e-4
            ref = np.mean(c[active] ** 2) if active.mean() > 0.05 else 10 ** (rng.uniform(-60, -30) / 10)
            n = n * np.sqrt(ref / ((np.mean(n ** 2) + 1e-12) * 10 ** (rng.uniform(-5, 20) / 10)))
        return c.astype(np.float32), n.astype(np.float32)

    return sample


def train(base, clean, noises, steps, batch, lr, device, seed):
    from df.enhance import df_features, init_df
    from df.loss import SpectralLoss
    model, _, _ = init_df(str(base), log_level="ERROR", log_file=None)
    model = model.to(device)
    loss_fn = SpectralLoss(gamma=0.3, factor_magnitude=1000, factor_complex=1000, factor_under=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            module.eval()  # a few minutes of one room would skew the running statistics
    torch.manual_seed(seed)
    sample = mixer(clean, noises, seed)
    for step in range(steps):
        pairs = [sample() for _ in range(batch)]
        c = torch.tensor(np.stack([p[0] for p in pairs]))
        mix = c + torch.tensor(np.stack([p[1] for p in pairs]))
        spec, erb_feat, spec_feat = df_features(mix, fresh_state(), 96, device=device)
        target, _, _ = df_features(c, fresh_state(), 96, device=device)
        loss = loss_fn(model(spec, erb_feat, spec_feat)[0], target)
        if not torch.isfinite(loss):
            raise RuntimeError("학습 손실이 유효하지 않습니다. 녹음의 음량과 내용을 확인해 주세요.")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0 or step == steps - 1:
            report("train", step + 1, steps, loss=round(loss.item(), 4))
    return model.eval()


def export_onnx(model, base, archive: Path, work: Path):
    """The encoder and the two decoders as ONNX graphs in the archive layout libDF reads (the same as the
    official DeepFilterNet3_onnx.tar.gz), traced with the TorchScript exporter."""
    import onnx
    from df.enhance import df_features
    work.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).cpu().eval()
    spec, feat_erb, feat_spec = df_features(torch.randn(1, RATE), fresh_state(), 96, device="cpu")
    feat_spec = feat_spec.transpose(1, 4).squeeze(4)

    def export(module, inputs, names, outputs, axes, path):
        torch.onnx.export(module, inputs, str(path), input_names=names, output_names=outputs, dynamic_axes=axes,
                          opset_version=14, keep_initializers_as_inputs=False, dynamo=False)

    with torch.no_grad():
        e0, e1, e2, e3, emb, c0, lsnr = model.enc(feat_erb, feat_spec)
        export(model.enc, (feat_erb, feat_spec), ["feat_erb", "feat_spec"], ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"],
               {"feat_erb": {2: "S"}, "feat_spec": {2: "S"}, "e0": {2: "S"}, "e1": {2: "S"}, "e2": {2: "S"},
                "e3": {2: "S"}, "emb": {1: "S"}, "c0": {2: "S"}, "lsnr": {1: "S"}}, work / "enc.onnx")
        export(model.erb_dec, (emb.clone(), e3, e2, e1, e0), ["emb", "e3", "e2", "e1", "e0"], ["m"],
               {"emb": {1: "S"}, "e3": {2: "S"}, "e2": {2: "S"}, "e1": {2: "S"}, "e0": {2: "S"}, "m": {2: "S"}},
               work / "erb_dec.onnx")
        export(model.df_dec, (emb.clone(), c0), ["emb", "c0"], ["coefs"],
               {"emb": {1: "S"}, "c0": {2: "S"}, "coefs": {1: "S"}}, work / "df_dec.onnx")
    for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx"):
        onnx.checker.check_model(str(work / name))
    shutil.copy(base / "config.ini", work / "config.ini")
    with tarfile.open(archive, mode="w:gz") as tar:
        for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"):
            tar.add(work / name, arcname=f"tmp/export/{name}")
    shutil.rmtree(work)


def check_in_engine(engine_dll: Path, archive: Path):
    """The model must load in the engine the app and the plugin run, and give finite audio."""
    lib = ctypes.CDLL(str(engine_dll))
    lib.pdn_open.restype = ctypes.c_void_p
    lib.pdn_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.pdn_process.restype = ctypes.c_float
    lib.pdn_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.pdn_close.argtypes = [ctypes.c_void_p]
    error = ctypes.create_string_buffer(1024)
    engine = lib.pdn_open(str(archive).encode("utf-8"), error, len(error))
    if not engine:
        raise RuntimeError(f"the new model does not load: {error.value.decode('utf-8', 'replace')}")
    try:
        x = np.random.default_rng(0).normal(0, 0.02, 100 * 480).astype(np.float32)
        y = np.zeros(480, np.float32)
        for i in range(0, len(x), 480):
            frame = np.ascontiguousarray(x[i:i + 480])
            lsnr = lib.pdn_process(engine, frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            if not np.isfinite(lsnr) or not np.isfinite(y).all():
                raise RuntimeError("the new model produced invalid samples")
    finally:
        lib.pdn_close(engine)


def frame_db(x, hop=480):
    n = len(x) // hop
    return 10 * np.log10(np.mean(np.asarray(x[:n * hop], np.float64).reshape(n, hop) ** 2, axis=1) + 1e-20)


def compare(models, voices, noises, snr=10.0):
    """Per voice take, mixed with each noise take at `snr` dB: the share of voice frames cut by more than
    10 dB (lower keeps more of the voice). Per noise take alone: the level change in dB (lower removes more)."""
    cut = {name: {label: [] for label in models} for name in voices}
    for vname, v in voices.items():
        fv = frame_db(v)
        audible = fv > np.percentile(fv[fv > -110], 10) + 10 if (fv > -110).any() else np.zeros_like(fv, bool)
        if not audible.any():
            audible = fv > -80
        if not audible.any():
            continue
        for n in noises.values():
            n = np.resize(n, len(v))
            active = np.abs(v) > 1e-4
            reference = np.mean(v[active] ** 2) if active.any() else np.mean(v ** 2)
            n = (n * np.sqrt(reference / ((np.mean(n ** 2) + 1e-12) * 10 ** (snr / 10)))).astype(np.float32)
            for label, run in models.items():
                fo = frame_db(run((v + n).astype(np.float32))[:len(v)])[:len(fv)]
                cut[vname][label].append(float(np.mean((fo - fv[:len(fo)])[audible[:len(fo)]] < -10)))
    noise_db = {name: {label: float(10 * np.log10((np.sum(run(n).astype(np.float64) ** 2) + 1e-20)
                                                  / (np.sum(n.astype(np.float64) ** 2) + 1e-20)))
                       for label, run in models.items()} for name, n in noises.items()}
    return {"snr_db": snr,
            "voice_cut": {name: {label: float(np.mean(v)) for label, v in r.items()} for name, r in cut.items() if all(r.values())},
            "noise_db": noise_db}


def main():
    if "--check-env" in sys.argv:
        import onnx
        from df.enhance import init_df
        fresh_state()
        report("environment", device="cuda" if torch.cuda.is_available() else "cpu", torch=torch.__version__)
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    started = time.time()
    try:
        if args.steps <= 0 or args.batch <= 0 or args.lr <= 0:
            raise ValueError("steps, batch and learning rate must be positive")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        report("prepare", device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)
        voices, noises = load_folder(args.takes / "voice"), load_folder(args.takes / "noise")
        voices = {k: x for k, x in voices.items() if len(x) >= 2 * RATE and np.isfinite(x).all() and np.max(np.abs(x)) > 1e-4}
        noises = {k: x for k, x in noises.items() if len(x) >= 2 * RATE and np.isfinite(x).all() and np.sqrt(np.mean(x ** 2)) >= 1e-4}
        if not voices or not noises:
            raise RuntimeError("need at least one voice take (2 s or more) and one noise take that is not silent")
        base = ensure_base(args.cache)
        model = train(base, [split(x)[0] for x in voices.values()], [split(x)[0] for x in noises.values()],
                      args.steps, args.batch, args.lr, device, args.seed)

        report("export")
        args.out.mkdir(parents=True, exist_ok=True)
        staging = args.out / "model.tar.gz.new"
        export_onnx(model, base, staging, args.cache / "export")
        if args.engine:
            check_in_engine(args.engine, staging)
        info = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "steps": args.steps,
                "device": device, "seconds": round(time.time() - started),
                "voice_seconds": round(sum(map(len, voices.values())) / RATE),
                "noise_seconds": round(sum(map(len, noises.values())) / RATE),
                "sha256": hashlib.sha256(staging.read_bytes()).hexdigest()}

        if not args.no_compare:
            report("compare")
            from df.enhance import enhance, init_df
            public = init_df(str(base), log_level="ERROR", log_file=None)[0].to(device).eval()

            def runner(m):
                def run(x):
                    with torch.no_grad():
                        return enhance(m, fresh_state(), torch.from_numpy(x).view(1, -1))[0].cpu().numpy()
                return run

            held_voice = {k: split(x)[1] for k, x in voices.items() if len(split(x)[1]) >= RATE // 2}
            held_noise = {k: split(x)[1] for k, x in noises.items() if len(split(x)[1]) >= RATE // 2}
            if held_voice and held_noise:
                info["compare"] = compare({"public": runner(public), "personal": runner(model)}, held_voice, held_noise)

        # Keep the last working model available for manual rollback.
        current = args.out / "model.tar.gz"
        if current.exists():
            shutil.copy2(current, args.out / "model.previous.tar.gz")
        os.replace(staging, current)
        (args.out / "model.tar.gz.sha256").write_text(info["sha256"] + "\n", encoding="ascii")
        (args.out / "model.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        report("done", 1, 1, info=info)
    except Exception as exc:  # the app shows the message; the traceback goes to its log
        import traceback
        traceback.print_exc(file=sys.stderr)
        report("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
