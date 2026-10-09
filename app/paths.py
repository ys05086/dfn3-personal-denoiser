"""Where everything lives.

Bundle (next to the program): the engine DLL, the plugins, and the training script.
Data (%APPDATA%\\Personal Denoiser): the takes, the trained model (the plugin reads it from here too), settings.
Local (%LOCALAPPDATA%\\Personal Denoiser): the training environment and download caches, which are large.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

APP_NAME = "Personal Denoiser"
FROZEN = getattr(sys, "frozen", False)
BUNDLE = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent
REPO = None if FROZEN else BUNDLE.parent


def _folder(variable: str, fallback: str) -> Path:
    root = os.environ.get(variable) or str(Path.home() / fallback)
    return Path(root) / APP_NAME


DATA = _folder("APPDATA", "AppData/Roaming")
LOCAL = _folder("LOCALAPPDATA", "AppData/Local")

TAKES = DATA / "takes"
MODEL = DATA / "model.tar.gz"
MODEL_INFO = DATA / "model.json"
SETTINGS = DATA / "settings.json"
LOG = DATA / "app.log"

TRAINING_SCRIPT = BUNDLE / "training" / "train.py"
ENV = LOCAL / "env"
CACHE = LOCAL / "cache"


def engine_dll() -> Path | None:
    """pdn_engine.dll: next to the program, or, when run from the source tree, the CI build unpacked into
    the repository (target/release)."""
    candidates = [BUNDLE / "pdn_engine.dll"]
    if REPO:
        candidates.append(REPO / "target" / "release" / "pdn_engine.dll")
    return next((p for p in candidates if p.is_file()), None)


def plugin_dir() -> Path | None:
    """The folder holding "Personal Denoiser.vst3" and "Personal Denoiser.clap"."""
    candidates = [BUNDLE / "plugin"]
    if REPO:
        candidates.append(REPO / "target" / "bundled")
    return next((p for p in candidates if (p / f"{APP_NAME}.vst3").is_dir()), None)
