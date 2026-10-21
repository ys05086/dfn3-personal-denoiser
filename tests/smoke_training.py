"""Optional end-to-end GUI training smoke test, run in a Python with training dependencies.

Uses synthetic audio in a temporary directory; never replaces the user's model or opens a mic.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import paths
from app.main import Window
from app.audio import Engine


def main():
    app = QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        overrides = dict(DATA=base, CACHE=base / "cache", TAKES=base / "takes", MODEL=base / "model.tar.gz",
                         MODEL_INFO=base / "model.json", SETTINGS=base / "settings.json", ENV=base / "env")
        with patch.multiple(paths, **overrides):
            window = Window()
            t = np.arange(48000 * 4) / 48000
            voice = (.03 * np.sin(2 * np.pi * 180 * t) * (.3 + .7 * np.sin(2 * np.pi * 3 * t) ** 2)).astype(np.float32)
            noise = np.random.default_rng(2).normal(0, .005, len(t)).astype(np.float32)
            sf.write(paths.TAKES / "voice" / "synthetic.wav", voice, 48000)
            sf.write(paths.TAKES / "noise" / "synthetic.wav", noise, 48000)
            window.training_python = lambda: Path(sys.executable)
            window.training_ready = lambda: True
            window.steps.setValue(2)
            window.batch.setValue(1)
            window.train()
            deadline = time.monotonic() + 180
            while window.process is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.02)
            if window.process:
                window.stop()
                while window.process:
                    app.processEvents()
                    time.sleep(.02)
                raise RuntimeError("Training timed out")
            if window.process_error or not paths.MODEL.is_file():
                raise RuntimeError(window.log.toPlainText())
            with Engine(paths.engine_dll(), paths.MODEL) as engine:
                output = engine.file(voice + noise)
                assert len(output) == len(voice) and np.isfinite(output).all()
            assert paths.MODEL_INFO.is_file()
            assert window.model.currentData() == "personal"
            assert window.steps.isEnabled()
            print("GUI training -> ONNX export -> native validation -> model promotion -> native playback data: OK")
            window.close()


if __name__ == "__main__":
    main()
