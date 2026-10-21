import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from app import paths
from app.audio import Engine, RATE, read_audio
from app.jobs import AudioJob
from app.main import Window
from app.model_job import ModelImport


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_resample_and_reject_nonfinite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stereo.wav"
            sf.write(path, np.ones((44100, 2), np.float32) * .1, 44100, subtype="FLOAT")
            audio = read_audio(path)
            self.assertEqual(audio.shape, (RATE,))
            sf.write(path, [float("nan")], RATE, subtype="FLOAT")
            with self.assertRaises(ValueError):
                read_audio(path)

    def test_record_saves_on_stop_without_touching_real_microphone(self):
        with tempfile.TemporaryDirectory() as folder:
            job = AudioJob("record", folder=Path(folder))
            class Stream:
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def read(self, size):
                    self.count = getattr(self, "count", 0) + 1
                    if self.count == 21: job.stop()
                    return np.ones((size, 1), np.float32) * .03, False
            with patch("app.jobs.sd.InputStream", return_value=Stream()):
                job.record()
            files = list(Path(folder).glob("*.wav"))
            self.assertEqual(len(files), 1)
            self.assertAlmostEqual(sf.info(files[0]).duration, 2.1)

    def test_process_failure_and_cancellation_unlock_ui(self):
        import sys
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            overrides = dict(DATA=base, CACHE=base / "cache", TAKES=base / "takes", MODEL=base / "model.tar.gz",
                             MODEL_INFO=base / "model.json", SETTINGS=base / "settings.json", ENV=base / "env")
            with patch.multiple(paths, **overrides):
                window = Window()
                window.commands = [[sys.executable, "-c", "raise SystemExit(7)"]]
                window.begin_process("train")
                self.wait_until(lambda: window.process is None)
                self.assertTrue(window.process_error)
                self.assertTrue(window.steps.isEnabled())
                window.commands = [[sys.executable, "-c", "import time; time.sleep(30)"]]
                window.begin_process("train")
                self.wait_until(lambda: bool(window.process.processId()))
                window.stop()
                self.wait_until(lambda: window.process is None)
                self.assertTrue(window.process_cancelled)
                self.assertTrue(window.steps.isEnabled())
                window.close()

    def wait_until(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.assertTrue(predicate())

    @unittest.skipUnless(paths.engine_dll(), "build engine DLL first")
    def test_native_engine_output_shape_noise_and_invalid_import(self):
        x = np.random.default_rng(0).normal(0, .01, RATE * 2 + 137).astype(np.float32)
        with Engine(paths.engine_dll()) as engine:
            result = engine.file(x)
            self.assertEqual(len(x), len(result))
            self.assertTrue(np.isfinite(result).all())
            self.assertLess(np.mean(result[RATE:] ** 2), np.mean(x[RATE:] ** 2) * .1)
            with self.assertRaises(ValueError): engine.frame(x[:100])
        with tempfile.TemporaryDirectory() as folder:
            bad, current = Path(folder) / "bad.tar.gz", Path(folder) / "model.tar.gz"
            bad.write_bytes(b"invalid archive")
            current.write_bytes(b"keep existing model")
            job = ModelImport(bad, current, paths.engine_dll())
            job.run()
            self.assertEqual(current.read_bytes(), b"keep existing model")
            self.assertFalse(list(Path(folder).glob("import-*")))


if __name__ == "__main__":
    unittest.main()
