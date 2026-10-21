import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from app import paths
from app.main import Window
from app.jobs import AudioJob
from app.routing import virtual_outputs, receiver_name, visible_receiver


class RoutingTests(unittest.TestCase):
    def test_only_known_virtual_outputs_and_wasapi_first(self):
        devices = [
            dict(name="Speakers", hostapi=0, max_input_channels=0, max_output_channels=2),
            dict(name="CABLE Input (VB-Audio Virtual Cable)", hostapi=0, max_input_channels=0, max_output_channels=16),
            dict(name="CABLE Input (VB-Audio Virtual Cable)", hostapi=1, max_input_channels=0, max_output_channels=2),
            dict(name="CABLE Output (VB-Audio Virtual Cable)", hostapi=1, max_input_channels=2, max_output_channels=0),
        ]
        hosts = [dict(name="MME"), dict(name="Windows WASAPI")]
        self.assertEqual([item[0] for item in virtual_outputs(devices, hosts)], [2, 1])
        self.assertTrue(visible_receiver("CABLE Output", devices, hosts))
        self.assertIsNone(receiver_name("Speakers"))
        self.assertIsNone(receiver_name("CABLE In 16ch"))
        self.assertEqual(receiver_name("CABLE-A Input (VB-Audio)"), "CABLE-A Output")

    def exercise_stream(self, muted=False, connected=True):
        job = AudioJob("route", dll=Path("unused.dll"), muted=muted)
        callbacks, output = {}, []

        class Input:
            active = connected
            def __init__(self, **kwargs): callbacks["input"] = kwargs["callback"]
            def __enter__(self):
                callbacks["input"](np.full((480, 1), .4, np.float32), 480, None, False)
                return self
            def __exit__(self, *_): pass

        class Output:
            active = True
            def __init__(self, **kwargs): callbacks["output"] = kwargs["callback"]
            def __enter__(self): return self
            def __exit__(self, *_): pass

        class FakeEngine:
            hop = 480
            def __init__(self, *_): self.count = 0
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def set_reduction(self, value): pass
            def frame(self, x):
                self.count += 1
                callbacks["input"](np.full((480, 1), .4, np.float32), 480, None, False)
                data = np.ones((480, 2), np.float32)
                callbacks["output"](data, 480, None, False)
                output.append(data.copy())
                if self.count == 6: job.stop()
                return np.full(480, .1, np.float32)

        with patch("app.jobs.Engine", FakeEngine), patch("app.jobs.sd.InputStream", Input), \
             patch("app.jobs.sd.OutputStream", Output), patch("app.jobs.sd.check_input_settings"), \
             patch("app.jobs.sd.check_output_settings"), \
             patch("app.jobs.sd.query_devices", return_value={"max_output_channels": 2}):
            job.monitor()
        return output

    def test_sends_processed_audio_and_mute_sends_zero(self):
        normal = self.exercise_stream()
        self.assertTrue(any(np.allclose(frame, .1) for frame in normal))
        self.assertFalse(any(np.allclose(frame, .4) for frame in normal))
        self.assertTrue(all(np.count_nonzero(frame) == 0 for frame in self.exercise_stream(muted=True)))

    def test_disconnected_device_stops_with_error(self):
        with self.assertRaisesRegex(RuntimeError, "오디오 장치"):
            self.exercise_stream(connected=False)

    def test_close_to_tray_keeps_stream_and_explicit_exit_stops_it(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            overrides = dict(DATA=base, CACHE=base / "cache", TAKES=base / "takes", MODEL=base / "model.tar.gz",
                             MODEL_INFO=base / "model.json", SETTINGS=base / "settings.json", ENV=base / "env")
            with patch.multiple(paths, **overrides):
                window = Window()
                window.show()
                app.processEvents()
                stopped = []
                window.job = SimpleNamespace(mode="route", stop=lambda: stopped.append(True))
                event = QCloseEvent()
                with patch("app.main.QSystemTrayIcon.isSystemTrayAvailable", return_value=True), patch.object(window.tray, "showMessage"):
                    window.closeEvent(event)
                self.assertFalse(event.isAccepted())
                self.assertFalse(window.isVisible())
                self.assertFalse(stopped)
                window.request_exit()
                self.assertTrue(stopped)
                window.job = None
                window.close()


if __name__ == "__main__":
    unittest.main()
