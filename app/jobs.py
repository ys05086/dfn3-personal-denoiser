"""Background audio jobs; GUI updates are delivered only through Qt signals."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from app.audio import Engine, RATE, read_audio


class AudioJob(QThread):
    message = Signal(str)
    level = Signal(float, float)
    result = Signal(object)
    failed = Signal(str)
    live_stats = Signal(float, float, float, int)
    live_ready = Signal()

    def __init__(self, mode, *, input_device=None, output_device=None, folder=None,
                 dll=None, model=None, reduction=100, source=None, parent=None, muted=False):
        super().__init__(parent)
        self.mode, self.input_device, self.output_device = mode, input_device, output_device
        self.folder, self.dll, self.model = folder, dll, model
        self.reduction, self.source = reduction, source
        self.stop_event = threading.Event()
        self.muted = threading.Event()
        if muted:
            self.muted.set()
        self.error_message = None

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            if self.mode == "record":
                self.record()
            elif self.mode in ("monitor", "route"):
                self.monitor()
            elif self.mode == "render":
                x = read_audio(self.source)
                with Engine(self.dll, self.model, self.reduction) as engine:
                    y = engine.file(x, self.stop_event.is_set)
                if not self.stop_event.is_set():
                    self.result.emit(y)
            elif self.mode == "play":
                self.play()
        except InterruptedError:
            self.message.emit("중지되었습니다.")
        except Exception as exc:
            self.error_message = str(exc)
            self.failed.emit(str(exc))

    def record(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.wav"
        samples = 0
        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=self.input_device,
                                blocksize=4800) as stream, sf.SoundFile(str(path), mode="w", samplerate=RATE,
                                                                       channels=1, subtype="FLOAT") as file:
                while not self.stop_event.is_set() and samples < RATE * 600:
                    x, overflow = stream.read(4800)
                    if overflow:
                        raise RuntimeError("녹음 입력이 끊겼습니다. 다른 오디오 장치를 선택해 주세요.")
                    file.write(x)
                    samples += len(x)
                    self.level.emit(float(np.max(np.abs(x))), samples / RATE)
            if samples < RATE * 2:
                raise ValueError("2초 이상 녹음해 주세요.")
            self.result.emit(path)
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def play(self):
        x = self.source if isinstance(self.source, np.ndarray) else read_audio(self.source)
        channels = min(2, int(sd.query_devices(self.output_device, "output")["max_output_channels"]))
        with sd.OutputStream(samplerate=RATE, channels=channels, dtype="float32", device=self.output_device) as stream:
            for start in range(0, len(x), 2400):
                if self.stop_event.is_set():
                    break
                chunk = np.repeat(x[start:start + 2400, None], channels, axis=1)
                stream.write(np.clip(chunk, -1, 1))

    def monitor(self):
        # Callbacks only move bounded blocks; inference happens on this worker.
        incoming, outgoing = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
        dropped = threading.Event()
        channels = min(2, int(sd.query_devices(self.output_device, "output")["max_output_channels"]))

        def capture(data, frames, time, status):
            if status:
                dropped.set()
            try:
                incoming.put_nowait(data[:, 0].copy())
            except queue.Full:
                dropped.set()
                # Keep current audio instead of accumulating old speech.
                try:
                    incoming.get_nowait()
                    incoming.put_nowait(data[:, 0].copy())
                except (queue.Empty, queue.Full):
                    pass

        def playback(data, frames, time, status):
            if status:
                dropped.set()
            try:
                frame = outgoing.get_nowait()
                if self.muted.is_set() or self.stop_event.is_set():
                    data.fill(0)
                else:
                    data[:] = frame[:, None]
            except queue.Empty:
                data.fill(0)

        self.message.emit("모델을 불러오는 중…")
        with Engine(self.dll, self.model, self.reduction) as engine:
            sd.check_input_settings(device=self.input_device, samplerate=RATE, channels=1, dtype="float32")
            sd.check_output_settings(device=self.output_device, samplerate=RATE, channels=channels, dtype="float32")
            with sd.InputStream(samplerate=RATE, blocksize=engine.hop, channels=1, dtype="float32",
                                device=self.input_device, callback=capture, latency="low") as input_stream, sd.OutputStream(
                    samplerate=RATE, blocksize=engine.hop, channels=channels, dtype="float32",
                    device=self.output_device, callback=playback, latency="low") as output_stream:
                self.live_ready.emit()
                self.message.emit("노이즈 제거 중 · 가상 케이블로 보내고 있습니다." if self.mode == "route" else "실시간 듣기 중 · 선택한 출력 장치로 재생됩니다.")
                frames = 0
                glitches = 0
                last_input = time.monotonic()
                while not self.stop_event.is_set():
                    if not input_stream.active or not output_stream.active:
                        raise RuntimeError("오디오 장치가 연결 해제되었거나 스트림이 중단되었습니다. 장치를 확인한 뒤 다시 켜 주세요.")
                    try:
                        x = incoming.get(timeout=0.2)
                    except queue.Empty:
                        if time.monotonic() - last_input > 3:
                            raise RuntimeError("마이크 입력이 중단되었습니다. 장치를 확인한 뒤 다시 켜 주세요.")
                        continue
                    last_input = time.monotonic()
                    started = time.perf_counter()
                    engine.set_reduction(self.reduction)
                    y = np.clip(engine.frame(x), -1, 1)
                    cost_ms = (time.perf_counter() - started) * 1000
                    try:
                        outgoing.put_nowait(y)
                    except queue.Full:
                        dropped.set()
                        try:
                            outgoing.get_nowait()
                            outgoing.put_nowait(y)
                        except (queue.Empty, queue.Full):
                            pass
                    frames += 1
                    if frames % 10 == 0:
                        self.level.emit(float(np.max(np.abs(x))), frames * engine.hop / RATE)
                        self.live_stats.emit(float(np.max(np.abs(x))), 0.0 if self.muted.is_set() else float(np.max(np.abs(y))), cost_ms, glitches)
                    if dropped.is_set():
                        glitches += 1
                        dropped.clear()
