"""Audio and native engine helpers. Each Engine belongs to one worker thread."""
from __future__ import annotations

import ctypes as ct
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 48000


def read_audio(path: Path) -> np.ndarray:
    x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("비어 있거나 유효하지 않은 오디오입니다.")
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        x = resample_poly(x, RATE // divisor, rate // divisor)
    return np.ascontiguousarray(x, dtype=np.float32)


class Engine:
    def __init__(self, dll: Path, model: Path | None = None, reduction=100):
        self.handle = None
        self.lib = ct.CDLL(str(dll))
        pointer = ct.POINTER(ct.c_float)
        for name, args, result in (
            ("pdn_open", [ct.c_char_p, ct.c_char_p, ct.c_size_t], ct.c_void_p),
            ("pdn_hop", [ct.c_void_p], ct.c_size_t),
            ("pdn_delay", [ct.c_void_p], ct.c_size_t),
            ("pdn_set_reduction", [ct.c_void_p, ct.c_float], None),
            ("pdn_process", [ct.c_void_p, pointer, pointer], ct.c_float),
            ("pdn_close", [ct.c_void_p], None),
        ):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        error = ct.create_string_buffer(4096)
        self.handle = self.lib.pdn_open(str(model).encode("utf-8") if model else None, error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode("utf-8", "replace"))
        self.hop = self.lib.pdn_hop(self.handle)
        self.delay = self.lib.pdn_delay(self.handle)
        self.lib.pdn_set_reduction(self.handle, reduction)

    def frame(self, x):
        if not self.handle:
            raise RuntimeError("엔진이 닫혔습니다.")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (self.hop,):
            raise ValueError(f"expected {self.hop} mono samples")
        y = np.empty_like(x)
        pointer = ct.POINTER(ct.c_float)
        snr = self.lib.pdn_process(self.handle, x.ctypes.data_as(pointer), y.ctypes.data_as(pointer))
        if not np.isfinite(snr) or not np.isfinite(y).all():
            raise RuntimeError("엔진에서 유효하지 않은 오디오가 반환되었습니다.")
        return y

    def set_reduction(self, value):
        if self.handle:
            self.lib.pdn_set_reduction(self.handle, float(value))

    def file(self, x, cancelled=lambda: False):
        # Flush the lookahead, then align output with the original for A/B playback.
        count = math.ceil((len(x) + self.delay) / self.hop) * self.hop
        padded = np.pad(x, (0, count - len(x)))
        y = np.empty(count, np.float32)
        for start in range(0, count, self.hop):
            if cancelled():
                raise InterruptedError("취소되었습니다.")
            y[start:start + self.hop] = self.frame(padded[start:start + self.hop])
        return y[self.delay:self.delay + len(x)]

    def close(self):
        if self.handle:
            self.lib.pdn_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
