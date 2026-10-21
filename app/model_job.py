from pathlib import Path
import os
import shutil
import threading
import uuid

import numpy as np
from PySide6.QtCore import QThread, Signal
from app.audio import Engine


class ModelImport(QThread):
    message = Signal(str)
    failed = Signal(str)

    def __init__(self, source: Path, destination: Path, dll: Path, parent=None):
        super().__init__(parent)
        self.source, self.destination, self.dll = source, destination, dll
        self.cancelled = threading.Event()

    def stop(self):
        self.cancelled.set()

    def run(self):
        staging = self.destination.with_name(f"import-{uuid.uuid4().hex}.tar.gz")
        try:
            self.message.emit("개인 모델을 검증하는 중…")
            shutil.copyfile(self.source, staging)
            with Engine(self.dll, staging) as engine:
                for _ in range(20):
                    if self.cancelled.is_set():
                        return
                    engine.frame(np.zeros(engine.hop, np.float32))
            if not self.cancelled.is_set():
                if self.destination.exists():
                    shutil.copy2(self.destination, self.destination.with_name("model.previous.tar.gz"))
                os.replace(staging, self.destination)
                self.destination.with_name("model.json").unlink(missing_ok=True)
                self.destination.with_name("model.tar.gz.sha256").unlink(missing_ok=True)
                self.message.emit("개인 모델을 가져왔습니다.")
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            staging.unlink(missing_ok=True)
