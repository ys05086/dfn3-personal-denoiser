"""A local-first recording, fine-tuning and listening desktop app."""
from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QLockFile, QUrl
from PySide6.QtGui import QCloseEvent, QIcon, QPixmap, QPainter, QColor, QPen, QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox, QTabWidget, QGroupBox,
    QProgressBar, QSpinBox, QSlider, QPlainTextEdit, QListWidget, QCheckBox, QSystemTrayIcon, QMenu)

from app import paths
from app.audio import read_audio, RATE
from app.jobs import AudioJob
from app.routing import virtual_outputs, visible_receiver, receiver_name

STYLE = """
QWidget { background: #101821; color: #e7edf4; font-family: 'Malgun Gothic'; font-size: 13px; }
QMainWindow { background: #101821; }
QLabel#title { font-size: 28px; font-weight: 700; }
QLabel#subtitle { color: #98acbd; }
QLabel#status { color: #8be0c0; padding: 10px; background: #1b2c35; border-radius: 6px; }
QGroupBox { border: 1px solid #304353; border-radius: 9px; margin-top: 14px; padding: 18px 12px 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; color: #8be0c0; padding: 0 5px; }
QPushButton { background: #263b4b; border: 1px solid #3a5265; border-radius: 6px; padding: 9px 15px; }
QPushButton:hover { background: #36536a; }
QPushButton#primary { background: #79d7b2; color: #102b25; font-weight: 700; }
QPushButton:disabled { background: #1b2732; color: #627383; border-color: #293744; }
QComboBox, QSpinBox, QListWidget, QPlainTextEdit { background: #182430; border: 1px solid #304353; border-radius: 5px; padding: 6px; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background: #2b514e; }
QTabWidget::pane { border: 0; }
QTabBar::tab { padding: 12px 22px; color: #98acbd; border-bottom: 2px solid #263b4b; }
QTabBar::tab:selected { color: #8be0c0; border-bottom: 2px solid #8be0c0; }
QProgressBar { border: 0; background: #233544; border-radius: 4px; text-align: center; height: 18px; }
QProgressBar::chunk { background: #4a9d87; border-radius: 4px; }
QSlider::groove:horizontal { height: 5px; background: #304353; }
QSlider::handle:horizontal { background: #8be0c0; width: 16px; margin: -6px 0; border-radius: 7px; }
"""


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        for folder in (paths.DATA, paths.CACHE, paths.TAKES / "voice", paths.TAKES / "noise"):
            folder.mkdir(parents=True, exist_ok=True)
        self.setWindowTitle("Personal Denoiser")
        self.resize(980, 840)
        self.live_active = False
        self.devices, self.hosts = [], []
        self.job = None
        self.process = None
        self.commands = []
        self.process_kind = ""
        self.process_cancelled = False
        self.process_error = False
        self.process_buffer = ""
        self.pending_audio = None
        self.rendered = None
        self.sample = None
        self.closing = False
        self.locked = []
        try:
            self.settings = json.loads(paths.SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.settings = {}
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 22, 28, 18)
        layout.setSpacing(12)
        title = QLabel("Personal Denoiser")
        title.setObjectName("title")
        layout.addWidget(title)
        sub = QLabel("내 목소리는 남기고, 내 공간의 소음은 줄이기")
        sub.setObjectName("subtitle")
        layout.addWidget(sub)
        self.model_status = QLabel()
        layout.addWidget(self.model_status)
        devices = QHBoxLayout()
        self.input = QComboBox()
        self.output = QComboBox()
        for combo in (self.input, self.output):
            combo.setMinimumContentsLength(18)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        devices.addWidget(QLabel("마이크"))
        devices.addWidget(self.input, 1)
        devices.addWidget(QLabel("미리듣기"))
        devices.addWidget(self.output, 1)
        devices.addWidget(self.button("새로고침", self.refresh_devices))
        layout.addLayout(devices)
        self.locked += [self.input, self.output]
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.live_tab()
        self.listen_tab()
        self.record_tab()
        self.train_tab()
        footer = QHBoxLayout()
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setFormat("입력 대기")
        footer.addWidget(self.meter, 1)
        self.stop_button = self.button("■ 중지", self.stop, lock=False)
        footer.addWidget(self.stop_button)
        footer.addWidget(self.button("저장 폴더", lambda: os.startfile(paths.DATA), lock=False))
        layout.addLayout(footer)
        self.status = QLabel("마이크와 보낼 장치를 선택한 뒤 노이즈 제거를 켜세요. 기본 모델은 학습 없이 사용할 수 있습니다.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.refresh_devices()
        self.refresh_takes()
        self.refresh_model()
        self.model.setCurrentIndex(1 if paths.MODEL.is_file() and self.settings.get("model", "personal") == "personal" else 0)
        self.reduction.setValue(int(self.settings.get("reduction", 100)))
        self.setup_tray()
        self.busy(False)
        if self.auto_start.isChecked() and "--smoke-test" not in sys.argv:
            QTimer.singleShot(500, self.toggle_live)

    def button(self, text, action, primary=False, lock=True):
        button = QPushButton(text)
        button.clicked.connect(action)
        if primary:
            button.setObjectName("primary")
        if lock:
            self.locked.append(button)
        return button

    def page(self, title):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 12, 0, 8)
        layout.setSpacing(12)
        self.tabs.addTab(widget, title)
        return layout

    def live_tab(self):
        layout = self.page("마이크 노이즈 제거")
        box = QGroupBox("디스코드 · 게임 · OBS에 보낼 마이크")
        form = QVBoxLayout(box)
        form.setSpacing(12)
        self.live_label = QLabel("꺼짐")
        self.live_label.setStyleSheet("font-size: 26px; font-weight: 700; color: #98acbd")
        form.addWidget(self.live_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("보낼 장치"))
        self.route = QComboBox()
        self.route.setMinimumContentsLength(25)
        self.route.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.route.currentIndexChanged.connect(self.update_route_hint)
        row.addWidget(self.route, 1)
        self.locked.append(self.route)
        form.addLayout(row)
        self.route_hint = QLabel()
        self.route_hint.setWordWrap(True)
        self.route_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addWidget(self.route_hint)
        row = QHBoxLayout()
        self.model = QComboBox()
        self.model.addItem("기본 DeepFilterNet3 · 학습 없이 사용", "public")
        self.model.addItem("내 개인 모델", "personal")
        row.addWidget(self.model)
        row.addWidget(QLabel("소음 제거"))
        self.reduction = QSlider(Qt.Orientation.Horizontal)
        self.reduction.setRange(1, 100)
        self.reduction.setValue(100)
        self.reduction_label = QLabel("최대 제거")
        self.reduction.valueChanged.connect(self.change_reduction)
        row.addWidget(self.reduction, 1)
        row.addWidget(self.reduction_label)
        self.locked += [self.model, self.reduction]
        form.addLayout(row)
        row = QHBoxLayout()
        self.live_button = self.button("노이즈 제거 켜기", self.toggle_live, primary=True, lock=False)
        self.live_button.setMinimumHeight(54)
        row.addWidget(self.live_button, 2)
        self.mute_button = self.button("마이크 음소거", self.toggle_mute, lock=False)
        self.mute_button.setCheckable(True)
        row.addWidget(self.mute_button, 1)
        form.addLayout(row)
        self.sent_meter = QProgressBar()
        self.sent_meter.setRange(0, 100)
        self.sent_meter.setFormat("전송 출력 대기")
        form.addWidget(self.sent_meter)
        self.live_detail = QLabel("마이크 → 노이즈 제거 → 가상 케이블 → 사용하는 프로그램")
        self.live_detail.setWordWrap(True)
        form.addWidget(self.live_detail)
        layout.addWidget(box)
        row = QHBoxLayout()
        row.addWidget(self.button("Windows 녹음 장치 열기", self.open_recording_devices, lock=False))
        row.addWidget(self.button("VB-CABLE 공식 안내", lambda: QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/")), lock=False))
        row.addStretch()
        layout.addLayout(row)
        self.keep_running = QCheckBox("창을 닫아도 트레이에서 노이즈 제거 계속하기")
        self.keep_running.setChecked(self.settings.get("keep_running", True))
        self.auto_start = QCheckBox("다음 앱 실행 시 노이즈 제거 자동으로 켜기")
        self.auto_start.setChecked(self.settings.get("auto_start", False))
        layout.addWidget(self.keep_running)
        layout.addWidget(self.auto_start)
        note = QLabel("다른 프로그램의 마이크 입력은 위에 안내된 장치로 선택하세요.\n"
                      "앱을 완전히 종료하면 전송이 중단됩니다. 개인 모델 학습은 선택 사항입니다.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

    def setup_tray(self):
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor("#101821"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#79d7b2"))
        painter.drawRoundedRect(23, 10, 18, 30, 9, 9)
        painter.setPen(QPen(QColor("#79d7b2"), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(16, 24, 32, 25, 180 * 16, 180 * 16)
        painter.drawLine(32, 48, 32, 56)
        painter.drawLine(24, 56, 40, 56)
        painter.end()
        self.setWindowIcon(QIcon(pixmap))
        self.tray = QSystemTrayIcon(QIcon(pixmap), self)
        menu = QMenu(self)
        menu.addAction("창 열기", self.show_window)
        self.tray_toggle = menu.addAction("노이즈 제거 켜기", self.toggle_live)
        self.tray_mute = menu.addAction("마이크 음소거", self.toggle_mute)
        menu.addSeparator()
        menu.addAction("완전히 종료", self.request_exit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.show_window() if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
        self.tray.setToolTip("Personal Denoiser · 꺼짐")
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def request_exit(self):
        self.closing = True
        self.close()

    def open_recording_devices(self):
        subprocess.Popen(["control.exe", "mmsys.cpl,,1"], creationflags=subprocess.CREATE_NO_WINDOW)

    def change_reduction(self, value):
        self.reduction_label.setText(f"최대 {value} dB" if value < 100 else "최대 제거")
        if self.job and getattr(self.job, "mode", None) in ("route", "monitor"):
            self.job.reduction = value

    def update_route_hint(self):
        if not hasattr(self, "route_hint"):
            return
        info = self.route.currentData()
        if info is None:
            self.route_hint.setText("가상 케이블이 없습니다. VB-CABLE 설치 후 앱을 다시 실행해 주세요.")
            return
        receiver = info["receiver"]
        text = f"다른 프로그램의 마이크 입력: {receiver}\n이 앱은 {info['name']}으로 처리된 소리를 보냅니다."
        if not visible_receiver(receiver, self.devices, self.hosts):
            text += "\n수신 장치가 일반 입력 목록에 보이지 않습니다. Windows 녹음 장치에서 사용 상태를 확인하세요."
        self.route_hint.setText(text)

    def toggle_live(self):
        if self.job and getattr(self.job, "mode", None) == "route":
            self.stop()
            return
        if self.job or self.process:
            self.say("현재 작업을 중지한 뒤 노이즈 제거를 켜 주세요.")
            return
        try:
            info = self.route.currentData()
            if info is None:
                raise ValueError("보낼 가상 케이블이 없습니다. VB-CABLE을 설치한 뒤 앱을 다시 실행해 주세요.")
            model = self.selected_model()
            actual_input = self.input.currentData()
            if actual_input is None:
                actual_input = int(sd.default.device[0])
            input_name = self.devices[actual_input]["name"]
            if input_name.split("(")[0].strip().casefold() == info["receiver"].casefold():
                raise ValueError("보내는 가상 케이블을 마이크 입력으로 다시 선택할 수 없습니다. 실제 마이크를 선택하세요.")
            self.live_label.setText("시작하는 중…")
            self.start_job("route", dll=paths.engine_dll(), model=model, reduction=self.reduction.value(),
                           output_device=info["index"], muted=self.mute_button.isChecked())
        except Exception as exc:
            self.error(str(exc))

    def toggle_mute(self, checked=None):
        if checked is None or not isinstance(self.sender(), QPushButton):
            self.mute_button.setChecked(not self.mute_button.isChecked())
        if self.job and getattr(self.job, "mode", None) == "route":
            if self.mute_button.isChecked():
                self.job.muted.set()
            else:
                self.job.muted.clear()
        self.update_live_state()

    def update_live_state(self):
        routing = self.job is not None and getattr(self.job, "mode", None) == "route"
        muted = self.mute_button.isChecked()
        self.live_button.setText("노이즈 제거 끄기" if routing else "노이즈 제거 켜기")
        self.mute_button.setText("음소거 해제" if muted else "마이크 음소거")
        if routing:
            self.live_label.setText(("음소거 중" if muted else "노이즈 제거 켜짐") if self.live_active else "시작하는 중…")
            self.live_label.setStyleSheet("font-size: 26px; font-weight: 700; color: #79d7b2")
        else:
            self.live_label.setText("꺼짐")
            self.live_label.setStyleSheet("font-size: 26px; font-weight: 700; color: #98acbd")
            self.sent_meter.setValue(0)
            self.sent_meter.setFormat("전송 출력 대기")
        if hasattr(self, "tray"):
            self.tray.setToolTip("Personal Denoiser · " + self.live_label.text())
            self.tray_toggle.setText(self.live_button.text())
            self.tray_toggle.setEnabled(self.live_button.isEnabled())
            self.tray_mute.setText(self.mute_button.text())
            self.tray_mute.setEnabled(routing)

    def on_live_ready(self):
        self.live_active = True
        self.update_live_state()

    def show_live_stats(self, input_peak, output_peak, cost_ms, glitches):
        if not self.job or getattr(self.job, "mode", None) != "route":
            return
        db = 20 * np.log10(max(output_peak, 1e-6))
        self.sent_meter.setValue(round(max(0, min(100, (db + 60) / 60 * 100))))
        self.sent_meter.setFormat("음소거 · 전송 출력 0" if self.mute_button.isChecked() else f"전송 출력 {db:.0f} dBFS")
        self.live_detail.setText(f"10 ms 프레임 처리 {cost_ms:.1f} ms  ·  버퍼 끊김 {glitches}회  ·  48 kHz\n"
                                "처리 시간과 별도로 모델 및 오디오 장치의 지연이 추가됩니다.")

    def record_tab(self):
        layout = self.page("학습용 녹음")
        hint = QLabel("목소리: 조용한 상태에서 평소 말투·작은 목소리·속삭임을 녹음하세요.\n"
                      "소음: 말하지 않고 키보드·마우스·선풍기 등 실제 주변 소리를 녹음하세요. 각 종류 1~3분을 권장합니다.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("목소리", "voice")
        self.kind.addItem("소음", "noise")
        self.kind.currentIndexChanged.connect(self.refresh_takes)
        row.addWidget(self.kind)
        row.addWidget(self.button("● 녹음 시작", self.record, primary=True))
        row.addWidget(self.button("파일 가져오기", self.import_takes))
        row.addStretch()
        self.locked.append(self.kind)
        layout.addLayout(row)
        self.take_list = QListWidget()
        layout.addWidget(self.take_list, 1)
        self.totals = QLabel()
        layout.addWidget(self.totals)
        row = QHBoxLayout()
        row.addWidget(self.button("선택 녹음 듣기", self.play_take))
        row.addWidget(self.button("학습에서 제외", self.exclude_take))
        row.addStretch()
        layout.addLayout(row)

    def train_tab(self):
        layout = self.page("개인 모델 학습")
        box = QGroupBox("처음 한 번 · 학습 환경")
        form = QVBoxLayout(box)
        self.env_status = QLabel()
        form.addWidget(self.env_status)
        row = QHBoxLayout()
        self.cuda = QCheckBox("NVIDIA GPU 사용 (CUDA 12.8)")
        self.cuda.setChecked(bool(shutil.which("nvidia-smi")))
        self.locked.append(self.cuda)
        row.addWidget(self.cuda)
        row.addStretch()
        row.addWidget(self.button("학습 환경 설치", self.install_environment))
        form.addLayout(row)
        form.addWidget(QLabel("최초 설치는 인터넷과 수 GB 공간이 필요합니다. 녹음과 학습은 이 PC에서 실행됩니다."))
        layout.addWidget(box)
        row = QHBoxLayout()
        self.steps = QSpinBox()
        self.steps.setRange(1, 20000)
        self.steps.setValue(2000)
        self.steps.setSingleStep(100)
        self.batch = QSpinBox()
        self.batch.setRange(1, 32)
        self.batch.setValue(8)
        self.locked += [self.steps, self.batch]
        row.addWidget(QLabel("학습 횟수"))
        row.addWidget(self.steps)
        row.addWidget(QLabel("배치 크기"))
        row.addWidget(self.batch)
        row.addStretch()
        row.addWidget(self.button("내 모델 학습 시작", self.train, primary=True))
        layout.addLayout(row)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        layout.addWidget(self.log, 1)
        self.metrics = QLabel("학습이 끝나면 기본 모델과의 비교 결과가 여기에 표시됩니다.")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)

    def listen_tab(self):
        layout = self.page("비교 · 미리듣기")
        layout.addWidget(QLabel("마이크 노이즈 제거 탭에서 선택한 모델과 제거 강도로 비교합니다."))
        box = QGroupBox("같은 파일로 비교하기")
        form = QVBoxLayout(box)
        self.sample_label = QLabel("비교할 녹음 파일을 선택하세요.")
        self.sample_label.setWordWrap(True)
        form.addWidget(self.sample_label)
        row = QHBoxLayout()
        row.addWidget(self.button("비교 파일 선택", self.choose_sample))
        row.addWidget(self.button("원본 듣기", self.play_original))
        row.addWidget(self.button("선택 모델로 처리 · 듣기", self.render, primary=True))
        form.addLayout(row)
        form.addWidget(self.button("마지막 처리 결과 WAV 저장", self.save_rendered))
        layout.addWidget(box)
        box = QGroupBox("마이크 실시간 듣기")
        form = QVBoxLayout(box)
        note = QLabel("헤드폰을 연결한 뒤 시작하세요. 처리된 마이크 소리를 위에서 선택한 출력 장치로 보냅니다.\n"
                      "이 기능은 직접 듣는 용도입니다. 다른 프로그램에 보내려면 첫 번째 탭에서 노이즈 제거를 켜세요.")
        note.setWordWrap(True)
        form.addWidget(note)
        form.addWidget(self.button("실시간 듣기 시작", self.monitor, primary=True))
        layout.addWidget(box)
        row = QHBoxLayout()
        row.addWidget(self.button("개인 모델 가져오기", self.import_model))
        row.addWidget(self.button("개인 모델 내보내기", self.export_model))
        layout.addLayout(row)
        layout.addStretch()

    def busy(self, value):
        for widget in self.locked:
            widget.setEnabled(not value)
        self.stop_button.setEnabled(value)
        routing = self.job is not None and getattr(self.job, "mode", None) == "route"
        self.live_button.setEnabled(not value or routing)
        self.mute_button.setEnabled(routing)
        if routing:
            self.reduction.setEnabled(True)
        self.update_live_state()

    def say(self, message):
        self.status.setText(message)
        logging.info(message)

    def error(self, message):
        self.say("오류: " + message)
        if not self.isVisible():
            self.show_window()
        QMessageBox.warning(self, "작업을 완료하지 못했어요", message)

    def refresh_devices(self):
        try:
            devices, hosts = sd.query_devices(), sd.query_hostapis()
            self.devices, self.hosts = devices, hosts
            for combo, key in ((self.input, "input"), (self.output, "output")):
                previous = combo.currentText() or self.settings.get(key, "")
                combo.clear()
                combo.addItem("Windows 기본 장치", None)
                for i, device in enumerate(devices):
                    if device[f"max_{key}_channels"]:
                        combo.addItem(f"{device['name']} · {hosts[device['hostapi']]['name']}", i)
                index = combo.findText(previous)
                combo.setCurrentIndex(max(0, index))
            previous = self.route.currentText() or self.settings.get("route", "")
            self.route.clear()
            for index, name, host, receiver in virtual_outputs(devices, hosts):
                self.route.addItem(f"{name} · {host}", {"index": index, "name": name, "receiver": receiver})
            if not self.route.count():
                self.route.addItem("가상 케이블이 없습니다", None)
            self.route.setCurrentIndex(max(0, self.route.findText(previous)))
            self.update_route_hint()
        except Exception as exc:
            self.say(f"오디오 장치를 읽지 못했습니다: {exc}")

    def refresh_takes(self):
        if not hasattr(self, "take_list"):
            return
        self.take_list.clear()
        totals = []
        for kind, label in (("voice", "목소리"), ("noise", "소음")):
            seconds, count = 0, 0
            for path in sorted((paths.TAKES / kind).glob("*")):
                if path.suffix.lower() not in (".wav", ".flac"):
                    continue
                try:
                    info = sf.info(str(path))
                    duration = info.duration
                except Exception:
                    continue
                seconds += duration
                count += 1
                if kind == self.kind.currentData():
                    self.take_list.addItem(f"{path.name}   ·   {duration:.1f}초")
                    self.take_list.item(self.take_list.count() - 1).setData(Qt.ItemDataRole.UserRole, path)
            totals.append(f"{label} {count}개 / {seconds:.0f}초")
        self.totals.setText("     ·     ".join(totals))

    def refresh_model(self):
        dll = paths.engine_dll()
        self.model_status.setText(f"엔진 {'준비됨' if dll else '없음 — DLL 빌드가 필요합니다'}  ·  "
                                  f"개인 모델 {'준비됨' if paths.MODEL.is_file() else '학습 전'}  ·  로컬 저장")
        self.env_status.setText("학습 환경 준비됨" if self.training_ready() else "학습 환경을 설치해 주세요. Python 3.10 또는 3.11이 필요합니다.")
        self.metrics.setText("학습이 끝나면 기본 모델과의 비교 결과가 여기에 표시됩니다.")
        if paths.MODEL_INFO.is_file():
            try:
                info = json.loads(paths.MODEL_INFO.read_text(encoding="utf-8"))
                lines = [f"최근 학습: {info.get('trained_at', '—')} · {info.get('steps', '—')}회"]
                for name, values in info.get("compare", {}).get("voice_cut", {}).items():
                    lines.append(f"{name} · 음성 프레임 감소 비율: 기본 {values['public']:.1%} → 개인 {values['personal']:.1%}")
                for name, values in info.get("compare", {}).get("noise_db", {}).items():
                    lines.append(f"{name} · 소음 레벨 변화: 기본 {values['public']:.1f} → 개인 {values['personal']:.1f} dB")
                self.metrics.setText("\n".join(lines))
            except (ValueError, OSError, KeyError, TypeError):
                self.metrics.setText("모델은 있지만 학습 보고서를 읽을 수 없습니다.")

    def start_job(self, mode, **kwargs):
        output_device = kwargs.pop("output_device", self.output.currentData())
        self.job = AudioJob(mode, input_device=self.input.currentData(), output_device=output_device, parent=self, **kwargs)
        self.busy(True)
        self.job.message.connect(self.say)
        self.job.failed.connect(self.error)
        self.job.level.connect(self.show_level)
        self.job.result.connect(self.job_result)
        self.job.live_stats.connect(self.show_live_stats)
        self.job.live_ready.connect(self.on_live_ready)
        self.job.finished.connect(self.job_finished)
        self.job.start()

    def show_level(self, peak, seconds):
        db = 20 * np.log10(max(peak, 1e-6))
        self.meter.setValue(round(max(0, min(100, (db + 60) / 60 * 100))))
        self.meter.setFormat(f"{seconds:.1f}초 · {db:.0f} dBFS" + (" · 입력이 너무 큽니다" if peak >= .99 else ""))

    def job_result(self, result):
        if isinstance(result, Path):
            self.say(f"녹음 저장됨: {result.name}")
            self.refresh_takes()
        else:
            self.rendered = result
            self.pending_audio = result
            self.say("처리 완료 · 결과를 재생합니다.")

    def job_finished(self):
        old, self.job = self.job, None
        if getattr(old, "mode", None) == "route":
            self.live_active = False
            if not old.error_message:
                self.say("노이즈 제거가 꺼졌습니다. 가상 케이블 전송이 중단되었습니다.")
        old.deleteLater()
        self.busy(False)
        if self.closing:
            self.close()
        elif self.pending_audio is not None:
            audio, self.pending_audio = self.pending_audio, None
            self.start_job("play", source=audio)

    def record(self):
        self.say("녹음 중 · 중지를 누르면 저장됩니다. 최대 10분까지 녹음합니다.")
        self.start_job("record", folder=paths.TAKES / self.kind.currentData())

    def import_takes(self):
        files, _ = QFileDialog.getOpenFileNames(self, "학습용 녹음 가져오기", "", "Audio (*.wav *.flac)")
        try:
            for filename in files:
                audio = read_audio(Path(filename))
                if len(audio) < RATE * 2:
                    raise ValueError(f"2초 이상인 파일이 필요합니다: {Path(filename).name}")
                destination = paths.TAKES / self.kind.currentData() / f"{Path(filename).stem}_{uuid.uuid4().hex[:6]}.wav"
                sf.write(str(destination), audio, RATE, subtype="FLOAT")
            self.refresh_takes()
        except Exception as exc:
            self.refresh_takes()
            self.error(str(exc))

    def selected_take(self):
        item = self.take_list.currentItem()
        if not item:
            raise ValueError("녹음 파일을 먼저 선택해 주세요.")
        return item.data(Qt.ItemDataRole.UserRole)

    def play_take(self):
        try:
            self.start_job("play", source=self.selected_take())
        except Exception as exc:
            self.error(str(exc))

    def exclude_take(self):
        try:
            source = self.selected_take()
            excluded = paths.DATA / "excluded"
            excluded.mkdir(exist_ok=True)
            shutil.move(str(source), str(excluded / f"{uuid.uuid4().hex[:6]}_{source.name}"))
            self.refresh_takes()
            self.say("학습에서 제외했습니다. 파일은 저장 폴더의 excluded에 보관됩니다.")
        except Exception as exc:
            self.error(str(exc))

    def selected_model(self):
        if not paths.engine_dll():
            raise ValueError("pdn_engine.dll이 없습니다. 엔진이 포함된 앱을 실행해 주세요.")
        if self.model.currentData() == "personal":
            if not paths.MODEL.is_file():
                raise ValueError("개인 모델을 먼저 학습하거나 가져와 주세요.")
            return paths.MODEL
        return None

    def choose_sample(self):
        filename, _ = QFileDialog.getOpenFileName(self, "비교할 파일", str(paths.TAKES), "Audio (*.wav *.flac)")
        if filename:
            self.sample = Path(filename)
            self.rendered = None
            self.sample_label.setText(filename)

    def play_original(self):
        if self.sample:
            self.start_job("play", source=self.sample)
        else:
            self.say("먼저 비교 파일을 선택해 주세요.")

    def render(self):
        try:
            if not self.sample:
                raise ValueError("먼저 비교 파일을 선택해 주세요.")
            model = self.selected_model()
            self.say("선택 모델로 처리 중…")
            self.start_job("render", source=self.sample, dll=paths.engine_dll(), model=model, reduction=self.reduction.value())
        except Exception as exc:
            self.error(str(exc))

    def save_rendered(self):
        if self.rendered is None:
            self.say("먼저 선택 모델로 파일을 처리해 주세요.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "처리 결과 저장", "denoised.wav", "WAV (*.wav)")
        if filename:
            try:
                sf.write(filename, self.rendered, RATE, subtype="FLOAT")
                self.say("처리 결과를 저장했습니다.")
            except Exception as exc:
                self.error(str(exc))

    def monitor(self):
        try:
            model = self.selected_model()
            self.start_job("monitor", dll=paths.engine_dll(), model=model, reduction=self.reduction.value())
        except Exception as exc:
            self.error(str(exc))

    def import_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "개인 모델 가져오기", "", "Model (*.tar.gz)")
        if not filename:
            return
        # Validate in a worker so model loading cannot block the GUI.
        if not paths.engine_dll():
            self.error("모델 검증에 필요한 엔진 DLL이 없습니다.")
            return
        from app.model_job import ModelImport
        self.busy(True)
        self.job = ModelImport(Path(filename), paths.MODEL, paths.engine_dll(), self)
        self.job.message.connect(self.say)
        self.job.failed.connect(self.error)
        self.job.finished.connect(self.refresh_model)
        self.job.finished.connect(self.job_finished)
        self.job.start()

    def export_model(self):
        if not paths.MODEL.is_file():
            self.say("개인 모델이 아직 없습니다.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "개인 모델 내보내기", "model.tar.gz", "Model (*.tar.gz)")
        if filename:
            try:
                if Path(filename).resolve() != paths.MODEL.resolve():
                    shutil.copy2(paths.MODEL, filename)
                self.say("모델을 내보냈습니다. VST3 / CLAP 플러그인에서도 사용할 수 있습니다.")
            except Exception as exc:
                self.error(str(exc))

    def training_python(self):
        return paths.ENV / "Scripts" / "python.exe"

    def training_ready(self):
        return self.training_python().is_file() and (paths.ENV / "ready.json").is_file()

    def install_environment(self):
        if paths.FROZEN:
            launcher = shutil.which("py")
            if not launcher:
                self.error("Python 3.10 또는 3.11을 먼저 설치해 주세요 (Python Launcher 포함).")
                return
            prefix = None
            for version in ("-3.11", "-3.10"):
                try:
                    result = subprocess.run([launcher, version, "-c", "import sys; print(sys.executable)"],
                                            capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                    if result.returncode == 0:
                        prefix = [launcher, version]
                        break
                except (OSError, subprocess.TimeoutExpired):
                    continue
            if prefix is None:
                self.error("Python 3.10 또는 3.11을 설치해 주세요.")
                return
        else:
            prefix = [sys.executable]
        python = str(self.training_python())
        index = "https://download.pytorch.org/whl/cu128" if self.cuda.isChecked() else "https://download.pytorch.org/whl/cpu"
        (paths.ENV / "ready.json").unlink(missing_ok=True)
        self.commands = [prefix + ["-m", "venv", str(paths.ENV)],
                         [python, "-m", "pip", "install", "--upgrade", "pip"],
                         [python, "-m", "pip", "install", "torch==2.11.0", "torchaudio==2.11.0", "--index-url", index],
                         [python, "-m", "pip", "install", "-r", str(paths.TRAINING_SCRIPT.with_name("requirements.txt"))],
                         [python, "-m", "pip", "install", "--no-deps", "deepfilternet==0.5.6"],
                         [python, str(paths.TRAINING_SCRIPT), "--check-env"]]
        self.begin_process("install")

    def train(self):
        if not self.training_ready():
            self.error("먼저 학습 환경을 설치해 주세요.")
            return
        if not paths.engine_dll():
            self.error("학습한 모델을 검증할 엔진 DLL이 없습니다.")
            return
        for kind in ("voice", "noise"):
            if not any(p.suffix.lower() in (".wav", ".flac") for p in (paths.TAKES / kind).iterdir()):
                self.error("목소리와 소음 녹음이 각각 하나 이상 필요합니다.")
                return
        self.commands = [[str(self.training_python()), "-u", str(paths.TRAINING_SCRIPT),
                          "--takes", str(paths.TAKES), "--out", str(paths.DATA), "--cache", str(paths.CACHE),
                          "--engine", str(paths.engine_dll()), "--steps", str(self.steps.value()), "--batch", str(self.batch.value())]]
        self.begin_process("train")

    def begin_process(self, kind):
        self.process_kind = kind
        self.process_cancelled = self.process_error = False
        self.progress.setRange(0, 0)
        self.busy(True)
        self.log.clear()
        self.say("학습 환경 설치 중…" if kind == "install" else "학습 준비 중…")
        self.next_command()

    def next_command(self):
        command = self.commands.pop(0)
        self.process_buffer = ""
        process = QProcess(self)
        self.process = process
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self.read_process)
        process.finished.connect(self.process_finished)
        process.errorOccurred.connect(self.process_failed)
        process.start(command[0], command[1:])

    def read_process(self):
        self.process_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = self.process_buffer.split("\n")
        self.process_buffer = lines.pop()
        for line in lines:
            self.process_line(line.rstrip())

    def process_line(self, line):
        self.log.appendPlainText(line)
        logging.info(line)
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        stage = event.get("stage")
        names = {"prepare": "학습 준비", "download": "기본 모델 다운로드", "train": "학습 중", "export": "모델 변환 · 검증", "compare": "기본 모델과 비교 중", "done": "학습 완료"}
        if stage in names:
            self.say(names[stage] + (f" · {event['done']} / {event['total']}" if event.get("total") else ""))
        if stage == "train":
            self.progress.setRange(0, event["total"])
            self.progress.setValue(event["done"])
        elif stage in ("export", "compare"):
            self.progress.setRange(0, 0)
        elif stage == "error":
            self.process_error = True
            self.say("학습 오류: " + event.get("message", "로그를 확인해 주세요."))

    def process_failed(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.process_error = True
            self.say("실행 실패: " + self.process.errorString())
            self.commands.clear()
            self.finish_process()

    def process_finished(self, code, status):
        self.read_process()
        if self.process_buffer:
            self.process_line(self.process_buffer)
            self.process_buffer = ""
        if self.process_cancelled:
            self.say("작업을 중지했습니다.")
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            self.process_error = True
            self.say("작업이 실패했습니다. 아래 로그 또는 저장 폴더의 app.log를 확인해 주세요.")
        elif self.commands:
            self.process.deleteLater()
            self.next_command()
            return
        elif not self.process_error:
            if self.process_kind == "install":
                (paths.ENV / "ready.json").write_text(json.dumps({"installed": datetime.now().isoformat()}), encoding="utf-8")
                self.say("학습 환경 설치 완료. 이제 내 모델을 학습할 수 있습니다.")
            else:
                self.say("학습 완료. 마이크 노이즈 제거 탭에서 내 개인 모델을 사용할 수 있습니다.")
                self.model.setCurrentIndex(1)
        self.finish_process()

    def finish_process(self):
        if self.process:
            self.process.deleteLater()
            self.process = None
        self.commands.clear()
        self.progress.setRange(0, 100)
        self.progress.setValue(0 if self.process_cancelled or self.process_error else 100)
        self.busy(False)
        self.refresh_model()
        if self.closing:
            self.close()

    def stop(self):
        self.pending_audio = None
        if self.job:
            self.job.stop()
        if self.process:
            self.process_cancelled = True
            self.commands.clear()
            # pip and venv can have child processes; terminate the whole owned tree.
            pid = self.process.processId()
            if pid and os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                                   timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
            else:
                self.process.kill()
        self.say("중지하는 중…")

    def closeEvent(self, event: QCloseEvent):
        if not self.closing and self.keep_running.isChecked() and self.job and getattr(self.job, "mode", None) == "route" and QSystemTrayIcon.isSystemTrayAvailable():
            self.save_settings()
            self.hide()
            self.tray.showMessage("Personal Denoiser", "노이즈 제거를 계속 실행합니다. 트레이 아이콘에서 다시 열거나 종료할 수 있습니다.")
            event.ignore()
            return
        if self.job or self.process:
            self.closing = True
            self.stop()
            event.ignore()
            return
        self.save_settings()
        self.tray.hide()
        event.accept()

    def save_settings(self):
        try:
            paths.SETTINGS.write_text(json.dumps({"input": self.input.currentText(), "output": self.output.currentText(),
                                                  "model": self.model.currentData(), "reduction": self.reduction.value(),
                                                  "route": self.route.currentText(), "keep_running": self.keep_running.isChecked(),
                                                  "auto_start": self.auto_start.isChecked()}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def main():
    paths.DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=paths.LOG, level=logging.INFO, encoding="utf-8", format="%(asctime)s %(message)s")
    app = QApplication(sys.argv)
    lock = QLockFile(str(paths.DATA / "app.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None, "Personal Denoiser", "이미 실행 중입니다. 열려 있는 창을 확인하거나 작업 표시줄 트레이의 마이크 아이콘을 더블클릭해 주세요.")
        return
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(1200, window.close)
    sys.exit(app.exec())
