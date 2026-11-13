from __future__ import annotations

import csv
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from Core.detector import DroneDetector
from Core.tracker import DroneTracker

if TYPE_CHECKING:
    from TracingUI import Ui_MainWindow


class _VideoPanelResizeFilter(QtCore.QObject):
    def __init__(self, controller: TrackingUIController):
        super().__init__(controller.main_window)
        self.controller = controller

    def eventFilter(self, watched, event):
        if (
            watched is self.controller.ui.videoPanel
            and event.type() == QtCore.QEvent.Resize
        ):
            self.controller._fit_video()
        return False


class TrackingUIController:
    def __init__(
        self,
        main_window: QtWidgets.QMainWindow,
        ui: Ui_MainWindow,
        model_path: Path,
        default_video_path: Path | None = None,
    ):
        self.main_window = main_window
        self.ui = ui
        self.model_path = model_path
        self.capture: cv2.VideoCapture | None = None
        self.detector: DroneDetector | None = None
        self.tracker = DroneTracker(max_lost_frames=30)
        self.frame_interval_ms = 33
        self.source_fps = 30.0
        self.processed_frame_count = 0
        self.video_writer: cv2.VideoWriter | None = None
        self.tracking_data_file = None
        self.csv_writer = None
        self.export_directory = tempfile.TemporaryDirectory(
            prefix="drone_tracking_"
        )
        self.processed_video_path = (
            Path(self.export_directory.name) / "processed_video.mp4"
        )
        self.tracking_data_path = (
            Path(self.export_directory.name) / "tracking_data.csv"
        )

        self.scene = QtWidgets.QGraphicsScene(main_window)
        self.frame_item = self.scene.addPixmap(QtGui.QPixmap())
        self.ui.videoPanel.setScene(self.scene)
        self.ui.videoPanel.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.videoPanel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ui.videoPanel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.resize_filter = _VideoPanelResizeFilter(self)
        self.ui.videoPanel.installEventFilter(self.resize_filter)

        self.timer = QtCore.QTimer(main_window)
        self.timer.timeout.connect(self._display_next_frame)

        self.ui.thresholdSetup.setRange(0.0, 1.0)
        self.ui.thresholdSetup.setDecimals(2)
        self.ui.thresholdSetup.setSingleStep(0.05)
        self.ui.thresholdSetup.setValue(0.4)
        self.ui.exportVideoButton.clicked.connect(self.export_processed_video)
        self.ui.exportDataButton.clicked.connect(self.export_tracking_data)
        self.ui.pushButton.clicked.connect(self.select_video)
        self.ui.startButton.clicked.connect(self.start_playback)
        self.ui.pauseButton.clicked.connect(self.pause_playback)
        self.ui.replayButton.clicked.connect(self.replay_video)
        self.ui.thresholdSetup.valueChanged.connect(self._set_threshold)
        self.ui.startButton.setEnabled(False)
        self.ui.pauseButton.setEnabled(False)
        self.ui.replayButton.setEnabled(False)
        self.ui.exportVideoButton.setEnabled(False)
        self.ui.exportDataButton.setEnabled(False)
        self.ui.secilenVideo.setText("Video seçilmedi")
        self.ui.activeTargetsValue.setText("0")
        self.ui.fpsDegeri.setText("0.0")

        if default_video_path is not None and default_video_path.is_file():
            self.load_video(default_video_path)

    def select_video(self):
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_window,
            "Video seçin",
            str(self.model_path.parent),
            "Video dosyaları (*.mp4 *.avi *.mov *.mkv);;Tüm dosyalar (*)",
        )
        if selected_path:
            self.load_video(Path(selected_path))

    def load_video(self, video_path: Path) -> bool:
        new_capture = cv2.VideoCapture(str(video_path))
        if not new_capture.isOpened():
            new_capture.release()
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "Video açılamadı",
                f"Video dosyası açılamadı:\n{video_path}",
            )
            return False

        ret, frame = new_capture.read()
        if not ret:
            new_capture.release()
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "Video açılamadı",
                f"Videodan kare okunamadı:\n{video_path}",
            )
            return False

        self.timer.stop()
        self._reset_recording()
        if self.capture is not None:
            self.capture.release()
        self.capture = new_capture
        self.tracker = DroneTracker(max_lost_frames=30)
        self.ui.secilenVideo.setText(video_path.name)
        self.ui.secilenVideo.setToolTip(str(video_path))

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if fps > 0 else 30.0
        self.frame_interval_ms = max(1, round(1000 / self.source_fps))
        self.ui.activeTargetsValue.setText("0")
        self.ui.fpsDegeri.setText("0.0")

        self._render_frame(frame)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.ui.startButton.setEnabled(True)
        self.ui.pauseButton.setEnabled(False)
        self.ui.replayButton.setEnabled(True)
        self._set_status(f"Video hazır: {video_path.name}")
        return True

    def start_playback(self):
        if self.capture is None:
            QtWidgets.QMessageBox.information(
                self.main_window, "Video seçilmedi", "Önce bir video dosyası seçin."
            )
            return

        if self.detector is None:
            try:
                if not self.model_path.is_file():
                    raise FileNotFoundError(
                        f"Model dosyası bulunamadı: {self.model_path}"
                    )
                self.detector = DroneDetector(
                    model_path=str(self.model_path),
                    conf_threshold=self.ui.thresholdSetup.value(),
                )
            except (OSError, RuntimeError, ValueError) as error:
                QtWidgets.QMessageBox.critical(
                    self.main_window,
                    "Model başlatılamadı",
                    f"Takip modeli yüklenemedi:\n{error}",
                )
                self._set_status("Model başlatılamadı.")
                return

        self.timer.start(self.frame_interval_ms)
        self.ui.startButton.setEnabled(False)
        self.ui.pauseButton.setEnabled(True)
        self._set_status("Video oynatılıyor.")

    def pause_playback(self):
        self.timer.stop()
        self.ui.startButton.setEnabled(self.capture is not None)
        self.ui.pauseButton.setEnabled(False)
        self._set_status("Video duraklatıldı.")

    def replay_video(self):
        if self.capture is None:
            return
        self.timer.stop()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._reset_recording()
        self.tracker = DroneTracker(max_lost_frames=30)
        self.start_playback()

    def close(self):
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self._finish_recording()
        self.export_directory.cleanup()

    def _set_threshold(self, threshold: float):
        if self.detector is not None:
            self.detector.set_threshold(threshold)

    def _display_next_frame(self):
        if self.capture is None or self.detector is None:
            self.pause_playback()
            return

        ret, frame = self.capture.read()
        if not ret:
            self._finish_playback()
            return

        frame_started_at = time.perf_counter()
        try:
            tracks = self.tracker.update(self.detector.detect(frame))
        except (cv2.error, RuntimeError, ValueError) as error:
            self.timer.stop()
            self.ui.startButton.setEnabled(True)
            self.ui.pauseButton.setEnabled(False)
            self._finish_recording()
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "Takip hatası",
                f"Video karesi işlenemedi:\n{error}",
            )
            self._set_status("Video işlenirken hata oluştu.")
            return

        for bbox, track_id, state in tracks:
            x1, y1, x2, y2 = bbox
            if state == "TRACKING":
                color = (0, 255, 0)
            elif state in ("UNCERTAIN", "REACQUIRED"):
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID: {track_id} [{state}]",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        active_targets = sum(
            state in ("TRACKING", "REACQUIRED") for _, _, state in tracks
        )
        cv2.putText(
            frame,
            f"Aktif Hedef Sayisi: {active_targets}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        self.ui.activeTargetsValue.setText(str(active_targets))

        try:
            self._write_outputs(frame, tracks)
        except (OSError, cv2.error, RuntimeError, ValueError) as error:
            self.timer.stop()
            self.ui.startButton.setEnabled(True)
            self.ui.pauseButton.setEnabled(False)
            self._finish_recording()
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "Dışa aktarma hatası",
                f"İşlenmiş çıktı kaydedilemedi:\n{error}",
            )
            self._set_status("Çıktı kaydedilirken hata oluştu.")
            return

        elapsed = time.perf_counter() - frame_started_at
        if elapsed > 0:
            self.ui.fpsDegeri.setText(f"{1.0 / elapsed:.1f}")
        self._render_frame(frame)

    def _write_outputs(self, frame: np.ndarray, tracks: list):
        if self.video_writer is None:
            height, width = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                str(self.processed_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.source_fps,
                (width, height),
            )
            if not self.video_writer.isOpened():
                self.video_writer.release()
                self.video_writer = None
                raise RuntimeError("MP4 video yazıcısı başlatılamadı.")

        if self.tracking_data_file is None:
            self.tracking_data_file = self.tracking_data_path.open(
                "w", newline="", encoding="utf-8"
            )
            self.csv_writer = csv.writer(self.tracking_data_file)
            self.csv_writer.writerow(
                [
                    "frame",
                    "timestamp_seconds",
                    "track_id",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "state",
                ]
            )

        self.video_writer.write(frame)
        timestamp = self.processed_frame_count / self.source_fps
        for bbox, track_id, state in tracks:
            self.csv_writer.writerow(
                [self.processed_frame_count, f"{timestamp:.3f}", track_id, *bbox, state]
            )
        self.processed_frame_count += 1

    def _finish_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.tracking_data_file is not None:
            self.tracking_data_file.close()
            self.tracking_data_file = None
            self.csv_writer = None
        outputs_ready = self.processed_frame_count > 0
        self.ui.exportVideoButton.setEnabled(
            outputs_ready and self.processed_video_path.is_file()
        )
        self.ui.exportDataButton.setEnabled(
            outputs_ready and self.tracking_data_path.is_file()
        )

    def _finish_playback(self):
        self.timer.stop()
        self._finish_recording()
        self.ui.startButton.setEnabled(True)
        self.ui.pauseButton.setEnabled(False)
        self._set_status(
            "Video sona erdi. Çıktıları dışa aktarabilir veya baştan başlatabilirsiniz."
        )

    def _reset_recording(self):
        self._finish_recording()
        self.processed_frame_count = 0
        for output_path in (self.processed_video_path, self.tracking_data_path):
            if output_path.exists():
                output_path.unlink()
        self.ui.exportVideoButton.setEnabled(False)
        self.ui.exportDataButton.setEnabled(False)

    def export_processed_video(self):
        self._export_file(
            self.processed_video_path,
            "İşlenmiş videoyu kaydet",
            "MP4 video (*.mp4)",
        )

    def export_tracking_data(self):
        self._export_file(
            self.tracking_data_path,
            "Takip verisini kaydet",
            "CSV dosyası (*.csv)",
        )

    def _export_file(self, source_path: Path, title: str, file_filter: str):
        if not source_path.is_file():
            QtWidgets.QMessageBox.warning(
                self.main_window, "Çıktı bulunamadı", "Önce video takibini tamamlayın."
            )
            return
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.main_window,
            title,
            str(self.model_path.parent / source_path.name),
            file_filter,
        )
        if not selected_path:
            return
        destination = Path(selected_path)
        if source_path.resolve() == destination.resolve():
            QtWidgets.QMessageBox.warning(
                self.main_window,
                "Geçersiz hedef",
                "Geçici çıktı dosyasının üzerine yazılamaz.",
            )
            return
        try:
            shutil.copy2(source_path, destination)
        except OSError as error:
            QtWidgets.QMessageBox.critical(
                self.main_window,
                "Dışa aktarma hatası",
                f"Dosya kaydedilemedi:\n{error}",
            )
            return
        self._set_status(f"Dosya kaydedildi: {destination.name}")

    def _render_frame(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QtGui.QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QtGui.QImage.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.frame_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.frame_item.boundingRect())
        self._fit_video()

    def _fit_video(self):
        if not self.frame_item.pixmap().isNull():
            self.ui.videoPanel.fitInView(
                self.frame_item, QtCore.Qt.KeepAspectRatio
            )

    def _set_status(self, message: str):
        self.ui.statusbar.showMessage(message)
