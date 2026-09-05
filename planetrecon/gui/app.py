"""Qt6 raster shell: open capture, run/cancel, progressive image and progress."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from planetrecon.jobs import JobHandle, start_stack_job
from planetrecon.reconstruction import ReconstructionConfig
from planetrecon.runtime import apply_thread_limits


def _to_qimage(image: np.ndarray):
    from PySide6.QtGui import QImage

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        plane = arr
        vis = np.stack([plane, plane, plane], axis=2)
    else:
        vis = arr[..., :3]
    finite = vis[np.isfinite(vis)]
    lo = float(np.percentile(finite, 1)) if finite.size else 0.0
    hi = float(np.percentile(finite, 99)) if finite.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((vis - lo) / (hi - lo), 0.0, 1.0)
    rgb8 = (scaled * 255.0).astype(np.uint8)
    rgb8 = np.ascontiguousarray(rgb8)
    h, w, _ = rgb8.shape
    return QImage(rgb8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def create_app(argv=None):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else argv)
    return app


class MainWindow:
    def __init__(self, path: Path | None = None, config: ReconstructionConfig | None = None):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import (
            QComboBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        self.config = config or ReconstructionConfig(device="auto")
        self.job: JobHandle | None = None
        owner = self

        class OwnedWindow(QMainWindow):
            def closeEvent(self, event):
                owner._shutdown()
                super().closeEvent(event)

        self.window = OwnedWindow()
        self.window.setWindowTitle("PlanetRecon")
        root = QWidget()
        layout = QVBoxLayout(root)
        self.image_label = QLabel("Open a SER or Gate-1 HDF5 capture")
        self.image_label.setMinimumSize(256, 256)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("idle")
        buttons = QHBoxLayout()
        self.open_btn = QPushButton("Open")
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "gpu"])
        self.device.setCurrentText(self.config.device)
        buttons.addWidget(self.open_btn)
        buttons.addWidget(self.run_btn)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(QLabel("Device"))
        buttons.addWidget(self.device)
        layout.addLayout(buttons)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        self.window.setCentralWidget(root)
        self.path = Path(path) if path else None
        self.open_btn.clicked.connect(self._choose)
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn.clicked.connect(self._cancel)
        self.timer = QTimer(self.window)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._poll)
        if self.path is not None:
            self.status.setText(str(self.path))

    def show(self) -> None:
        self.window.show()

    def _choose(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        name, _ = QFileDialog.getOpenFileName(
            self.window, "Open capture", "", "Captures (*.ser *.h5 *.hdf5)"
        )
        if name:
            self.path = Path(name)
            self.status.setText(str(self.path))

    def _run(self) -> None:
        if self.path is None:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(self.window, "PlanetRecon", "Open a capture first.")
            return
        if self.job is not None:
            self.job.close()
        cfg = replace(self.config, device=self.device.currentText())
        self.job = start_stack_job(self.path, cfg)
        self.status.setText(f"running {self.job.job_id} on {cfg.device}")
        self.progress.setValue(0)
        self.timer.start()

    def _cancel(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.job.close()
            self.job = None
            self.status.setText("cancelled")
            self.timer.stop()

    def _poll(self) -> None:
        if self.job is None:
            return
        for event in self.job.poll(timeout=0.0):
            if event.kind == "progress":
                frac = float(event.payload.get("fraction", 0.0))
                self.progress.setValue(int(round(100.0 * min(max(frac, 0.0), 1.0))))
                self.status.setText(
                    f"{event.payload.get('stage', '')} "
                    f"{event.payload.get('backend', '')} "
                    f"n={event.payload.get('n_used', '')}"
                )
            elif event.kind == "preview":
                image = event.payload.get("image")
                if image is not None:
                    self.image_label.setPixmap(_pixmap(image))
                warns = event.payload.get("warnings") or []
                if warns:
                    self.status.setText("; ".join(str(w) for w in warns[:2]))
            elif event.kind in ("completed", "cancelled", "error"):
                if event.kind == "completed":
                    self.progress.setValue(100)
                    self.status.setText(
                        f"done backend={event.payload.get('backend')} n={event.payload.get('n_used')}"
                    )
                    image = event.payload.get("image")
                    if image is not None:
                        self.image_label.setPixmap(_pixmap(image))
                elif event.kind == "error":
                    self.status.setText(event.payload.get("message", "error"))
                self.timer.stop()
                self.job.close()
                self.job = None
                break
        if self.job is not None and self.job.state == "failed":
            self.status.setText(f"worker exited with code {self.job.process.exitcode}")
            self._shutdown()

    def _shutdown(self, *_args) -> None:
        self.timer.stop()
        if self.job is not None:
            self.job.close()
            self.job = None


def _pixmap(image: np.ndarray):
    from PySide6.QtGui import QPixmap

    return QPixmap.fromImage(_to_qimage(image))


def main(argv: list[str] | None = None) -> int:
    apply_thread_limits()
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else None
    app = create_app(args)
    win = MainWindow(path=path)
    app.aboutToQuit.connect(win._shutdown)
    win.show()
    return app.exec()
