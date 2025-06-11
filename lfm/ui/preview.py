from PyQt5 import QtWidgets
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import logging
import threading

logger = logging.getLogger(__name__)

class PreviewWindow(QtWidgets.QWidget):
    def __init__(self, filter_fcn=None, window_title="Preview", update_interval_ms=30, verbose=False):
        super().__init__()
        self.verbose = verbose
        if filter_fcn is None:
            def filter_fcn(im):
                return im[::2, ::2]
        self.filter_fcn = filter_fcn
        self.setWindowTitle(window_title)
        self.resize(900, 950)

        # Add a frame buffer to store the latest frame
        self.latest_frame = None
        self.frame_updated = False
        self.thread = None
        self.stop_flag = False

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.imv = pg.ImageView()
        self.imv.ui.menuBtn.hide()
        self.imv.ui.roiBtn.hide()
        layout.addWidget(self.imv)

        # Create a timer for UI updates
        self.update_timer = QTimer()
        self.update_timer.setInterval(update_interval_ms)
        self.update_timer.timeout.connect(self._refresh_display)

        # Connect the showEvent to start the timer
        self.showEvent = self._on_show
        # Override closeEvent to clean up
        self.closeEvent = self._on_close

        logger.debug(f"Preview window initialized")

    def _on_show(self, event):
        # Start the timer when the window is shown
        self.update_timer.start()
        if self.verbose:
            logger.debug("Update timer started.")
            logger.debug(f"Timer active: {self.update_timer.isActive()}")
        super().showEvent(event)

    def _on_close(self, event):
        logger.debug("Window closing, stopping timer and thread")
        self.update_timer.stop()
        self.stop_flag = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        super().closeEvent(event)

    def update(self, im):
        if self.verbose:
            logger.debug(f"Frame received with shape: {im.shape}, dtype: {im.dtype}")
        self.latest_frame = im
        self.frame_updated = True

    def _refresh_display(self):
        if self.verbose:
            logger.debug("Timer triggered _refresh_display.")
        if self.frame_updated and self.latest_frame is not None:
            filtered = self.filter_fcn(self.latest_frame)
            self.imv.setImage(filtered.T, levels=(0, 255), autoHistogramRange=False)
            self.frame_updated = False
        else:
            if self.verbose:
                logger.debug("No new frame to refresh.")
        QtWidgets.QApplication.processEvents()

    def is_stopping(self):
        """Check if the window is being closed"""
        return self.stop_flag or not self.isVisible()

class OldPreviewWindow(QtWidgets.QWidget):
    def __init__(self, filter_fcn=None, window_title="Preview"):
        super().__init__()
        if filter_fcn is None:
            def filter_fcn(im):
                return im[::2, ::2]
        self.filter_fcn = filter_fcn
        self.setWindowTitle(window_title)
        self.resize(900, 950)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.imv = pg.ImageView()
        self.imv.ui.menuBtn.hide()
        self.imv.ui.roiBtn.hide()
        layout.addWidget(self.imv)

        self.show()

    def update(self, im):
        filtered = self.filter_fcn(im)
        self.imv.setImage(filtered.T, levels=(0, 255), autoHistogramRange=False)
        QtWidgets.QApplication.processEvents()