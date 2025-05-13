
from PyQt5 import QtWidgets
import pyqtgraph as pg

class PreviewWindow(QtWidgets.QWidget):
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