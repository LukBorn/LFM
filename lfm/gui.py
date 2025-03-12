import sys
from ast import literal_eval
import warnings
import yaml
import numpy as np
import pathlib
import traceback
import logging
logger = logging.getLogger(__name__)

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree, registerParameterType

from opm_acquire.control import OPM


class OtherParameterItem(pg.parametertree.parameterTypes.WidgetParameterItem):
    """ParameterItem for strings that can be converted to python objects via ast.literal_eval"""

    def makeWidget(self):
        def setValue(v):
            w.setText(str(v))

        def getValue():
            return literal_eval(w.text())

        w = pg.Qt.QtWidgets.QLineEdit()
        w.setStyleSheet("border: 0px")
        w.sigChanged = w.editingFinished
        w.value = getValue
        w.setValue = setValue
        w.sigChanging = w.textChanged
        return w


class OtherParameter(Parameter):
    """Parameter for strings that can be converted to python objects via ast.literal_eval."""

    itemClass = OtherParameterItem


registerParameterType("other", OtherParameter, override=True)


class ParameterProxy:
    """
    Proxy for nested parameter access in a `Parameter` object.

    Attributes:
        param (Parameter): The current parameter object.
    """

    def __init__(self, param):
        self.param = param

    def __getitem__(self, key):
        child = self.param.child(key)
        if child.type() == "group":
            return ParameterProxy(child)
        return child.value()

    def __setitem__(self, key, value):
        child = self.param.child(key)
        if child.type() == "group":
            raise ValueError("Cannot directly set value to a group parameter")
        child.setValue(value)

    def _ipython_key_completions_(self):
        return [c.name() for c in self.param.children()]


class DictInspector(ParameterTree):
    """
    A custom ParameterTree for inspecting dictionaries with PyQtGraph.

    Attributes:
        p (Parameter): The root parameter object.
    """

    def __init__(self, params_dict, title="Parameter Inspector"):
        """
        Initializes the DictInspector with the provided dictionary.

        Args:
            params_dict (dict): The dictionary to be converted into a ParameterTree.
        """
        super().__init__()
        self.p = Parameter.create(name="params", type="group", children=self.dict_to_params(params_dict))
        self.setParameters(self.p, showTop=False)
        self.setWindowTitle(title)
        self.resize(350, 800)
        self.header().setMinimumSectionSize(150)
        self.show()

    def dict_to_params(self, d):
        """Convert a dictionary to a list of parameters suitable for Parameter.create().

        Args:
            d (dict): The dictionary to be converted.

        Returns:
            list: List of parameters.
        """
        params_list = []
        for key, value in d.items():
            if isinstance(value, dict):  # If the value is a dict, create a subgroup
                params_list.append({"name": key, "type": "group", "children": self.dict_to_params(value)})
            else:
                if isinstance(value, str):
                    param_type = "str"
                    if key.endswith("file"):
                        param_type = "file"
                    if key.endswith("text"):
                        param_type = "text"
                elif isinstance(value, bool):
                    param_type = "bool"
                elif isinstance(value, int):
                    param_type = "int"
                elif isinstance(value, float):
                    param_type = "float"
                    if key.endswith("time") or key.endswith("duration"):
                        params_list.append(
                            {
                                "name": key,
                                "type": param_type,
                                "value": value,
                                "default": None,
                                "suffix": "s",
                                "siPrefix": True,
                            }
                        )
                        continue
                    if key.endswith("progress"):
                        param_type = "progress"
                elif callable(value):
                    param = Parameter.create(name=key, type="action", value=self.wrapper(value))
                    param.sigActivated.connect(param.opts["value"])
                    params_list.append(param)
                    continue
                    # param_type = 'action'
                elif isinstance(value, (list, np.ndarray)):
                    param_type = "other"
                else:
                    warnings.warn(f"{key} has incompatible type {type(value)}. skipping...")
                    continue
                params_list.append({"name": key, "type": param_type, "value": value, "default": None})
        return params_list

    def wrapper(self, f):
        return lambda _: f(self.to_dict())

    def __getitem__(self, key):
        """
        Retrieve the child parameter or value associated with the key.

        Args:
            key (str): The key/name of the child parameter.

        Returns:
            ParameterProxy: Proxy object facilitating further nested access.
        """
        child = self.p.child(key)
        if child.type() == "group":
            return ParameterProxy(self.p.child(key))
        else:
            return child.value()

    def __setitem__(self, key, value):
        """
        Set the value of the child parameter associated with the key.

        Args:
            key (str): The key/name of the child parameter.
            value (Any): The value to set.

        Raises:
            ValueError: If trying to set value directly to a group parameter.
        """
        child = self.p.child(key)
        if child.type() == "group":
            raise ValueError("Cannot directly set value to a group parameter")
        child.setValue(value)

    def _ipython_key_completions_(self):
        return di.p.names.keys()

    def to_dict(self, include_callables=False):
        def params_to_dict(param):
            data = {}

            for child in param.children():
                if child.type() == "group":
                    data[child.name()] = params_to_dict(child)
                else:
                    val = child.value()
                    if not callable(val) or include_callables:
                        data[child.name()] = val
            return data

        return params_to_dict(self.p)

    def to_yaml(self, fn):
        d = self.to_dict()
        with open(fn, "w") as file:
            yaml.dump(d, file)

    def __del__(self):
        self.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OPM GUI')
    parser.add_argument('--config', help='Config file name', default='gui_defaults.yml')
    args = parser.parse_args()
    config_fn = args.config

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info("Starting up")

    app = pg.Qt.QtWidgets.QApplication(sys.argv)

    def setOptsRecursive(p, **kwargs):
        """Set the options of a parameter and all its children recursively."""
        for ch in p.children():
            setOptsRecursive(ch, **kwargs)
        if len(p.children()) == 0:
            p.setOpts(**kwargs)

    def button_callback(name):
        """Callback function for buttons in the GUI."""
        opm.cam.set_decentered_roi(
            di["camera"]["ROI"]["y_size"],
            di["camera"]["ROI"]["x_size"],
            di["camera"]["ROI"]["y_decenter"],
            di["camera"]["ROI"]["x_decenter"],
            di["camera"]["ROI"]["y_bin"],
            di["camera"]["ROI"]["x_bin"],
        )
        opm.cam.exposure_time = di["camera"]["exposure_ms"] / 1000
        setOptsRecursive(di.p, enabled=False)
        if name == "preview":
            opm.start_preview(di.to_dict())
        # elif name == "preview_test":
        #     opm.cam.set_trigger(False)
        #     threaded(opm.start_ortho_preview)(di.to_dict())
        elif name == "ortho_view":
            opm.start_ortho_preview(di.to_dict())
        elif name == "grab":
            try:
                opm.acquire_timelapse(di.to_dict())
            except: 
                logger.error('Error during acquisition')
                logger.error(traceback.format_exc())
                opm.point()
        elif name == "abort":
            opm.interrupt_flag = True
        elif name == "mosaic":
            opm.cam.set_trigger(False)
            opm.cam.preview_mosaic([100, 100], [50, 50])
        elif name == "beam_array":
            opm.start_beam_array(di.to_dict())
        elif name == "point":
            opm.point()
        else:
            print(f"did not understand {name}")
        setOptsRecursive(di.p, enabled=True)

    # Load default parameters from file
    with open(f"{pathlib.Path(__file__).parent.resolve()}/config/{config_fn}", "r") as file:
        params_dict = yaml.safe_load(file)

    # Add callback functions to buttons
    params_dict["acquisition"].update(
        dict(
            preview=lambda d: button_callback("preview"),
            ortho_view=lambda d: button_callback("ortho_view"),
            grab=lambda d: button_callback("grab"),
            # preview_test=lambda d: button_callback("preview_test"),
        )
    )
    params_dict["alignment"].update(
        dict(
            mosaic=lambda d: button_callback("mosaic"),
            beam_array=lambda d: button_callback("beam_array"),
            point=lambda d: button_callback("point"),
        )
    )

    # Initialize OPM object
    logger.info("Initializing OPM object")
    opm = OPM(
        rate=params_dict["hardware"]["daq"]["rate"],
        aochans=f"{params_dict['hardware']['daq']['device']}/{params_dict['hardware']['daq']['ao_channels']}",
        cam_trigger_line=params_dict["hardware"]["daq"]["do_port"],
        shutter_line=params_dict["hardware"]["daq"]["shutter_line"],
        cam_gain=params_dict["hardware"]["camera"]["gain"],
    )

    # Initialize GUI
    logger.info("Initializing GUI")
    di = DictInspector(params_dict, "OPM")
    di.move(0, 0)
    di.p.child("acquisition").child("stimulus_file").sigValueChanged.connect(
        lambda p, fn: opm.load_stimdata(p, fn, di.to_dict())
    )
    di.p.child("camera").child("ROI").setOpts(expanded=False)
    di.p.child("alignment").setOpts(expanded=False)
    di.p.child("hardware").setOpts(expanded=False)

    # Start Qt event loop.
    logger.info("Startup complete")
    app.exec()
    logger.info("GUI closed")
    del opm


# class Worker(pg.Qt.QtCore.QRunnable):
#     """Worker thread
#     Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
#     """

#     def __init__(self, fn, *args, **kwargs):
#         super(Worker, self).__init__()
#         self.fn = fn
#         self.args = args
#         self.kwargs = kwargs

#     @pg.Qt.QtCore.pyqtSlot()
#     def run(self):
#         self.fn(*self.args, **self.kwargs)


# def threaded(fn):
#     def wrapped(*args, **kwargs):
#         worker = Worker(fn, *args, **kwargs)
#         pool = pg.Qt.QtCore.QThreadPool.globalInstance()
#         pool.start(worker)
#         # pool.waitForDone()

#     return wrapped
