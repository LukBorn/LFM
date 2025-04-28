import numpy as np
import os
import arrow
import json
import h5py
from scipy.signal import sawtooth
from contextlib import nullcontext
import logging
logger = logging.getLogger(__name__)

import pyqtgraph as pg
from tqdm.auto import tqdm

from nidaqmx.utils import flatten_channel_string, unflatten_channel_string


N_BACKGROUND_FRAMES = 100


class SimpleArrayProxy:
    """A simple array object proxy with custom shape and getitem function"""

    def __init__(self, custom_getitem, shape=None):
        self.shape = shape
        self._custom_getitem = custom_getitem

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        return self._custom_getitem(slices)

def get_full_waveforms(conf,preview = False, ft = 0.01):
    ''' Generate full waveforms for the entire recording, including ramping. Note this uses SimpleArrayProxy to avoid storing the entire array in memory.

    Args:
        conf (dict): Configuration dictionary.
        preview

    Returns:
        arraylike: Control AO voltages array.
        arraylike: Control DO voltages array.
        float: Updated frame time in ms.
    '''
    fps = conf['camera']['preview_fps'] if preview else conf['camera']['recording_fps']
    frame_time = 1/fps
    if ft is not None:
        frame_time = ft
        fps = int(1/ft)

    trigger_single = np.zeros((1,conf['daq']['rate']))
    for frame_start in range(0, conf['daq']['rate'], conf['daq']['rate']//fps):
        trigger_single[0,frame_start : frame_start + int((conf['camera']['cam_trig_width_ms'] / 1000) * conf['daq']['rate'])] = 1
    trigger_single = trigger_single.astype(bool)
    led_single = np.ones((1,conf['daq']['rate']))*conf['acquisition']['led_percent']*0.01*conf["hardware"]["led_control_v"]
    # if conf['acquisition']['ramp_seconds'] == 0:
    #     ramp = np.ones(1)
    # else:
        # ramp = np.geomspace(0.01, 1, conf['acquisition']['ramp_seconds'] + 1)
    ramp = (1 - np.exp(-4 * np.linspace(0, 1, int(conf['acquisition']['ramp_seconds'])+1)))

    def trigger_getter(slices):
        slices = list(slices)
        slices[1] = np.r_[slices[1]] % trigger_single.shape[1]
        return trigger_single.__getitem__(tuple(slices))

    def led_getter(slices):
        slices = list(slices)
        ix = np.minimum(np.r_[slices[1]] // led_single.shape[1], len(ramp)-1)
        slices[1] = np.r_[slices[1]] % led_single.shape[1]
        out = led_single.__getitem__(tuple(slices))
        out[0, :] *= ramp[ix]
        return out

    full_shape = [1, int(conf['daq']['rate'] * (conf['acquisition']['ramp_seconds'] + conf['acquisition']['recording_seconds']))]
    led_full = SimpleArrayProxy(led_getter, shape=full_shape)
    trigger_full = SimpleArrayProxy(trigger_getter, shape=full_shape)

    return led_full, trigger_full, led_single, trigger_single, frame_time

def get_preview_callback(im_shape, edge_sz=10, refresh_every=5, window_title='Preview'):
    """Return a callback function for generating an preview in an ImageView.
    
    Args:
        im_shape (tuple): Shape of the image.
        frames_per_vol (int): Number of frames per volume.
        edge_sz (int): Size of the edge in pixels. Defaults to 10.
        refresh_every (int): Refresh the image every n frames. Defaults to 30.

    Returns:
        function: Callback function.
        function: Interrupt function.
    """

    # Create a main window with layout
    main_window = pg.Qt.QtWidgets.QWidget()
    layout = pg.Qt.QtWidgets.QVBoxLayout()
    main_window.setLayout(layout)
    main_window.setWindowTitle("PSF Preview with Z Position")
    main_window.resize(900, 950)  # Adjust dimensions for both image and label
    main_window.show()

    # Create the ImageView and add it to the layout
    imv = pg.ImageView()
    imv.ui.menuBtn.hide()  # Hide unnecessary buttons
    imv.ui.roiBtn.hide()
    layout.addWidget(imv)  # Add ImageView to the layout

    # Add a QLabel for displaying the relative Z position
    z_label = pg.Qt.QtWidgets.QLabel()
    layout.addWidget(z_label)

    def callback(im_np, i_frame, timestamp, frame_count):
        """Updates the displayed image every 'refresh_every' frames."""
        if i_frame % refresh_every == 0:
            imv.setImage(im_np[::2,::2], levels=(0, 255), autoHistogramRange=False)
            pg.Qt.QtWidgets.QApplication.processEvents()

    return callback, imv


from cams import DCamera as Camera
from daq import unifiedDAO
from write import ParallelCompressedWriter, VanillaWriter



class LFM:
    """A class for controlling the DAQ and camera for LFM"""
    def __init__(self, rate=50000, aochans='Dev1/ao0:2', cam_trigger_line='port0/line0', shutter_line='PFI0', cam_gain=1):
        """Initialize OPM object.
        """
        logger.info('Initializing DAQ')
        self.dao = unifiedDAO(rate, aochans, cam_trigger_line, shutter_line)
        self.stim_dao = None
        self.shutter_open = self.dao.shutter_open
        self.shutter_close = self.dao.shutter_close
        logger.info('Initializing Camera')
        self.cam = Camera()
        self.point()
        self.interrupt_flag = False

    def point(self):
        """
        Set output voltages to zero
        """
        chans = unflatten_channel_string(self.dao.ao_chans)
        self.dao.queue_data(np.zeros((len(chans), 2)), finite=True, chunked=False)
        self.dao.task_ao.start()
        self.dao.task_ao.wait_until_done()
        try:
            self.dao.close()  # Attempt to clean up, but handle warnings gracefully
        except Exception as e:
            logger.warning(f"Attempted to close an already-closed task: {e}")
        logger.info('point')
    
    def init_stage(self, conf, verbose):
        logger.info("Initializing Stage")
        from stage import StandaStage, get_connected_axes
        from stage_old import sutterMP285
        if conf["stage"]["type"] == "standa":
            self.stage = StandaStage(uris=get_connected_axes(),
                                     calibration=conf["stage"]["calibration"],
                                     overshoot=conf["stage"]["overshoot"],
                                     verbose=verbose)
        elif conf["psf"]["stage_type"] == "sutter":
            self.stage = sutterMP285("COM4")
        else:
            logger.warning("Stage type not supported (must be either 'standa' or 'sutter')")

    def grab_psf(self, conf):
        """
        record the point spread function
        stage control and stuff
        """
        self.cam.frame_dtype = conf["camera"]["dtype"]
        self.cam.exposure_time = conf["psf"]["exposure_ms"] / 1000

        # collect background frame
        logger.info('Acquiring background frame...')
        self.cam.set_trigger(external=False)  # internal trigger because were aquiring stack not streaming data?
        bg_im = self.cam.acquire_stack(int(5/self.cam.exposure_time))[0].mean(0, dtype='float32') #five seconds

        self.init_stage(conf,verbose=False)
        original_pos = self.stage.get_position(verbose=True)

        _, _, ao_single, do_single, ft = get_full_waveforms(conf,ft=conf["psf"]["exposure_ms"]/1000)
        psf = np.zeros(shape=(conf["psf"]["z_layers"], self.cam.frame_shape[0], self.cam.frame_shape[1]))

        z_positions = np.zeros(shape=(conf["psf"]["z_layers"]))

        # Create a main window with layout
        main_window = pg.Qt.QtWidgets.QWidget()
        layout = pg.Qt.QtWidgets.QVBoxLayout()
        main_window.setLayout(layout)
        main_window.setWindowTitle("PSF Preview with Z Position")
        main_window.resize(900, 950)  # Adjust dimensions for both image and label
        main_window.show()

        # Create the ImageView and add it to the layout
        imv = pg.ImageView()
        imv.ui.menuBtn.hide()  # Hide unnecessary buttons
        imv.ui.roiBtn.hide()
        layout.addWidget(imv)  # Add ImageView to the layout
        imv.setImage(bg_im, levels=(0, 255), autoHistogramRange=False)
        pg.Qt.QtWidgets.QApplication.processEvents()

        # Add a QLabel for displaying the relative Z position
        z_label = pg.Qt.QtWidgets.QLabel()
        layout.addWidget(z_label)  # Add QLabel below the image

        self.cam.set_trigger(external=True, each_frame=False)
        logger.info(f"Exposure time{self.cam.exposure_time*1000}ms")

        with self.dao.queue_data(ao_single, do_single, finite=False, chunked=False):
            outer = tqdm(range(conf["psf"]["z_layers"]),f"Acquiring PSF of {conf["psf"]["z_layers"]} layers with distance {conf["psf"]["z_distance_mm"]}mm",position=0, leave=True)
            for z in range(conf["psf"]["z_layers"]): #tdqm(range(conf["psf"]["z_layers"]),f"Acquiring PSF of {conf["psf"]["z_layers"]}layers with distance {conf["psf"]["z_distance_mm"]}mm"):
                z_pos = self.stage.get_position(verbose=False)[2]
                z_positions[z] = z_pos
                z_label.setText(f"Relative Z Position: {original_pos[2] - z_pos:.3f} mm")
                pg.Qt.QtWidgets.QApplication.processEvents()
                avg_frame = np.zeros(shape=(conf["psf"]["n_frames"],self.cam.frame_shape[0], self.cam.frame_shape[1]))
                for n in tqdm(range(conf["psf"]["n_frames"]), desc=f"Acquiring layer {z+1} of {conf["psf"]["z_layers"]} at zpos {z_pos:.5f}",position=0, leave=True):
                    frame_data = self.cam.acquire_stack(1, verbose=False)[0]
                    imv.setImage(frame_data.T, levels=(0, 255), autoHistogramRange=False)  # Update data
                    pg.Qt.QtWidgets.QApplication.processEvents()  # Update the GUI
                    avg_frame[n,:,:] = frame_data
                    # Detect if the preview window has been closed
                if not imv.isVisible():
                    logger.warning(f"Acquisition interrupted after layer {z}.")
                    self.interrupt_flag = True
                    break

                psf[z, :, :] = avg_frame.mean(axis=0)
                self.stage.move((0, 0, -conf["psf"]["z_distance_mm"]))

        self.stage.move_to(original_pos, verbose= True)
        self.stage.close()
        self.point()

        if conf["psf"]["name_suffix"].strip() != "" or self.interrupt_flag:
            # create directory, save metadata and open HDF5 file
            dirname = arrow.now().format("YYYYMMDD_HHmm") + "_PSF_" + conf['psf']['name_suffix']
            p_target = os.path.join(conf['psf']['base_directory'], dirname)

            if not os.path.exists(p_target):
                os.mkdir(p_target)
            else:
                logger.error('Directory exists! aborting')
                return
            fn = os.path.join(p_target, "psf.h5")
            logger.info(f"Saving PSF to {fn}")
            with h5py.File(fn, 'w') as fh5:
                fh5.create_dataset("bg", data=bg_im)
                fh5.create_dataset("psf", data=psf)
                fh5.create_dataset("z_positions", data=z_positions)
            self.interrupt_flag = False
        logger.info(f"PSF acquisition complete.")

    def preview_psf(self, conf):
        #todo add stage controls
        self.cam.frame_dtype = conf["camera"]["dtype"]
        self.cam.exposure_time = conf["psf"]["exposure_ms"] / 1000
        _, _, ao_single, do_single, ft = get_full_waveforms(conf,ft=conf["psf"]["exposure_ms"]/1000)
        self.cam.set_trigger(external=False, each_frame=True)
        filter_fcn = lambda im: ((im - im.min()) * (255 / (im.max() - im.min()))).astype(im.dtype)

        with self.dao.queue_data(ao_single, do_single, finite=True, chunked=False):
            self.cam.preview(filter_fcn=filter_fcn)

        self.point()
        logger.info(f"Preview stopped")


    def start_preview(self, conf):
        #todo doesnt work :/
        """Start a preview of camera frames."""
        self.cam.frame_dtype = conf["camera"]["dtype"]
        preview_callback, imv = get_preview_callback(self.cam.frame_shape, refresh_every=2)
        pg.Qt.QtWidgets.QApplication.processEvents(pg.Qt.QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 100)
        interrupt = lambda: not imv.isVisible() or self.interrupt_flag

        self.cam.exposure_time = 1/conf['camera']['preview_fps']
        conf['camera']['preview_fps'] = int(1/self.cam.exposure_time)+1  # mismatch from when setting it
        _, _, ao_single, do_single, ft = get_full_waveforms(conf, preview=True)

        self.cam.set_trigger(external=True, each_frame=True)
        self.cam.arm()

        with self.dao.queue_data(ao_single, do_single, finite=False, chunked=False):
            self.cam.stream(num_frames=int(1e9), callback=preview_callback, interrupt=interrupt, already_armed=True)

        self.point()
        logger.info(f"Preview stopped")



    def acquire_timelapse(self, conf):
        """Acquire a timelapse.
        """
        self.cam.frame_dtype = conf["camera"]["dtype"]

        # create directory, save metadata and open HDF5 file
        dirname = arrow.now().format("YYYYMMDD_HHmm_") + conf['acquisition']['name_suffix']
        p_target = os.path.join(conf['acquisition']['base_directory'], dirname)

        if not os.path.exists(p_target):
            os.mkdir(p_target)
        else:
            logger.error('Directory exists! aborting')
            return
        with open(os.path.join(p_target, "meta.json"), "w+") as f:
            json.dump(conf, f, indent=4)

        # open HDF5 file
        fn = os.path.join(p_target, "data.h5")
        #fh5 = h5py.File(fn, "w")


        # collect background frame
        logger.info('Acquiring background frame...')
        self.cam.frame_dtype = conf["camera"]["dtype"]
        self.cam.set_trigger(external=False, each_frame=True) #internal trigger because were aquiring stack not streaming data?
        bg_im = self.cam.acquire_stack(N_BACKGROUND_FRAMES)[0].mean(0, dtype='float32')
        with h5py.File(fn, 'w') as fh5:
            fh5.create_dataset("bg", data=bg_im)

        # setup camera acquisition
        self.cam.exposure_time = 1 / conf['camera']['recording_fps']
        self.cam.set_trigger(external=True, each_frame=False)
        conf["camera"]["recording_fps"] = 1/self.cam.exposure_time

        # create dataset
        logger.info('Set up image dataset...')
        n_frames = conf['acquisition']['recording_seconds']*conf["camera"]["recording_fps"]
        n_ramp_frames = conf['acquisition']['ramp_seconds']*conf["camera"]["recording_fps"]
        stack_shape = (n_frames, *self.cam.frame_shape)
        stack_dtype = self.cam.frame_dtype
        if conf['acquisition']['compress']:
            writer = ParallelCompressedWriter(fn=fn,
                                              name="data",
                                              dtype=stack_dtype,
                                              shape=stack_shape,
                                              chunk_shape=(1, 1, *self.cam.frame_shape),
                                              num_workers=8)
        else:
            writer = VanillaWriter(fn=fn,
                                   name="data",
                                   dtype=stack_dtype,
                                   shape=stack_shape)
        logger.info(f"Dataset with shape {stack_shape} and dtype {stack_dtype} created in {fn}")

        # set up preview
        preview_callback, imv = get_preview_callback(self.cam.frame_shape, 
                                                     window_title='Acquisition preview',
                                                     refresh_every=30)
        pg.Qt.QtWidgets.QApplication.processEvents(pg.Qt.QtCore.QEventLoop.AllEvents, 100)

        
        def interrupt():
            if not imv.isVisible():
                self.interrupt_flag = True
            return self.interrupt_flag 

        # define callback function
        tstmp = np.ones(n_frames, dtype=np.float64) * np.nan
        nfrm = np.zeros(n_frames, dtype=np.int64) - 1

        def callback(im, ii, timestamp, frame_number):
            frame_number -= n_ramp_frames 
            if frame_number == 0:
                logger.info(f'Starting to save at frame {n_ramp_frames+1}')
            if frame_number >= 0:
                #writer.write_chunk(im[None,None], (iVol, iPlane, 0, 0))
                writer.write_frame(im, frame_number)
                tstmp[frame_number] = timestamp
                nfrm[frame_number] = frame_number
            # if frame_number % 60 == 0:
            #     pg.Qt.QtWidgets.QApplication.processEvents(pg.Qt.QtCore.QEventLoop.AllEvents, 1)
            preview_callback(im, ii, timestamp, frame_number)

        #start camera aquisition
        self.cam.arm()

        # setup DAQ
        conf["acquisition"]["ramp_seconds"] = 1 if conf["acquisition"]["ramp_seconds"] == 0 else conf["acquisition"]["ramp_seconds"],
        ao_full, do_full, ao_single, do_single, ft = get_full_waveforms(conf)
        stim_delay_sec = conf['acquisition']['ramp_s']

        # run (with statement manages DAO start and cleanup)
        logger.info(f"Starting acquisition of {n_frames} frames after a ramp of {n_ramp_frames} ...")
        self.interrupt_flag = False
        with self.stim_dao.queue_data(self.stim_data, trigger_delay_sec=stim_delay_sec) if self.stim_dao is not None else nullcontext():
            with self.dao.queue_data(ao_full, do_full, finite=True, chunked=True), writer:
                self.cam.stream(num_frames=n_ramp_frames+n_frames,
                                callback=callback, 
                                already_armed=True, 
                                interrupt=interrupt)
        
        self.point()

        imv.close()

        # stop camera
        self.cam.disarm()

        # save timestamps and frame numbers
        with h5py.File(fn, "a") as fh5:
            fh5.create_dataset("n_frm", data=nfrm)
            fh5.create_dataset("tstmp", data=tstmp)

        if self.interrupt_flag:
            logger.warning(f"Acquisition interrupted after frame {nfrm.max()}.")
            self.interrupt_flag = False
        else:
            logger.info(f"Acquisition complete.")
            

    def load_stimdata(self, p, filename, conf):
        """Load stimulus data from file.
        """
        if filename == '': return
        try:
            with h5py.File(filename, 'r') as fh5:
                rate = fh5['samplerate'][()]
                self.stim_data = fh5['stimulus'][:]
                name = fh5['name'][()].decode('utf-8')
                logger.info(f"Loading stimulus {name} (shape {self.stim_data.shape} {self.stim_data.dtype}), sampled at {rate} Hz (duration: {self.stim_data.shape[1]/rate} s)" )
                chans = unflatten_channel_string(conf['daq']['stim_channels'])
                chans = flatten_channel_string(chans[:self.stim_data.shape[0]])
                self.stim_dao = unifiedDAO(rate, chans, parent=False)
        except:
            logger.error(f"Could not load stimulus from {filename}")
            self.stim_data = None
            self.stim_dao = None
            p.setValue('')
