from .cams import PVCamera as Camera
from .daq import unifiedDAO
from .write import ParallelCompressedWriter, VanillaWriter

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


def generate_opm_waveforms(rate, ft, sweep_t, nplanes, fixed_vps=0, sweep_d=0, galvo_lag_ms=0.12, trig_width_ms=0.1, flyback_frames=0,smooth_x= True):
    """Generate control waveforms for imaging volumes.
    
    Args:
        rate (int): Sampling rate in Hz.
        ft (float): Time per plane in ms.
        sweep_t (float): Time for laser sweep in ms.
        sweep_d (float): Delay from camera trigger to laser sweep in ms.
        nplanes (int): Number of planes.
        fixed_vps (int): Fixed volume rate in volumes per second. If set to a value > 0, ft will be ignored.
        galvo_lag_ms (float): Galvo lag in ms. Defaults to 0.12.
        trig_width_ms (float): Camera trigger width in ms. Defaults to 0.1.

    Returns:
        np.ndarray: Control AO voltages array.
        np.ndarray: Control DO voltages array.
        float: Updated frame time in ms.
    """

    if fixed_vps > 0:
        assert rate % fixed_vps == 0, 'fixed_vps has to be a divisor of the sampling rate'
        nsamps_target = rate // fixed_vps
        nsamps, samps_excess = divmod(nsamps_target, (nplanes+flyback_frames))
        ft = nsamps / rate * 1000
        assert ft >= sweep_t, "Frame time cannot be shorter than sweep time"
        logger.info(f'Volume rate: {fixed_vps} Hz --> frame time set to {ft} ms ({nplanes} planes, {flyback_frames} flyback frames, {1000*samps_excess/rate} excess ms per volume) ')
    else:
        nsamps = int(ft * rate / 1000)
        samps_excess = 0

    y_vals = np.linspace(-1, 1, nplanes)

    ao_data = []
    do_data = []
    for ii in range(nplanes+flyback_frames):
        tvec_len = nsamps + (ii < samps_excess)
        #tvec_len = nsamps + samps_excess if (ii == (nplanes-1)) else nsamps
        tvec = np.arange(0, tvec_len, 1.0) * 1000.0 / rate
        ft_ = tvec_len / rate * 1000
        if ii >= nplanes: # flyback frames 
            cam_bool = cam_bool & False
            laser_v = laser_v * 0
            n_ff = ii - nplanes 
            my_v = np.linspace(1-2*n_ff/flyback_frames, 1-2*(n_ff+1)/flyback_frames, len(my_v))
        else: # all other frames
            cam_bool = (tvec <= trig_width_ms)
            laser_v = (tvec <= sweep_t).astype('float')
            my_v = np.ones(len(tvec)) * y_vals[ii]
            if not((flyback_frames>0) & (ii == (nplanes-1))):
                my_v[tvec > sweep_t] = np.linspace(y_vals[ii], y_vals[(ii + 1) % nplanes], (tvec > sweep_t).sum())
        sawtooth_func = smooth_sawtooth if smooth_x else sawtooth
        mx_v = ((sawtooth_func(2 * np.pi * np.arange(0, tvec_len) / tvec_len, sweep_t / ft_)))
        #laser_v = np.roll(laser_v, int(nsamps * sweep_d / ft_))
        laser_v = np.roll(laser_v, int(rate * sweep_d / 1000))
        ao_data.append(np.vstack([mx_v, my_v, laser_v]))
        do_data.append(cam_bool[None,:])

    ao_data = np.hstack(ao_data)
    do_data = np.hstack(do_data)
    ao_data[:2, :] = np.roll(ao_data[:2, :], int(rate * (sweep_d - galvo_lag_ms) / 1000), axis=-1)
    #ao_data[:2, :] = np.roll(ao_data[:2, :], int(nsamps * (sweep_d - galvo_lag_ms) / ft_), axis=-1)
    return ao_data, do_data, ft


class SimpleArrayProxy:
    """A simple array object proxy with custom shape and getitem function"""

    def __init__(self, custom_getitem, shape=None):
        self.shape = shape
        self._custom_getitem = custom_getitem

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        return self._custom_getitem(slices)


def get_full_waveforms(conf, preview=False):
    ''' Generate full waveforms for the entire recording, including ramping. Note this uses SimpleArrayProxy to avoid storing the entire array in memory.

    Args:
        conf (dict): Configuration dictionary.

    Returns:
        arraylike: Control AO voltages array.
        arraylike: Control DO voltages array.
        float: Updated frame time in ms.
    '''
    ao_single, do_single, ft = generate_opm_waveforms(conf['hardware']['daq']['rate'], conf['scan_parameters']['frame_time_ms'], conf['scan_parameters']['sweep_time_ms'], conf['scan_parameters']['y_planes'],
                                                  conf['scan_parameters']['fixed_vps'], conf['scan_parameters']['sweep_delay_ms'], conf['hardware']['galvo_lag_ms'], conf['hardware']['cam_trig_width_ms'], conf['scan_parameters']['flyback_frames'], conf['scan_parameters']['smooth_x'])
    ao_single[0] = ao_single[0] * conf['scan_parameters']['x_amp_volt'] + conf['scan_parameters']['x_offset_volt']
    ao_single[1] = ao_single[1] * (conf['scan_parameters']['y_planes']-1)/2 * conf['scan_parameters']['y_step_um'] / conf['hardware']['y_um_per_volt']
    assert np.abs(ao_single[:2]).max() <= 5, "sensible voltage range"
    if preview:
        ao_single[2] = ao_single[2] * conf['acquisition']['laser_percent'] / 100 * (np.diff(conf['hardware']['laser_range'])) + conf['hardware']['laser_range'][0]
        assert ao_single[2].max() <= conf['hardware']['laser_range'][1], "laser control voltage range exceeded"
        return ao_single, do_single, ft

    if conf['acquisition']['ramp_vols'] == 0:
        ramp = np.ones(1)
    else:
        ramp = np.geomspace(0.01, 1, conf['acquisition']['ramp_vols']+1)
    ramp = ramp * conf['acquisition']['laser_percent'] / 100 * (np.diff(conf['hardware']['laser_range'])) + conf['hardware']['laser_range'][0]
    assert ramp.max() <= conf['hardware']['laser_range'][1], "exceeding laser voltage range"

    def do_samples_getter(slices):
        slices = list(slices)
        slices[1] = np.r_[slices[1]] % do_single.shape[1]
        return do_single.__getitem__(tuple(slices))

    def ao_samples_getter(slices):
        assert slices[0] == slice(None), f"First slice has to be :, not {slices[0]}"
        slices = list(slices)
        ix = np.minimum(np.r_[slices[1]] // ao_single.shape[1], len(ramp)-1)
        slices[1] = np.r_[slices[1]] % ao_single.shape[1]
        out = ao_single.__getitem__(tuple(slices))
        out[2, :] *= ramp[ix]
        return out

    full_shape = [do_single.shape[0],  (conf['acquisition']['n_volumes'] + conf['acquisition']['ramp_vols']) * do_single.shape[1]]
    do_full = SimpleArrayProxy(do_samples_getter, shape=full_shape)
    ao_full = SimpleArrayProxy(ao_samples_getter, shape=full_shape)
    return ao_full, do_full, ao_single, do_single, ft


def get_ortho_view_callback(im_shape, frames_per_vol, edge_sz=10, refresh_every=30, window_title='Ortho-preview'):
    """Return a callback function for generating an ortho view preview in an ImageView. Requires a GPU and cupy.
    
    Args:
        im_shape (tuple): Shape of the image.
        frames_per_vol (int): Number of frames per volume.
        edge_sz (int): Size of the edge in pixels. Defaults to 10.
        refresh_every (int): Refresh the image every n frames. Defaults to 30.

    Returns:
        function: Callback function.
        function: Interrupt function.
    """
    import cupy as cp

    im_buffer = cp.zeros((im_shape[0] + edge_sz + frames_per_vol, im_shape[1] + edge_sz + frames_per_vol), dtype=np.uint8)
    vol_buffer = cp.zeros((frames_per_vol, *im_shape), 'uint8')

    imv = pg.ImageView()
    imv.ui.menuBtn.hide()
    imv.ui.roiBtn.hide()
    imv.resize(im_buffer.shape[1] // 2, im_buffer.shape[0] // 2)
    imv.move(355, 0)
    imv.setWindowTitle(window_title)
    imv.show()


    def callback(im_np, i_frame, timestamp, frame_count):
        ii = i_frame % frames_per_vol
        vol_buffer[ii] = cp.array(im_np)
        if i_frame % refresh_every == 0:
            im_buffer[:im_shape[0], :im_shape[1]] = vol_buffer.max(0)
            im_buffer[-frames_per_vol:, :im_shape[1]] = vol_buffer[:, im_np.shape[0] // 2, :]
            im_buffer[:im_shape[0], -frames_per_vol:] = vol_buffer.max(2).T
            imv.setImage(im_buffer.T.get(), levels=(0, 255), autoHistogramRange=False)
            pg.Qt.QtWidgets.QApplication.processEvents()
    return callback, imv


class OPM:
    """A class for controlling the DAQ and camera for OPM"""

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
        self.cam.cam.gain = cam_gain
        self.interrupt_flag = False

    def point(self):
        """Set output voltages to zero
        """
        chans = unflatten_channel_string(self.dao.ao_chans)
        self.dao.queue_data(np.zeros((len(chans), 2)), finite=True, chunked=False)
        self.dao.task_ao.start()
        self.dao.task_ao.wait_until_done()
        self.dao.close()
        logger.info('point')

    def start_preview(self, conf):
        """Start a preview of camera frames."""
        ao_single, do_single, ft = get_full_waveforms(conf, preview=True)

        self.cam.exposure_time = conf['camera']['exposure_ms'] / 1000
        self.cam.set_trigger(True)

        with self.dao.queue_data(ao_single, do_single, finite=False, chunked=False):
            self.cam.preview(fifo=False)

        self.point()
        logger.info(f"Preview stopped")

    def start_ortho_preview(self, conf):
        """Start a preview which shows the orthogonal views of the volume. Requires a GPU and cupy.
        """

        preview_callback, imv = get_ortho_view_callback(self.cam.frame_shape, conf['scan_parameters']['y_planes'])
        pg.Qt.QtWidgets.QApplication.processEvents(pg.Qt.QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 100)
        interrupt = lambda: not imv.isVisible() or self.interrupt_flag

        self.cam.exposure_time = conf['camera']['exposure_ms'] / 1000
        self.cam.set_trigger(True)
        self.cam.arm()

        ao_single, do_single, ft = get_full_waveforms(conf, preview=True)

        with self.dao.queue_data(ao_single, do_single, finite=False, chunked=False):
            self.cam.stream(num_frames=int(1e9), callback=preview_callback, interrupt=interrupt, already_armed=True)

        self.point()
        logger.info(f"Preview stopped")

    def start_beam_array(self, conf):
        """Start a preview of the beam array."""

        ao_single, do_single, ft = get_full_waveforms(conf, preview=True)
        time_vec = np.r_[:ao_single.shape[1]]/self.dao.rate
        ao_single[2,:] = conf['hardware']['laser_range'][0]
        ao_single[2,::10] = conf['hardware']['laser_range'][1]

        self.cam.exposure_time = conf['camera']['exposure_ms'] / 1000
        self.cam.set_trigger(True)

        with self.dao.queue_data(ao_single, do_single, finite=False, chunked=False):
            self.cam.preview(fifo=False)

        logger.info(f"Preview stopped")

    def acquire_timelapse(self, conf):
        """Acquire a timelapse.
        """

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
        self.cam.set_trigger(False)
        bg_im = self.cam.acquire_stack(N_BACKGROUND_FRAMES)[0].mean(0, dtype='float32')
        with h5py.File(fn, 'w') as fh5:
            fh5.create_dataset("bg", data=bg_im)

        # create dataset
        logger.info('Set up image dataset...')
        n_vols = conf['acquisition']['n_volumes']
        n_ramp_vols = conf['acquisition']['ramp_vols']
        n_planes = conf['scan_parameters']['y_planes']
        stack_shape = (n_vols, n_planes, *self.cam.frame_shape)
        stack_dtype = 'uint8'
        if conf['acquisition']['compress']:
            writer = ParallelCompressedWriter(fn=fn, name="data", dtype=stack_dtype, shape=stack_shape, chunk_shape=(1, 1, *self.cam.frame_shape), num_workers=8)
        else:
            writer = VanillaWriter(fn=fn, name="data", dtype=stack_dtype, shape=stack_shape)
        logger.info(f"Dataset with shape {stack_shape} and dtype {stack_dtype} created in {fn}")

        # set up ortho preview
        preview_callback, imv = get_ortho_view_callback(self.cam.frame_shape, conf['scan_parameters']['y_planes'], window_title='Acquisition preview', refresh_every=conf['scan_parameters']['y_planes'])
        pg.Qt.QtWidgets.QApplication.processEvents(pg.Qt.QtCore.QEventLoop.AllEvents, 100)

        def interrupt():
            if not imv.isVisible():
                self.interrupt_flag = True
            return self.interrupt_flag 

        # define callback function
        tstmp = np.ones(n_vols * n_planes, dtype=np.float64) * np.nan
        nfrm = np.zeros(n_vols * n_planes, dtype=np.int64) - 1

        def callback(im, ii, timestamp, frame_number):
            frame_number -= n_ramp_vols * n_planes
            if frame_number == 0:
                logger.info(f'Starting to save at frame {n_ramp_vols * n_planes}')
            if frame_number >= 0:
                iVol, iPlane = np.unravel_index(frame_number, [n_vols, n_planes])
                #writer.write_chunk(im[None,None], (iVol, iPlane, 0, 0))
                writer.write_frame(im, iVol, iPlane)
                tstmp[frame_number] = timestamp
                nfrm[frame_number] = frame_number
            # if frame_number % 60 == 0:
            #     pg.Qt.QtWidgets.QApplication.processEvents(pg.Qt.QtCore.QEventLoop.AllEvents, 1)
            preview_callback(im, ii, timestamp, frame_number)

        # start camera acquisition
        self.cam.exposure_time = conf['camera']['exposure_ms'] / 1000
        self.cam.set_trigger(True)
        self.cam.arm()

        # setup DAQ
        ao_full, do_full, ao_single, do_single, ft = get_full_waveforms(conf)
        stim_delay_sec = n_ramp_vols * ao_single.shape[1] / self.dao.rate

        # run (with statement manages DAO start and cleanup)
        logger.info(f"Starting acquisition of {n_vols} volumes after a ramp of {n_ramp_vols} ...")
        self.interrupt_flag = False
        with self.stim_dao.queue_data(self.stim_data, trigger_delay_sec=stim_delay_sec) if self.stim_dao is not None else nullcontext():
            with self.dao.queue_data(ao_full, do_full, finite=True, chunked=True), writer:
                self.cam.stream(num_frames=(n_vols + n_ramp_vols) * n_planes, callback=callback, already_armed=True, interrupt=interrupt)
        
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
                chans = unflatten_channel_string(conf['hardware']['daq']['stim_channels'])
                chans = flatten_channel_string(chans[:self.stim_data.shape[0]])
                self.stim_dao = unifiedDAO(rate, chans, parent=False)
        except:
            logger.error(f"Could not load stimulus from {filename}")
            self.stim_data = None
            self.stim_dao = None
            p.setValue('')

def smooth_sawtooth(t, width=1.0):

    """
    Generate a smooth sawtooth wave with smooth transitions between upstroke and downstroke.

    Parameters:
    - t: array_like
        Time values where the waveform is evaluated.
    - width: float, optional
        Width of the rising slope (0 <= width <= 1). Default is 1.0.

    Returns:
    - y: ndarray
        Output waveform evaluated at time t.
    """
    # Normalize time to the range [0, 1)
    t_mod = np.linspace(0,1,t.shape[0]+1)[:-1]

    # Initialize the waveform array
    y = np.zeros_like(t_mod)

    # Rising slope
    rising_mask = t_mod < width
    if width > 0:
        y[rising_mask] = t_mod[rising_mask] / width

    # Downstroke using a quintile hermite spline
    if width < 1:
        downstroke_mask = ~rising_mask
        downstroke_times = t_mod[downstroke_mask]

        # Define points and derivatives
        x_point = np.array([width, 1.0])
        
        derivatives = np.array([
            (1.0, 0.0), # y points
            (1/width, 1/width), # 1st order derivatives -ensures smooth velocity
            (-2,2) #2nd order derivatives - ensures smooth acceleration
        ]).T
        # Fit Hermite spline 
        from scipy.interpolate import BPoly
        spline = BPoly.from_derivatives(x_point, derivatives)
        y[downstroke_mask] = spline(downstroke_times)

    return (y*2.0)-1.0