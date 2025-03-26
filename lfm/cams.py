import numpy as np
import time
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

import logging
logger = logging.getLogger(__name__)


class CameraBase(ABC):
    """Base class for cameras."""

    @property
    @abstractmethod
    def roi(self):
        '''Get ROI as dict'
        '''

    @abstractmethod
    def set_roi(self, y_size=None, x_size=None, y_offset=None, x_offset=None, y_bin=1, x_bin=1):
        '''Set ROI

        Args:
            y_size: height of ROI
            x_size: width of ROI
            y_offset: y offset of ROI
            x_offset: x offset of ROI
            y_bin: y binning
            x_bin: x binning
        '''

    def set_decentered_roi(self, y_size=None, x_size=None, y_offset=None, x_offset=None, y_bin=1, x_bin=1):
        sh = self.sensor_shape
        self.set_roi(y_size=y_size, x_size=x_size, y_offset=y_offset + (sh[0] - y_size) // 2, x_offset=x_offset + (sh[1] - x_size) // 2, y_bin=y_bin, x_bin=x_bin)

    @abstractmethod
    def set_trigger(self, external=True, each_frame=True):
        '''Set trigger type
        Args:
            external: True for external trigger
            each_frame: True for each frame, False for first frame
        '''

    @property
    @abstractmethod
    def exposure_time(self):
        '''Get exposure time (in s)'''

    @exposure_time.setter
    @abstractmethod
    def exposure_time(self, t):
        '''Set exposure time (in s)

        Args:
            t: exposure time in s'''

    @abstractmethod
    def arm(self, fifo=True):
        '''Arm camera

        Args:
            fifo: True to use FIFO frame polling
        '''

    @abstractmethod
    def disarm(self):
        '''Disarm camera'''

    @abstractmethod
    def poll_frame(self, copy=False):
        '''Get frame as numpy array. 
        Args:
            copy: True to copy frame, False to return view
        Return:
            im: frame as numpy array
            timestamp: timestamp of frame
            meta: meta data of frame'''

    @abstractmethod
    def __del__(self):
        '''cleanup'''

    @property
    @abstractmethod
    def frame_dtype(self):
        '''Get dtype of output frame'''

    @property
    @abstractmethod
    def frame_shape(self):
        '''Get shape of frame as [height, width]'''

    @property
    @abstractmethod
    def sensor_shape(self):
        '''Get shape of frame as [height, width]'''

    def triggered_stream(self, varargin):
        #TODO remove the need for this function
        print('nah')
        return

    def set_binning(self, val):
        #TODO remove the need for this function
        self._binning = val
        print('nah')
        return

    def frame_bytes(self):
        '''Get number of bytes per frame'''
        return np.prod(self.frame_shape) * np.dtype(self.frame_dtype).itemsize

    def stream(self, num_frames, callback=None, interrupt=None, stream_to_disk_path=None, fifo=True, already_armed=False):
        '''Loop through frames.

        Args:
            num_frames: number of frames to acquire
            callback: function to call on each frame. Should take arguments (im, i_frame, timestamp, frame_count)
            interrupt: function returning True to stop acquisition
            stream_to_disk_path: path to stream frames to disk
            fifo: True to use FIFO buffer, False to use LIFO buffer
            already_armed: True if camera is already armed
        '''
        if not already_armed:
            self.arm(stream_to_disk_path=stream_to_disk_path, fifo=fifo)

        last_frame = -1
        for i_frame in tqdm(range(num_frames)):

            if interrupt is not None and interrupt():
                break
            try:
                im, timestamp, frame_count = self.poll_frame(copy=False)
            except RuntimeError as e:
                logger.warning(f'Timeout for frame {i_frame} - stopping acquisition. Last frame (camera index): {last_frame}')
                break
            if i_frame == 0:
                first_frame = frame_count
            frame_count -= first_frame
            #logger.info(f'frame: {frame_count}')
            if fifo and frame_count - last_frame > 1:
                logger.warning(f'{frame_count-last_frame-1} frames dropped. Current frame: {frame_count} (actual); {i_frame} (desired)')
            if callback is not None:
                callback(im, i_frame, timestamp, frame_count)
            last_frame = frame_count
        self.disarm()

    def acquire_stack(self, num_frames):
        '''Acquire stack of frames

        Args:
            num_frames: number of frames to acquire
        Return:
            im_stack: stack of frames
            timestamps: timestamps of frames
        '''
        im_stack = np.zeros((num_frames, *self.frame_shape), dtype=self.frame_dtype)
        timestamps = np.zeros(num_frames)
        frame_counts = np.zeros(num_frames)

        def callback(im, i_frame, timestamp, frame_count):
            im_stack[i_frame] = im
            timestamps[i_frame] = timestamp
            frame_counts[i_frame] = frame_count

        self.stream(num_frames, callback=callback)
        return im_stack, timestamps, frame_counts

    def preview(self, window_shape=None, filter_fcn=lambda x: x, fifo=False):
        '''Simple camera preview. Press ESC to end.
        
        Args:
            window_shape: shape of preview window (None for full frame)
            filter_fcn: optional function to apply to each frame before display
        '''
        import cv2
        if window_shape is None:
            window_shape = (self.frame_shape[1] // 2, self.frame_shape[0] // 2)
        cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('preview', *window_shape)
        cv2.setWindowTitle('preview', f'Preview')
        cv2.setWindowProperty('preview', cv2.WND_PROP_TOPMOST, 1)

        def preview_callback(im_np, i_frame, timestamp, frame_count):
            im_np = filter_fcn(im_np)
            cv2.imshow('preview', im_np)
            if i_frame % 30 == 0:
                cv2.waitKey(1)
            #cv2.setWindowProperty('preview', cv2.WND_PROP_TOPMOST, 1)

        interrupt = lambda: (cv2.getWindowProperty('preview', cv2.WND_PROP_VISIBLE) < 1)
        self.stream(int(1e9), callback=preview_callback, interrupt=interrupt, fifo=fifo)
        cv2.destroyAllWindows()

    def preview_mosaic(self, win_shape=[50, 50], edge_dist=[50, 50]):
        '''3x3 mosaic preview

        Args:
            win_shape: shape of each window
            edge_dist: distance of each window from edge of frame
        '''

        def filter_fcn(im):
            ix = [(np.array([edge_dist[i], (self.frame_shape[i] - win_shape[i]) // 2, self.frame_shape[i] - win_shape[i] - edge_dist[i]])[:, None] + np.arange(win_shape[i])[None, :]).flatten() for i in range(2)]
            out = im[ix[0]][:, ix[1]]
            out[::win_shape[0], :] = 255
            out[:, ::win_shape[1]] = 255
            return out[1:, 1:]

        #old_roi = self.roi
        #self.set_roi() #set to full frame
        self.preview(window_shape=(np.array(win_shape) * 3 - 1) * 2, filter_fcn=filter_fcn)
        #.set_roi(**old_roi)

class PVCamera(CameraBase):
    '''Photometrics PVCAM camera'''

    def __init__(self, conf=None):
        from pyvcam import pvc, constants
        from pyvcam.camera import Camera
        try:
            pvc.init_pvcam()
        except RuntimeError as e:
            logger.warning(e)
        self.pvc = pvc
        self.const = constants
        self.cam = next(Camera.detect_camera())
        self.cam.open()
        self.cam.exp_res = 1  # set to 1 for µs (do not change)
        self.cam.meta_data_enabled = False
        self.cam.readout_port = 1
        self.cam.__dtype = 'uint8'
        self.cam.set_param(self.const.PARAM_FAN_SPEED_SETPOINT, self.const.FAN_SPEED_HIGH)
        self.set_roi()
        self.set_trigger(external=False)
        self._buffer_frames = 256

    @property
    def roi(self):
        '''Get ROI'''
        return self._roi

    def set_roi(self, y_size=None, x_size=None, y_offset=None, x_offset=None, y_bin=1, x_bin=1):
        '''Set ROI'''
        if y_size is None:
            y_size = self.sensor_shape[0]
        if x_size is None:
            x_size = self.sensor_shape[1]
        if y_offset is None:
            y_offset = (self.sensor_shape[0] - y_size) // 2
        if x_offset is None:
            x_offset = (self.sensor_shape[1] - x_size) // 2
        self.cam.reset_rois()
        self.cam.binning = (x_bin, y_bin)
        self.cam.set_roi(x_offset, y_offset, x_size, y_size)
        self._roi = dict(y_size=y_size, x_size=x_size, y_offset=y_offset, x_offset=x_offset, y_bin=y_bin, x_bin=x_bin)

    def set_trigger(self, external=True, each_frame=True):
        '''Set trigger type'''
        if external:
            if each_frame:
                self.cam.exp_mode = 'Edge Trigger'
            else:
                self.cam.exp_mode = 'Trigger First'
        else:
            self.cam.exp_mode = 'Internal Trigger'

    @property
    def exposure_time(self):
        '''Get exposure time (in s)'''
        return (self.cam.exp_time / 1e6)

    @exposure_time.setter
    def exposure_time(self, t):
        '''Set exposure time (in s)'''
        self.cam.exp_time = int(t * 1e6)

    def arm(self, stream_to_disk_path=None, fifo=True):
        '''Arm camera'''
        self._last_frame = None
        self.cam.start_live(buffer_frame_count=self._buffer_frames, stream_to_disk_path=stream_to_disk_path)
        self._fifo = fifo

    def disarm(self):
        '''Disarm camera'''
        self.cam.finish()

    def poll_frame(self, copy=False):
        '''Get frame as numpy array. Return im, timestamp, frame_count'''
        d, fps, frame_count = self.cam.poll_frame(copyData=False, oldestFrame=self._fifo, timeout_ms=2000)
        return d['pixel_data'], time.time(), frame_count

    @property
    def frame_dtype(self):
        '''Get dtype of output frame'''
        return self.cam.__dtype

    @property
    def frame_shape(self):
        '''Get dtype of output frame'''
        return self.cam.shape(0)[::-1]

    @property
    def sensor_shape(self):
        '''Return shape of frame as [height, width]'''
        return self.cam.sensor_size[::-1]

    def __del__(self):
        self.cam.set_param(self.const.PARAM_FAN_SPEED_SETPOINT, self.const.FAN_SPEED_LOW)
        self.cam.close()
        try:
            self.pvc.uninit_pvcam() # this is cleaner but only works for single camera setups
        except RuntimeError as e:
            logger.warning(e)


class XimeaCamera(CameraBase):
    '''Ximea XiAPI camera'''

    def __init__(self, conf):
        from ximea import xiapi
        self.xicam = xiapi.Camera()
        self.xicam.open_device()
        self.xicam.set_imgdataformat('XI_MONO8')
        self.xicam.set_image_data_bit_depth('XI_BPP_8')
        self.xicam.set_sensor_bit_depth('XI_BPP_10')
        self.xicam.set_binning_selector('XI_BIN_SELECT_HOST_CPU')
        self.xicam.set_gain_selector('XI_GAIN_SELECTOR_DIGITAL_ALL')
        self.xicam.set_gain(0)
        self.xicam.set_gpi_selector('XI_GPI_PORT2')
        self.xicam.set_gpi_mode('XI_GPI_TRIGGER')
        self.xicam.set_acq_buffer_size(1024 * 1024**2)  # 1 GB
        self._sensor_shape = [self.xicam.get_height_maximum(), self.xicam.get_width_maximum()]
        self.LUT_slope = conf['LUTSlope']
        self.LUT_offset = conf['LUTOffset']
        self.set_trigger(external=False)
        self.xiimage = xiapi.Image()
        self.set_lut()

    @property
    def roi(self):
        '''Get ROI as dict'
        '''
        out = dict(y_size=self.xicam.get_height(), x_size=self.xicam.get_width(), y_offset=self.xicam.get_offsetY(), x_offset=self.xicam.get_offsetX(), y_bin=self.xicam.get_binning_vertical(), x_bin=self.xicam.get_binning_horizontal())
        return out

    def set_roi(self, y_size=None, x_size=None, y_offset=None, x_offset=None, y_bin=1, x_bin=1):
        '''Set ROI

        Args:
            y_size: height of ROI
            x_size: width of ROI
            y_offset: y offset of ROI
            x_offset: x offset of ROI
            y_bin: y binning
            x_bin: x binning
        '''
        if y_size is None:
            y_size = self.sensor_shape[0]
        if x_size is None:
            x_size = self.sensor_shape[1]
        if y_offset is None:
            y_offset = (self.sensor_shape[0] - y_size) // 2
        if x_offset is None:
            x_offset = (self.sensor_shape[1] - x_size) // 2
        print(self.sensor_shape, y_size, x_size, y_offset, x_offset)
        dx = self.xicam.get_width_increment()
        dy = self.xicam.get_height_increment()
        self.xicam.set_offsetY(0)
        self.xicam.set_offsetX(0)
        self.xicam.set_height((y_size // dy) * dy)
        self.xicam.set_width((x_size // dx) * dx)
        self.xicam.set_offsetY((y_offset // dy) * dy)
        self.xicam.set_offsetX((x_offset // dx) * dx)
        self.xicam.set_binning_vertical(y_bin)
        self.xicam.set_binning_horizontal(x_bin)

    def set_trigger(self, external=True, each_frame=True):
        '''Set trigger type
        Args:
            external: True for external trigger
            each_frame: True for each frame, False for first frame
        '''
        if not each_frame:
            logger.warning('Start trigger currently not supported')
            return
        if external:
            self.xicam.set_trigger_source('XI_TRG_EDGE_RISING')
            self._triggered = 1
            logger.info('Camera set to Trigger Mode')
        else:
            self.xicam.set_trigger_source('XI_TRG_OFF')
            logger.info('Camera set to Free Running Mode')
            self._triggered = 0

    @property
    def exposure_time(self):
        '''Get exposure time (in s)'''
        return self.xicam.get_exposure() / 1e6

    @exposure_time.setter
    def exposure_time(self, t):
        '''Set exposure time (in s)

        Args:
            t: exposure time in s'''
        self.xicam.set_exposure(int(t * 1e6))  # us

    def set_lut(self):
        '''
        Change Camera Lookup table. Has to be called before every acquisition
        '''
        LUT = np.clip((np.arange(0, 2**12) * self.LUT_slope - self.LUT_offset), 0, 2**12 - 1).astype('int')
        for ii in range(self.xicam.get_LUTIndex_minimum(), self.xicam.get_LUTIndex_maximum() + 1, 1):
            self.xicam.set_LUTIndex(ii)
            self.xicam.set_LUTValue(LUT[ii])
        # self.xicam.disable_LUTEnable()
        # self.xicam.enable_LUTEnable()
        self.xicam.get_param('LUTEnable')
        self.xicam.set_param('LUTEnable', 1)
        self.xicam.get_param('LUTEnable')

    def arm(self, stream_to_disk_path=None, fifo=True):
        '''Arm camera'''
        # self.cam.set_acq_buffer_size(int(self.cam.mem_prev * 1024 * 1024))
        self.xicam.set_buffers_queue_size(self.xicam.get_buffers_queue_size_maximum())
        self.xicam.set_param("recent_frame", int(not fifo))
        self._fifo = fifo
        self._last_frame = None
        self.set_lut()
        self.xicam.start_acquisition()

    def disarm(self):
        '''Disarm camera'''
        self.xicam.stop_acquisition()

    def poll_frame(self, copy=False):
        '''Get frame as numpy array. 
        
        Args:
            copy: True to copy frame, False to return view
        Return:
            im: frame as numpy array
            timestamp: timestamp of frame
            meta: meta data of frame'''
        self.xicam.get_image(self.xiimage, timeout=5000)
        im = np.frombuffer(self.xiimage.get_image_data_raw(), dtype=np.uint8)
        im = im.reshape((self.xiimage.height, self.xiimage.width))
        if copy:
            im = im.copy()
        frame_count = self.xiimage.acq_nframe
        if self._fifo and (self._last_frame is not None) and (frame_count - self._last_frame > 1):
            logger.warning(f'{frame_count - self._last_frame - 1} frames dropped')
        self._last_frame = frame_count
        ts = self.xiimage.tsSec + self.xiimage.tsUSec * 1e-6
        return im, ts, frame_count

    def __del__(self):
        '''cleanup'''
        self.xicam.close()
        del self.xicam

    @property
    def frame_dtype(self):
        '''Get dtype of output frame'''
        bit_depth = self.xicam.get_image_data_bit_depth()[7:]
        dtype = 'uint' + bit_depth
        return dtype

    @property
    def frame_shape(self):
        '''Get shape of frame as [height, width]'''
        
        return [self.xicam.get_height(), self.xicam.get_width()]

    @property
    def sensor_shape(self):
        '''Get shape of sensor as [height, width]'''
        return self._sensor_shape

    def close(self):
        pass


from pylablib.devices.DCAM import DCAMCamera
class HanumatsuCamera(CameraBase):
    '''Hamamatsu DCAM camera using pylablib
       written by chat GPT :)
    '''

    def __init__(self, conf=None):
        '''Initialize the camera'''
        self.cam = DCAMCamera()
        self.cam.open()
        
        self._buffer_frames = 256
        self._fifo = True
        self._last_frame = None

    @property
    def roi(self):
        '''Get ROI as a dictionary'''
        roi = self.cam.get_roi()
        return dict(y_size=roi["hend"]-roi["hstart"],
                    x_size=roi["hend"]-roi["hstart"], 
                    y_offset=roi["hstart"], 
                    x_offset=roi["vstart"], 
                    y_bin=roi["hbin"], 
                    x_bin=roi["vbin"],)

    def set_roi(self, y_size=None, x_size=None, y_offset=None, x_offset=None, y_bin=1, x_bin=1):
        '''Set the Region of Interest (ROI)'''
        roi = self.roi
        if y_size is not None:
            y_size = roi["y_size"]
        if x_size is not None:
            x_size = roi["x_size"]
        if y_offset is not None:
            y_offset = roi["y_offset"]
        if x_offset is not None:
            x_offset = roi["x_offset"]
        if y_bin is not None:
            y_bin = roi["y_bin"]
        if x_bin is not None:
            x_bin = roi["x_bin"]
        self.cam.set_roi(hstart=y_offset,
                         hend=y_offset+y_size,
                         vstart=x_offset,
                         vend=x_offset+x_size,
                         hbin=y_bin,
                         vbin=x_bin,) #due to library, this is ignored and always set to ybin

    def set_trigger(self, external=True, each_frame=True):
        '''Set trigger mode'''
        if external:
            self.cam.set_trigger_mode("ext")

        else:
            self.cam.set_trigger_mode("int")

    @property
    def exposure_time(self):
        '''Get the exposure time (in seconds)'''
        return self.cam.get_exposure()

    @exposure_time.setter
    def exposure_time(self, t):
        '''Set the exposure time'''
        self.cam.set_exposure(t)

    def arm(self, stream_to_disk_path=None, fifo=True):
        '''Start acquisition (live mode)'''
        self.cam.setup_acquisition(nframes=self._buffer_frames)
        self.cam.start_acquisition()
        self._fifo = fifo
        self._last_frame = None

    def disarm(self):
        '''Stop acquisition'''
        self.cam.stop_acquisition()

    def poll_frame(self, copy=False):
        '''Retrieve a frame from the camera'''
        frame_data = self.cam.read_newest_image()
        if frame_data is None:
            logger.warning("No frames captured")
            return None, None, None
        if copy:
            frame_data = np.copy(frame_data)
        timestamp = time.time()
        return frame_data, timestamp, None  # No direct frame count tracking available

    @property
    def frame_dtype(self):
        '''Get the data type of frames'''
        bit_depth = self.cam.get_attribute_value("image_pixel_type")
        if "16" in bit_depth:
            return "uint16"
        else:
            return "uint8"

    @property
    def frame_shape(self):
        '''Get the shape of frames (height, width)'''
        height = self.cam.get_attribute_value("image_height")
        width = self.cam.get_attribute_value("image_width")
        return (int(height), int(width))

    @property
    def sensor_shape(self):
        '''Get the full sensor resolution as (height, width)'''
        return self.cam.get_detector_size()

    def __del__(self):
        '''Clean up and close camera connection'''
        self.cam.close()
        

from pyDCAM import *
class DCamera(CameraBase):
    '''Hamamatsu DCAM camera using pyDCAM'''

    def __init__(self, conf=None):
        '''Initialize the camera'''
        dcamapi_init()
        self.hdcam = dcamapi.HDCAM()
        self._roi = {}
        self._buffer_frames = 256
        self._fifo = True
        self._last_frame = None

    @property
    def roi(self):
        '''Get ROI as a dictionary'''
        return self._roi

    def set_roi(self, y_size=None, x_size=None, y_offset=None, x_offset=None, y_bin=1, x_bin=1):
        '''Set the Region of Interest (ROI)'''
        # Assuming the camera supports setting ROI through properties
        if y_size is not None:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.IMAGE_HEIGHT, y_size)
        if x_size is not None:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.IMAGE_WIDTH, x_size)
        if y_offset is not None:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.IMAGE_TOP, y_offset)
        if x_offset is not None:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.IMAGE_LEFT, x_offset)
        if y_bin is not None:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.BINNING, y_bin)
        if x_bin is not None:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.BINNING, x_bin)
        self._roi = {
            'y_size': y_size,
            'x_size': x_size,
            'y_offset': y_offset,
            'x_offset': x_offset,
            'y_bin': y_bin,
            'x_bin': x_bin
        }

    def set_trigger(self, external=True, each_frame=True):
        '''Set trigger mode'''
        if external:
            if each_frame:
                self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.TRIGGER_MODE, dcamprop.DCAMPROP.TRIGGER_MODE__NORMAL)
                self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.TRIGGER_SOURCE, dcamprop.DCAMPROP.TRIGGER_SOURCE__EXTERNAL)
            else:
                self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.TRIGGER_MODE, dcamprop.DCAMPROP.TRIGGER_MODE__START)
                self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.TRIGGER_SOURCE, dcamprop.DCAMPROP.TRIGGER_SOURCE__EXTERNAL)
        else:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.TRIGGER_MODE, dcamprop.DCAMPROP.TRIGGER_MODE__NORMAL)
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.TRIGGER_SOURCE, dcamprop.DCAMPROP.TRIGGER_SOURCE__INTERNAL)

    @property
    def exposure_time(self):
        '''Get the exposure time (in seconds)'''
        return self.hdcam.dcamprop_getvalue(dcamprop.DCAM_IDPROP.EXPOSURETIME) * 1e6

    @exposure_time.setter
    def exposure_time(self, t):
        '''Set the exposure time (in seconds)'''
        self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.EXPOSURETIME, t)

    def arm(self, stream_to_disk_path=None, fifo=True):
        '''Start acquisition (live mode)'''
        self.hdcam.dcambuf_alloc(self._buffer_frames)
        self.hdcam.dcamcap_start(dcamapi.DCAMCAP_START.SEQUENCE)
        self._fifo = fifo
        self._last_frame = None

    def disarm(self):
        '''Stop acquisition'''
        self.hdcam.dcamcap_stop()
        self.hdcam.dcambuf_release()

    def poll_frame(self, copy=False):
        '''Retrieve a frame from the camera'''
        frame_index, frame_count = self.hdcam.dcamcap_transferinfo()
        if frame_index < 0:
            logger.warning('No frames captured')
            return None, None, None
        frame_data = self.hdcam.dcambuf_copyframe(frame_index)
        if copy:
            frame_data = np.copy(frame_data)
        timestamp = time.time()
        if self._fifo and self._last_frame is not None and frame_count - self._last_frame > 1:
            logger.warning(f'{frame_count - self._last_frame - 1} frames dropped')
        self._last_frame = frame_count
        return frame_data, timestamp, frame_count

    @property
    def frame_dtype(self):
        '''Get the data type of frames (e.g., uint8, uint16)'''
        bit_depth = self.hdcam.dcamprop_getvalue(dcamprop.DCAM_IDPROP.IMAGE_PIXELTYPE)
        if bit_depth == dcamprop.DCAMPROP.IMAGE_PIXELTYPE__MONO16:
            return np.uint16
        elif bit_depth == dcamprop.DCAMPROP.IMAGE_PIXELTYPE__MONO8:
            return np.uint8
        else: #12bit
            self.frame_dtype = np.uint8
            return np.uint8

    @frame_dtype.setter
    def frame_dtype(self, dtype):
        if dtype == np.uint8:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.IMAGE_PIXELTYPE, dcamprop.DCAMPROP.IMAGE_PIXELTYPE__MONO8)
        elif dtype == np.uint16:
            self.hdcam.dcamprop_setvalue(dcamprop.DCAM_IDPROP.IMAGE_PIXELTYPE, dcamprop.DCAMPROP.IMAGE_PIXELTYPE__MONO16)
        else:
            print("dtype not supported idiont")

    @property
    def frame_shape(self):
        '''Get the shape of frames (height, width)'''
        height = int(self.hdcam.dcamprop_getvalue(dcamprop.DCAM_IDPROP.IMAGE_HEIGHT))
        width = int(self.hdcam.dcamprop_getvalue(dcamprop.DCAM_IDPROP.IMAGE_WIDTH))
        return (height, width)

    @property
    def sensor_shape(self):
        '''Get the full sensor resolution as (height, width)'''

        return [self.hdcam.dcamprop_getvalue(dcamprop.DCAM_IDPROP_IMAGEDETECTOR_PIXELHEIGHT),
                self.hdcam.dcamprop_getvalue(dcamprop.DCAM_IDPROP_IMAGEDETECTOR_PIXELWIDTH)]

    def __del__(self):
        '''Clean up and close camera connection'''
        self.hdcam.dcamdev_close()
        dcamapi.uninit()




class FrameQueue():
    '''Queue for sharing frames between processes using multiprocessing.shared_memory. 
    This is faster than multiprocessing.Queue, but requires the frame size to be constant.'''

    def __init__(self, qsize=1):
        import multiprocessing
        import multiprocessing.shared_memory
        self.q = multiprocessing.Queue(qsize)
        self.sema = multiprocessing.Semaphore(qsize)
        self.counter = multiprocessing.Value('i', 0)
        self.shm = None
        self.view = None

    def put(self, frame, meta=None, drop_when_full=False):
        '''Put frame into queue. This will block if the queue is full. All frames have to be of the same size and dtype.

        Args:
            frame: frame to put into queue
            meta: optional meta data to put into queue (keep this small, as it will pass through a conventional queue)
            drop_when_full: True to drop oldest frame in queue if full, False to block if full (default)
        '''
        if self.shm is None:
            self.shm = multiprocessing.shared_memory.SharedMemory(create=True, size=frame.nbytes * self.q._maxsize)
            self.view = np.ndarray((self.q._maxsize, *frame.shape), dtype=frame.dtype, buffer=self.shm.buf)
        if drop_when_full and self.q.full():  #drop oldest frame
            self.q.get()
            self.sema.release()
        self.sema.acquire()
        index = self.counter.value % self.q._maxsize
        self.view[index] = frame
        self.q.put(dict(name=self.shm.name, shape=frame.shape, dtype=frame.dtype, index=index, meta=meta))
        with self.counter.get_lock():
            self.counter.value += 1

    def get(self):
        '''Get frame from queue. This will block if the queue is empty.
        
        Returns:
            frame: frame from queue
            meta: any meta data from queue 
        '''
        item = self.q.get()
        if (self.shm is None) or (self.shm.name != item['name']):
            nbytes = self.q._maxsize * np.prod(item['shape']) * item['dtype'].itemsize
            self.shm = multiprocessing.shared_memory.SharedMemory(name=item['name'], create=False, size=nbytes)
            self.view = np.ndarray((self.q._maxsize, *item['shape']), dtype=item['dtype'], buffer=self.shm.buf)
        out = self.view[item['index']].copy()
        self.sema.release()
        return out, item['meta']

    def close(self):
        '''Close queue'''
        self.q.put(None)
        self.shm.close()

    def __del__(self):
        if self.shm is not None:
            self.shm.close()
            if multiprocessing.parent_process() is None:
                self.shm.unlink()

    def __iter__(self):
        '''Iterate over frames in queue. This will block if the queue is empty. The sentinel value is None.'''
        return iter(self.get, sentinel=None)


class FrameBuffer():
    """Buffer for sharing frames between processes using multiprocessing.Array."""

    def __init__(self, shape, dtype, num_frames=32):
        import multiprocessing
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self.q = multiprocessing.JoinableQueue(num_frames)
        self.sema = multiprocessing.Semaphore(num_frames)
        self.counter = multiprocessing.Value('i', 0)
        self.array = multiprocessing.Array("c", int(np.prod(shape)) * num_frames * self.dtype.itemsize)
        self._view = None  #np.ndarray((self.q._maxsize, *self.shape), dtype=self.dtype, buffer=self.array.get_obj())

    def put(self, frame, meta=None, drop_when_full=False):
        """Put frame into queue. This will block if the queue is full. All frames have to be of the same size and dtype.
        
        Args:
            frame: frame to put into queue
            meta: optional meta data to put into queue (keep this small, as it will pass through a conventional queue)
            drop_when_full: True to drop oldest frame in queue if full, False to block if full (default)"""
        if drop_when_full and self.q.full():  #drop oldest frame
            self.q.get()
            self.sema.release()
        self.sema.acquire()
        index = self.counter.value % self.q._maxsize
        self.view[index] = frame
        self.q.put(dict(index=index, meta=meta))
        with self.counter.get_lock():
            self.counter.value += 1

    def get(self, copy=True):
        """Get frame from queue. This will block if the queue is empty.
        
        Args:
            copy: True to copy frame, False to return view. Default: True"""
        item = self.q.get()
        if item is None:
            return None
        out = self.view[item['index']]
        if copy:
            out = out.copy()
            self.sema.release()
        return out

    @property
    def view(self):
        if self._view is None:
            self._view = np.ndarray((self.q._maxsize, *self.shape), dtype=self.dtype, buffer=self.array.get_obj())
        return self._view

    @property
    def latest_frame(self):
        return self.view[(self.counter.value - 1) % self.q._maxsize]

    def __del__(self):
        pass

    def __iter__(self):
        '''Iterate over frames in queue. This will block if the queue is empty. The sentinel value is None.'''
        #return iter(self.get, sentinel=None)
        return self

    def __next__(self):
        '''Iterate over frames in queue. This will block if the queue is empty. The sentinel value is None.'''
        item = self.get()
        if item is None:
            raise StopIteration
        return item

    def close(self):
        '''Close queue'''
        self.q.put(None)
