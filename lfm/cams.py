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

    def stream(self, num_frames, callback=None, interrupt=None, stream_to_disk_path=None, fifo=True, already_armed=False, verbose=True):
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
        looper = tqdm(range(num_frames)) if verbose else range(num_frames)
        for i_frame in looper:

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

    def acquire_stack(self, num_frames, verbose=True):
        '''Acquire stack of frames

        Args:
            num_frames: number of frames to acquire
        Return:
            im_stack: stack of frames
            timestamps: timestamps of frames
        '''
        im_stack = np.zeros((num_frames, *self.frame_shape), dtype=self.frame_dtype)
        timestamps = np.zeros(num_frames+1)
        frame_counts = np.zeros(num_frames)

        def callback(im, i_frame, timestamp, frame_count):
            im_stack[i_frame] = im
            timestamps[i_frame] = timestamp
            frame_counts[i_frame] = frame_count

        self.stream(num_frames, callback=callback,verbose=verbose)
        timestamps[-1]=time.time()
        return im_stack, timestamps, frame_counts

    '''
    def preview_with_controls(self, stage, step, window_shape=(1024, 768), filter_fcn=lambda x: x, fifo=False,range=0.005):
        #gave up on this bc too frustrating,  just use normal preivew and the XiLAB
        # 
        # A preview function with custom controls for live camera streaming.
        # Displays a video feed in the top half of the window and adds control buttons in the bottom half.
        # 
        # Args:
        #     stage: The current stage object (passed from LFM preview_psf).
        #     step: The step value used for controlling movement or other actions.
        #     window_shape: shape of preview window (None for full frame, same as in `preview`).
        #     filter_fcn: optional function to apply to each frame before display.
        #     fifo: Whether to use the frame FIFO buffer (same as in `preview`).
        # 
        import cv2
        if window_shape is None:
            window_shape = (self.frame_shape[1] // 2, self.frame_shape[0] // 2)

        # Set up the preview window
        window_name = 'preview_with_controls'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *window_shape)
        cv2.setWindowTitle(window_name, 'Preview with Controls')
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Button definitions
        button_labels = [
            ("Forward", "↑"), ("Backward", "↓"),
            ("Left", "←"), ("Right", "→"),
            ("Up", "PgUp"), ("Down", "PgDn"), ("Test Record", "Space")
        ]
        buttons = []
        button_height = 60
        top_video_area = int(window_shape[1] * 0.6)  # Top 60% for video feed
        button_margin = 10
        button_width = (window_shape[0] - (button_margin * (len(button_labels) + 1))) // len(button_labels)
        button_pressed_states = {label: False for label, _ in
                                 button_labels}  # Keep track of whether buttons are pressed

        # Calculate button regions
        for i, (label, key) in enumerate(button_labels):
            x1 = button_margin + i * (button_width + button_margin)
            y1 = top_video_area + button_margin
            x2 = x1 + button_width
            y2 = y1 + button_height
            buttons.append({"label": label, "key": key, "region": (x1, y1, x2, y2), "color": (70, 70, 70)})

        # Define mouse click callback
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for button in buttons:
                    x1, y1, x2, y2 = button["region"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        if not button_pressed_states[button["label"]]:  # Check if button was not already pressed
                            button_pressed_states[button["label"]] = True
                            handle_button_action(button["label"])
                            button["color"] = (0, 255, 0)  # Change button color to indicate the click
                        break
            elif event == cv2.EVENT_LBUTTONUP:
                for button in buttons:
                    button_pressed_states[button["label"]] = False  # Reset state when mouse is released

        # Attach mouse callback to the window
        cv2.setMouseCallback(window_name, mouse_callback)

        def handle_button_action(label):
            """Handles the functionality associated with each button."""
            if label == "Forward":
                # Code for moving forward
                stage.move((step,0,0))
            elif label == "Backward":
                # Code for moving backward
                stage.move((-step,0,0))
            elif label == "Left":
                # Code for moving left
                stage.move((0,step,0))
            elif label == "Right":
                # Code for moving right
                stage.move((0,-step,0))
            elif label == "Up":
                # Code for moving up
                stage.move((0,0,step))
            elif label == "Down":
                stage.move((0,0,-step))
            elif label == "Test Record":
                stage.move((0,0,-range))
                stage.move((0,0,range))
            print(f"Action executed for button: {label}")  # Placeholder for visual feedback

        def draw_buttons(canvas):
            """Draws buttons on the bottom portion of the preview window."""
            for button in buttons:
                x1, y1, x2, y2 = button["region"]
                label, key = button["label"], button["key"]
                color = button["color"]

                # Draw button rectangle
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Add text label
                text = f"{label} ({key})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x1 + (button_width - text_size[0]) // 2
                text_y = y1 + (button_height + text_size[1]) // 2
                cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        def preview_callback(im_np, i_frame, timestamp, frame_count):
            # Create a blank canvas for the display
            canvas = np.zeros((window_shape[1], window_shape[0], 3), dtype=np.uint8)
            # Resize the video frame (if needed)
            video_frame_resized = cv2.resize(im_np, (window_shape[0], top_video_area))  # Assuming resize logic here

            video_frame_resized_rgb = cv2.cvtColor(video_frame_resized, cv2.COLOR_GRAY2BGR)

            # Add the video feed at the top
            canvas[0:top_video_area, :, :] = video_frame_resized_rgb

            # Draw buttons on the bottom portion
            draw_buttons(canvas)

            # Show the canvas
            cv2.imshow(window_name, canvas)

            # Interrupt logic to terminate stream when the window is closed
        interrupt = lambda: (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1)

        # Start streaming
        self.stream(int(1e9), callback=preview_callback, interrupt=interrupt, fifo=fifo)
        cv2.destroyAllWindows()
'''

    def preview(self, window_shape=None, filter_fcn=lambda x: x, fifo=False, rate = 5):
        '''Simple camera preview. Press ESC to end.
        
        Args:
            window_shape: shape of preview window (None for full frame)
            filter_fcn: optional function to apply to each frame before display
            rate: rate at which to update preview (in Hz)
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
            if i_frame % rate == 0:
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

from pyDCAM import dcamapi_init, HDCAM, DCAMIDPROP, DCAMPROPMODEVALUE, dcamapi_uninit
class DCamera(CameraBase):
    '''Hamamatsu DCAM camera using pyDCAM
       C:/Users/jlab/Desktop/dcam/dcamsdk4/doc/camera_properties/propC13440-20CU_en.html
    '''

    def __init__(self, conf=None):
        '''Initialize the camera'''
        device_count = dcamapi_init()
        self.hdcam = HDCAM(range(device_count)[0])
        #self.hdcam.readout_speed = DCAMPROPMODEVALUE.DCAMPROP_READOUTSPEED__SLOWEST
        self.set_trigger(external = False)
        self._buffer_frames = 256
        self._sensor_shape = self.frame_shape
        self._fifo = True
        self._last_frame = None

    @property
    def roi(self):
        '''Get ROI as a dictionary'''
        y_size, x_size = self.hdcam.subarray_size
        y_offset, x_offset = self.hdcam.subarray_pos
        bin = self.hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_BINNING)
        match bin:
            case DCAMPROPMODEVALUE.DCAMPROP_BINNING__1:
                binning=1
            case DCAMPROPMODEVALUE.DCAMPROP_BINNING__2:
                binning = 2
            case DCAMPROPMODEVALUE.DCAMPROP_BINNING__4:
                binning = 4

        return dict(y_size= y_size,
                    x_size= x_size,
                    y_offset= y_offset,
                    x_offset= x_offset,
                    y_bin= binning,
                    x_bin= binning)

    def set_roi(self, y_size=None, x_size=None, y_offset=0, x_offset=0, y_bin:int=1, x_bin:int=1):
        '''Set the Region of Interest (ROI)'''
        if y_size is None:
            y_size = self.sensor_shape[0]
        if x_size is None:
            x_size = self.sensor_shape[1]
        if [(y_size, x_size), y_offset, x_offset] == [self.sensor_shape, 0,0]:
            self.hdcam.subarray_mode = False
        else:
            self.hdcam.subarray_mode = True
            self.hdcam.subarray_size = (y_size, x_size)
            self.hdcam.subarray_pos = (y_offset,x_offset)

        assert x_bin == y_bin, "only symmetric binning is supported: y_bin must == x_bin"
        match y_bin:
            case 1:
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_BINNING, DCAMPROPMODEVALUE.DCAMPROP_BINNING__1)
            case 2:
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_BINNING, DCAMPROPMODEVALUE.DCAMPROP_BINNING__2)
            case 4:
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_BINNING, DCAMPROPMODEVALUE.DCAMPROP_BINNING__4)


    def set_trigger(self, external=True, each_frame=True):
        '''Set trigger mode'''
        if external:
            if each_frame:
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_TRIGGER_MODE, DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__NORMAL)
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE, DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__EXTERNAL)
                self.trigger = "External EachFrame"
            else:
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_TRIGGER_MODE, DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__START)
                self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE, DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__EXTERNAL)
                self.trigger = "External Start"
        else:
            self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_TRIGGER_MODE, DCAMPROPMODEVALUE.DCAMPROP_TRIGGER_MODE__NORMAL)
            self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_TRIGGERSOURCE, DCAMPROPMODEVALUE.DCAMPROP_TRIGGERSOURCE__INTERNAL)
            self.trigger = "Internal"

    @property
    def exposure_time(self):
        '''Get the exposure time (in seconds)'''
        return self.hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_EXPOSURETIME)

    @exposure_time.setter
    def exposure_time(self, t):
        '''Set the exposure time (in seconds)'''
        self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_EXPOSURETIME, t)

    def arm(self, stream_to_disk_path=None, fifo=True):
        '''Start acquisition (live mode)'''
        self.hdcam.dcambuf_alloc(self._buffer_frames)
        if self.trigger[0] =="E": #external trigger 
            hwait = self.hdcam.dcamwait_open()
        self.hdcam.dcamcap_start()
        if self.trigger[0] == "E":
            hwait.dcamwait_start()
        self._fifo = fifo
        self._last_frame = None

    def disarm(self):
        '''Stop acquisition'''
        self.hdcam.dcamcap_stop()
        self.hdcam.dcambuf_release()

    def poll_frame(self, copy=False):
        '''Retrieve a frame from the camera'''
        hwait = self.hdcam.dcamwait_open()
        hwait.dcamwait_start(timeout = 5000)
        frame_index, frame_count = self.hdcam.dcamcap_transferinfo()

        if frame_index < 0:
            logger.warning('No frames captured')
        frame_data = self.hdcam.dcambuf_copyframe(frame_index)
        if copy:
            frame_data = np.copy(frame_data)
        # logger.warning(f"{frame_index, timestamp}")
        if self._fifo and self._last_frame is not None and frame_count - self._last_frame > 1:
            logger.warning(f'{frame_count - self._last_frame - 1} frames dropped')
        self._last_frame = frame_count
        return frame_data, time.time(), frame_count

    @property
    def frame_dtype(self):
        '''Get the data type of frames (e.g., uint8, uint16)'''
        bit_depth = self.hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_PIXELTYPE)
        if bit_depth == 2.:
            return "uint16"
        elif bit_depth == 1.:
            return "uint8"
        elif bit_depth == 3.: #12bit
            self.frame_dtype = "uint8"
            return "uint8"

    @frame_dtype.setter
    def frame_dtype(self, dtype):
        if dtype == "uint8":
            self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_PIXELTYPE, 1.)
        elif dtype == "uint16":
            self.hdcam.dcamprop_setvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_PIXELTYPE, 2.)
        else:
            print("dtype not supported idiont")

    @property
    def frame_shape(self):
        '''Get the shape of frames (height, width)'''
        return[int(self.hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_HEIGHT)),
               int(self.hdcam.dcamprop_getvalue(DCAMIDPROP.DCAM_IDPROP_IMAGE_WIDTH))]


    @property
    def sensor_shape(self):
        '''Get the full sensor resolution as (height, width)'''

        return self._sensor_shape

    def close(self):
        '''Clean up and close camera connection'''
        self.hdcam.dcamdev_close()
        dcamapi_uninit()
        
    def __del__(self):
        '''Clean up and close camera connection'''
        self.close()




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
