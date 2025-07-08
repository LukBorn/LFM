#put all imports here @copilot
import cupy as cp
import cv2
import numpy as np
from tqdm.auto import tqdm
import os
import av
import h5py
import skimage
import json
import IPython
import threading
import queue
import time

class AVWriter:
    """ Write video file using av library
        
    Args:
        filename (string): full path of output file
        width (int): width of the video
        height (int): height of the video
        codec (string): codec of the video. Default is 'h264'
        fps (int): frames per second. Default is 25
        bit_rate (int): bit rate of the video. Default is 1000000
        pix_fmt (string): pixel format of the video. Default is 'yuv420p'
        out_fmt (string): output format of the video. Default is 'rgb24'
        """

    def __init__(self, filename, 
                 width=None, height=None,
                 codec='libx264', 
                 fps=25, 
                 bit_rate=1000000,
                 pix_fmt='yuv420p', 
                 out_fmt='rgb24', 
                 template=None):
        filename = os.path.expanduser(filename)
        self.container = av.open(filename, mode='w')
        if template:
            self.stream = self.container.add_stream(template=template)
        else:
            self.stream = self.container.add_stream(codec, rate=int(fps), width=int(width), height=int(height),
                                                    pix_fmt=pix_fmt, bit_rate=int(bit_rate))
        self.stream.codec_context.thread_type = 'AUTO'
        self.out_fmt = out_fmt

    def write(self, im):
        out_frame = av.VideoFrame.from_ndarray(im, format=self.out_fmt)
        for packet in self.stream.encode(out_frame):
            self.container.mux(packet)
            # del packet
        del out_frame

    def close(self):
        if hasattr(self, "_closed") and self._closed:
            return
        self._closed = True
        if hasattr(self, "stream") and self.stream is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            except Exception as e:
                print(f"AVWriter: Error during stream.encode() in close(): {e}")
            self.stream = None
        if hasattr(self, "container") and self.container is not None:
            try:
                self.container.close()
            except Exception as e:
                print(f"AVWriter: Error during container.close(): {e}")
            self.container = None

    
    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"AVWriter: Exception ignored in __del__: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class AVWriter2:
    """
    Asynchronous video writer using av, supporting out-of-order frame submission.
    """
    def __init__(self, filename, width=None, height=None,
                 codec='libx264', fps=25, bit_rate=1000000,
                 pix_fmt='yuv420p', out_fmt='gray', template=None,
                 expected_indeces=None, verbose=False,):
        filename = os.path.expanduser(filename)
        self.filename = filename
        self.codec = codec
        self.fps = int(fps)
        self.bit_rate = int(bit_rate)
        self.pix_fmt = pix_fmt
        self.out_fmt = out_fmt
        self.template = template
        self.width = width
        self.height = height

        self.verbose = verbose

        self._frame_buffer = {}
        self._expected_indeces = list(expected_indeces) if expected_indeces is not None else None
        self._written_indices = set()
        self._closed = False
        if self._expected_indeces is None or len(self._expected_indeces) == 0:
            raise ValueError("expected_indices must be specified and non-empty for AVWriter2")
        self._thread = threading.Thread(target=self._writer_thread, daemon=True)
        self._thread.start()

    def checkframe(self, im, idx=None):
        # Frame validation and adjustment (pad, dtype, etc.)
        if not isinstance(im, np.ndarray):
            raise ValueError(f"Frame is not a numpy array: {type(im)}")
        if im.dtype != np.uint8:
            Warning(f"Frame dtype is not uint8: {im.dtype}, converting to uint8.")
            im = im.astype(np.uint8)
        if self.out_fmt == 'gray':
            if im.ndim != 2:
                raise ValueError(f"Frame for 'gray' must be 2D, got shape {im.shape}")
        elif self.out_fmt == 'rgb24':
            if not (im.ndim == 3 and im.shape[2] == 3):
                raise ValueError(f"Frame for 'rgb24' must be 3D (H,W,3), got shape {im.shape}")
        # Pad to even dimensions if needed for yuv420p
        if self.pix_fmt == 'yuv420p':
            h, w = im.shape[:2]
            pad_h = h % 2
            pad_w = w % 2
            if pad_h or pad_w:
                pad_shape = ((0, pad_h), (0, pad_w)) if im.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0))
                im = np.pad(im, pad_shape, mode='edge')
                if self.verbose:
                    print(f"AVWriter2: Padding frame idx={idx} from ({h},{w}) to ({h+pad_h},{w+pad_w}) for codec compatibility.")
        return im

    def write(self, im, idx):
        if self._closed:
            raise RuntimeError("Cannot write to closed AVWriter")
        im = self.checkframe(im, idx)
        if self.width is None or self.height is None:
            self.height, self.width = im.shape[:2]
        # Just put every frame in the buffer, unlimited size
        self._frame_buffer[idx] = im

    def _init_writer(self):
        self.container = av.open(self.filename, mode='w')
        if self.template:
            self.stream = self.container.add_stream(template=self.template)
        else:
            self.stream = self.container.add_stream(
                self.codec, rate=self.fps, width=int(self.width), height=int(self.height),
                pix_fmt=self.pix_fmt, bit_rate=self.bit_rate)
        self.stream.codec_context.thread_type = 'AUTO'

    def _writer_thread(self):
        try:
            for next_idx in self._expected_indeces:
                # Wait until the next expected frame is in the buffer
                while not self._closed:
                    if next_idx in self._frame_buffer:
                        im = self._frame_buffer.pop(next_idx)
                        break
                    time.sleep(0.01)
                else:
                    break
                if not hasattr(self, "container"):
                    self._init_writer()
                if self.verbose:
                    print(f"AVWriter2: Writing frame idx={next_idx}, shape={im.shape}, dtype={im.dtype}, min={im.min()}, max={im.max()}")
                out_frame = av.VideoFrame.from_ndarray(im, format=self.out_fmt)
                for packet in self.stream.encode(out_frame):
                    self.container.mux(packet)
                del out_frame
                self._written_indices.add(next_idx)
                del im
        except Exception as e:
            print(f"AVWriter2: Exception in writer thread: {e}")
        finally:
            if hasattr(self, "stream"):
                try:
                    for packet in self.stream.encode():
                        self.container.mux(packet)
                except Exception as e:
                    print(f"AVWriter2: Error during stream.encode() in close(): {e}")
            if hasattr(self, "container"):
                try:
                    self.container.close()
                except Exception as e:
                    print(f"AVWriter2: Error during container.close(): {e}")

    def close(self):
        self._closed = True
        if hasattr(self, '_thread'):
            self._thread.join()
        if self._expected_indeces is not None:
            missing = set(self._expected_indeces) - self._written_indices
            if missing and self.verbose:
                print(f"AVWriter2 WARNING: The following frame indices were expected but not written: {sorted(missing)}")
            elif self.verbose:
                print("AVWriter2: All expected frames were written.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_projection_image(volume, 
                            projection="max", 
                            vmin=0,
                            vmax=100,
                            absolute_limits=False,
                            pad=None, 
                            text=None,
                            text_size=3,
                            scalebar=200, 
                            zpos=None,):
    """
    Creates a 2D image showing projections of a 3D volume along all three axes.

    Parameters:
    -----------
    volume : cp.ndarray or np.ndarray
        3D input volume of shape (depth, height, width)
    projection : callable, 
        "max", "mean", or "sum" to specify the type of projection
    vmin (float): 
        lower percentile (0-100) or absolute limit, depending on absolute_lims flag
    vmax (float): 
        upper percentile (0-100) or absolute limit, depending on absolute_lims flag
    absolute_lims (bool): 
        whether vmin or vmax are interpreted as absolute limits or percentiles
    text : str or None
        Text to overlay on the projection image. If None, no text is added.
    scalebar : float or None
        Length of the scalebar in um.
    zpos: array of shape volume.shape[0] or None
        z positions for the scalebar. If None, no scalebar is added.
    
    pad : int
        Padding between the projections

    Returns:
    --------
    projection_image : same type as volume
        2D image showing all three projections arranged with:
        - xy (axial) in the center
        - xz at the bottom
        - yz on the right side
    """
    volume = cp.asarray(volume)
    # Get dimensions
    depth, height, width = volume.shape

    if pad is None:
        pad = int(depth/10)

    # Calculate output dimensions with padding
    output_height = height + depth + 3 * pad
    output_width = width + depth + 3 * pad
    output = cp.zeros((output_height, output_width), dtype=volume.dtype)
    
    if projection == "max":
        projection_func = cp.max
    elif projection == "mean":
        projection_func = cp.mean
    elif projection == "sum":
        projection_func = cp.sum
    else:
        raise ValueError(f"Unknown projection type: {projection}")
            

    # XY projection (center)
    output[pad:pad + height, pad:pad + width] = projection_func(volume, axis=0)

    # XZ projection (bottom)
    output[pad + height + pad:pad + height + pad + depth, pad:pad + width] = projection_func(volume, axis=1)

    # YZ projection (right side) 
    output[pad:pad + height, pad + width + pad:pad + width + pad + depth] = projection_func(volume, axis=2).T

    output = get_clipped_array(output, vmin=vmin, vmax=vmax, absolute_limits=absolute_limits).get()

    color = 240 if not absolute_limits else int(0.95*vmax)
    if text is not None:
        font, lineType = cv2.FONT_HERSHEY_PLAIN, text_size+1
        text_width, text_height = cv2.getTextSize(text, font, text_size, lineType)[0]
        org = (output_width - pad - text_width, output_height - pad)
        output = cv2.putText(output, text, 
                             org=org, 
                             fontFace=font,
                             fontScale=text_size, 
                             color = color,
                             thickness=lineType)
    if zpos is not None:
        # Draw a scalebar at the bottom of the xz projection
        zpos-=zpos.min()
        scalebar_length = int((scalebar*0.001 / np.abs(np.abs(zpos[-1]) - np.abs(zpos[0]))) * depth)
        output = cv2.line(output,
                          (width-pad, height+pad), 
                          (width-pad, height+scalebar_length+pad), 
                          color=color, 
                          thickness=text_size)
        # Draw a scalebar at the bottom of the yz projection
        output = cv2.line(output,
                          (width+pad, height-pad), 
                          (width+pad+scalebar_length, height-pad), 
                          color=color, 
                          thickness=text_size)
        # Add text for the scalebar
        text = f"{scalebar} um"
        font, lineType = cv2.FONT_HERSHEY_PLAIN, text_size+1
        text_width, text_height = cv2.getTextSize(text, font, text_size, lineType)[0]
        org = (width, height + pad//2+ text_height)
        output = cv2.putText(output, text, 
                             org=org, 
                             fontFace=font,
                             fontScale=text_size, 
                             color = color,
                             thickness=lineType)    

    return output


def recording_to_video(paths,
                       fps=10, 
                       vmin=0, vmax=100, absolute_limits=False
                       ):
    """
    Save a fully deconvolved recording to video

    Parameters:
    -----------
    recording : iterable
        file_path of h5py with f["data"] being a 4d array of [frames, z,x,y]
    filename : str
        Full path of the output video file.
    codec : str
        Codec for the video. Default is 'h264'.
    fps : int
        Frames per second for the video. Default is 25.
    bit_rate : int
        Bit rate for the video. Default is 1000000.
    pix_fmt : str
        Pixel format for the video. Default is 'yuv420p'.
    out_fmt : str
        Output format for the video frames. Default is 'rgb24'.
    template : av.codec.CodecContext or None
        Template codec context to use for the stream. Default is None.
    """

    fn_vid = paths.deconvolved[:-3] + f"_mip_vmin{vmin}_vmax{vmax}{"_al" if absolute_limits else ""}.mp4"
    
    if os.path.exists(fn_vid):
        print(f"Video file {fn_vid} already exists. Skipping video creation.")
        return fn_vid
    
    with h5py.File(paths.deconvolved, "r") as f:
        video_writer = None
        for i in tqdm(range(f["data"].shape[0]), desc="Creating video"):
            vol = f["data"][i]
            mip = create_projection_image(vol, projection="max", 
                                          vmin=vmin, vmax=vmax, 
                                          absolute_limits=absolute_limits,
                                          text=str(i))
            if video_writer is None:
                height, width = mip.shape
                video_writer = AVWriter(fn_vid, height=height, width=width,
                                        codec = "mpeg4", fps=int(fps), 
                                        pix_fmt='yuv420p', out_fmt='gray')
            video_writer.write(mip.astype(np.uint8))
    video_writer.close()
    return fn_vid

def array3d_to_video(array,
                   filename,
                   fps=10, 
                   vmin=0, vmax=100, absolute_limits=False,
                   ):
    """
    Save a 3D array to a video file using the AVWriter class.

    Parameters:
    -----------
    array : np.ndarray or cp.ndarray
        3D array of shape (frames, height, width).
    filename : str
        Full path of the output video file.
    fps : int
        Frames per second for the video. Default is 10.
    vmin : float
        Minimum value for clipping. Default is 0.
    vmax : float
        Maximum value for clipping. Default is 100.
    absolute_limits : bool
        Whether to use absolute limits for clipping. Default is False.
    codec : str
        Codec for the video. Default is 'h264'.
    bit_rate : int
        Bit rate for the video. Default is 1000000.
    pix_fmt : str
        Pixel format for the video. Default is 'yuv420p'.
    out_fmt : str
        Output format for the video frames. Default is 'rgb24'.
    """
    
    writer = AVWriter(filename, width=array.shape[2], height=array.shape[1],
                      codec='h264', fps=int(fps), bit_rate=2 * 8e6,
                      pix_fmt='yuv420p', out_fmt='gray')
    
    for i in tqdm(range(array.shape[0]), desc="Creating video"):
        frame = array[i]
        frame = get_clipped_array(frame, vmin=vmin, vmax=vmax, absolute_limits=absolute_limits)
        writer.write(frame)
    
    writer.close()
    return filename

def array4d_to_mip_video(array, 
                        filename,
                        fps=10, 
                        vmin=0, vmax=100, absolute_limits=False,
                        ):
    writer = None
    
    for i in tqdm(range(array.shape[0]), desc="Creating video"):
        data = array[i]
        frame = create_projection_image(data,vmin=vmin, absolute_limits=absolute_limits,text=f"f{i}",)
        if writer is None:
            writer = AVWriter(filename, width=frame.shape[1], height=frame.shape[0],
                      codec='h264', fps=int(fps), bit_rate=2 * 8e6,
                      pix_fmt='yuv420p', out_fmt='gray')
        writer.write(frame)
    
    writer.close()
    return filename

def recording_to_overlay_preview(paths,
                                 cx, # x of the centers to align
                                 cy, # y of the centers to align
                                 padxy, # how far from center to plot from
                                 masks=None, # masks to use for overlay
                                 write_video=True,
                                 write_file=True, 
                                 normalize=True,
                                 min_clip=0, # minimum value to clip the image to
                                 fps=None,
                                 vmin=0,
                                 vmax=100,
                                 absolute_limits=False,
                                 dff=False,
                                 write_dff_video=True,
                                 write_dff_file=True,
                                 avg_method='exp',
                                 exp_avg_alpha=0.1,
                                 median_avg_window=20,
                                 
                                 verbose=True):
    # define all the filenames and wether to read/write
    fn_overlays = os.path.join(paths.pn_outrec,"overlay_previews")
    os.makedirs(fn_overlays, exist_ok=True)
    fn_vid = fn_overlays + f"/overlay{"_clip"+str(min_clip) if min_clip!=0 else""}{"_norm" if normalize else ""}_preview_{fps}fps_{vmin}vmin_{vmax}vmax{"_al" if absolute_limits else ""}.mp4" 
    fn_file = fn_overlays + f"/overlay{"_clip"+str(min_clip) if min_clip!=0 else""}{"_norm" if normalize else ""}.h5"
    read_file = False
    read_dff_file = False
    if os.path.exists(fn_vid):
        write_video = False
    if os.path.exists(fn_file):
        with h5py.File(fn_file, "r") as f:
            match = (np.array_equal(f.attrs.get("cx", []), cx) and
                     np.array_equal(f.attrs.get("cy", []), cy) and
                     f.attrs["padxy"] == padxy and
                      np.array_equal(f["masks"], masks))
        if match:
            read_file = True
            write_file = False
        else:
            print(f"Overlay file {fn_file} exists but does not match the parameters. Will overwrite.") if verbose else None
    if dff:
        fn_dff_vid= fn_overlays + f"/overlay{"_clip"+str(min_clip) if min_clip!=0 else""}{"_norm" if normalize else ""}_preview_dff_{avg_method}avg_{exp_avg_alpha if avg_method=="exp" else median_avg_window}_{fps}fps_{vmin}vmin_{vmax}vmax{"_al" if absolute_limits else ""}.mp4"
        fn_dff_file = fn_overlays + f"/overlay{"_clip"+str(min_clip) if min_clip!=0 else""}{"_norm" if normalize else ""}_dff_{avg_method}avg_{exp_avg_alpha if avg_method=="exp" else median_avg_window}.h5"
        if os.path.exists(fn_dff_vid):
            write_dff_video = False 
        if os.path.exists(fn_dff_file):
            read_dff_file = True
            write_dff_file = False
        else:
            read_dff_file = False
        avg = None
        window_buffer = []
    else:
        write_dff_video = False
        write_dff_file = False

    fps = json.loads(paths.meta)["acquisition"]["fps"] if fps is None else fps

    #initialize the video writers
    if write_video:
        writer = AVWriter(fn_vid, width=padxy*2, height=padxy*2, 
                            codec='h264', fps=fps, bit_rate=2 * 8e6, 
                            pix_fmt='yuv420p', out_fmt='gray')
    if write_dff_video:
        writer_dff = AVWriter(fn_dff_vid, width=padxy*2, height=padxy*2, 
                                codec='h264', fps=fps, bit_rate=2 * 8e6, 
                                pix_fmt='yuv420p', out_fmt='gray')

    with h5py.File(paths.raw, "r") as f:
        with h5py.File(fn_file, "a") as f_overlay:
            with h5py.File(fn_dff_file, "a") as f_overlay_dff:
                if write_file:
                    # Delete existing datasets if they exist to avoid "name already exists" error
                    if "data" in f_overlay:
                        del f_overlay["data"]
                    if "masks" in f_overlay:
                        del f_overlay["masks"]
                    
                    f_overlay.create_dataset("data", shape=(f["data"].shape[0], padxy*2, padxy*2), dtype=np.float32)
                    f_overlay.attrs["cx"] = cx
                    f_overlay.attrs["cy"] = cy
                    f_overlay.attrs["padxy"] = padxy
                    f_overlay["masks"] = masks
                if write_dff_file:
                    # Delete existing dataset if it exists
                    if "data" in f_overlay_dff:
                        del f_overlay_dff["data"]
                    
                    f_overlay_dff.create_dataset("data", shape=(f["data"].shape[0], padxy*2, padxy*2), dtype=np.float32)
                print("test2")
                main_loop =tqdm(range(f["data"].shape[0]),desc="Calculating overlay") if verbose else range(f["data"].shape[0])
                for i in main_loop:
                    if read_file:
                        sub_img = cp.asarray(f_overlay["data"][i])
                    else: 
                        img = f["data"][i]
                        if normalize:
                            img = (cp.asarray(img)/cp.asarray(img).flatten().mean()).get()
                        sub_img = img_to_overlay_preview(img, cx, cy, masks, padxy,mean=False)
                        sub_img= sub_img.clip(1, None)
                    if read_dff_file:
                        sub_img_dff = cp.asarray(f_overlay_dff["data"][i])
                    elif dff:
                        if avg_method == 'exp':
                            avg = sub_img.copy() if avg is None else exp_avg_alpha * sub_img + (1 - exp_avg_alpha) * avg
                        
                        elif avg_method == 'median':
                            # Sliding median window
                            window_buffer.append(sub_img.copy())
                            if len(window_buffer) > median_avg_window:
                                window_buffer.pop(0)
                            if len(window_buffer) >= 5:  # Need at least a few frames
                                avg = cp.median(cp.stack(window_buffer), axis=0)
                            else:
                                avg = sub_img.copy()
                        # Calculate ΔF/F
                        sub_img_dff = (sub_img - avg)/sub_img
                        
                    
                    if write_file:
                        f_overlay["data"][i,:,:] = sub_img.get().astype(np.float32)
                    if write_dff_file:
                        f_overlay_dff["data"][i,:,:] = sub_img_dff.get().astype(np.float32)
                    
                    if write_video:
                        sub_img = get_clipped_array(sub_img, vmin=vmin, vmax=vmax, absolute_limits=absolute_limits).get()
                        writer.write(sub_img)
                    if write_dff_video:
                        sub_img_dff = get_clipped_array(sub_img_dff, vmin=vmin, vmax=vmax, absolute_limits=absolute_limits).get()    
                        writer_dff.write(sub_img_dff.astype(np.uint8))
                

    return fn_vid,fn_file,fn_dff_vid if dff else None, fn_dff_file if dff else None
                    
def get_lenses(im,lenses = 7, radius_range=np.arange(260,310), min_distance=200):
    edges = skimage.feature.canny(im,sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    hough_res = skimage.transform.hough_circle(edges,radius_range)
    _ , cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res,
                                                             radius_range,
                                                             total_num_peaks=lenses,
                                                             min_xdistance=min_distance,
                                                             min_ydistance=min_distance)
    masks = np.zeros(shape=(lenses, im.shape[0], im.shape[1]), dtype=bool)
    for i in range(lenses):
        masks[i][skimage.draw.disk((cy[i], cx[i]), radii[i], shape=im.shape)] = 1
    return cx,cy,radii, masks

def img_to_overlay_preview(img,cx,cy,masks,padxy,mean=True):
    sub_img = cp.zeros(shape=(padxy*2, padxy*2))
    if masks is not None:
        masks = cp.asarray(masks)
        for i in range(masks.shape[0]):
            sub_img +=(cp.asarray(img)*masks[i])[cy[i]-padxy:cy[i]+padxy,cx[i]-padxy:cx[i]+padxy]
    else:
        for i in range(masks.shape[0]):
            sub_img += (cp.asarray(img))[cy[i]-padxy:cy[i]+padxy,cx[i]-padxy:cx[i]+padxy]
    if mean:
        sub_img /= np.array(cx).shape[0]
    return sub_img

def get_clipped_array(arr, vmin=0, vmax=100, absolute_limits=False):
    arr = cp.asarray(arr)
    if not absolute_limits:
        # Normalize to percentiles (linear interpolation between min and max)
        arr_min = arr.min()
        arr_max = arr.max()
        vmin = arr_min + vmin/100 * (arr_max - arr_min)
        vmax = arr_min + vmax/100 * (arr_max - arr_min)
    # When absolute_limits=True, use vmin and vmax as provided
    return (cp.clip((arr - vmin)/(vmax-vmin),0,1) * 255).astype(cp.uint8)

def showvid(filename, width=600, embed=False, loop=True):
    """
    Display a video in a Jupyter notebook.
    Args:
        filename (str): Path to the video file.
        width (int): Width of the video display.
        embed (bool): Whether to embed the video in the notebook.
        loop (bool): Whether to loop the video.
    """
    try:
        from IPython.display import Video, display
    except ImportError:
        raise ImportError(
            "The 'IPython' package is required to display videos. Please install it."
        )

    html_attributes = "controls loop" if loop else "controls"
    display(Video(filename, embed=embed, width=width, html_attributes=html_attributes))