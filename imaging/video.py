#put all imports here @copilot
import cupy as cp
import cv2
import numpy as np
from tqdm.auto import tqdm
import os
import av
import h5py
import skimage


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
                 codec='h264', 
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
        for packet in self.stream.encode():
            self.container.mux(packet)
            del packet
        self.container.close()

    def __del__(self):
        self.close()

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
        scalebar_length = int((scalebar*0.001 / (np.abs(zpos[-1]) - np.abs(zpos[0]))) * depth)
        output = cv2.line(output,
                          (width-pad, height+pad), 
                          (width-pad, height+scalebar_length+pad), 
                          color=color, 
                          thickness=4)
        # Draw a scalebar at the bottom of the yz projection
        output = cv2.line(output,
                          (width+pad, height-pad), 
                          (width+pad+scalebar_length, height-pad), 
                          color=color, 
                          thickness=4)
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


def recording_to_video(fn_rec,fps=10, 
                        codec='h264',
                        pix_fmt='yuv420p',
                        bit_rate=2 * 8e6, 
                        out_fmt="gray", 
                        template=None,
                        vmin=0, vmax=100, 
                        absolute_limits=False):
    """
    Save a recording to a video file using the AVWriter class.

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
    fn_vid = fn_rec[-3]+ f"_mip_vmin{vmin}_vmax{vmax}{"_al" if absolute_limits else ""}.mp4"
    
    with h5py.File(fn_rec, "r") as f:
        shape = f["data"].shape
        height, width = create_projection_image(f["data"][0], projection="max", vmin=vmin, vmax=vmax, 
                                                absolute_limits=absolute_limits).shape[:2]
        
        writer = AVWriter(fn_vid, height=height, width=width,codec=codec, fps=int(fps), bit_rate=int(bit_rate),
                      pix_fmt=pix_fmt, out_fmt=out_fmt, template=template)
        for i in tqdm(range(shape[0])):
            vol = f["data"][i]
            mip = create_projection_image(vol, projection="max", 
                                          vmin=vmin, vmax=vmax, 
                                          absolute_limits=absolute_limits,
                                          text=str(i))
            writer.write(mip.astype(np.uint8))
    writer.close()
    return fn_vid

def get_clipped_array(arr, vmin=0, vmax=100, absolute_limits=False):
    arr = cp.asarray(arr)
    if not absolute_limits:
        vmin = cp.percentile(arr, vmin)
        vmax = cp.percentile(arr, vmax)
        # Normalize to absolute limits
    return (cp.clip((arr - vmin)/(vmax-vmin),0,1) * 255).astype(cp.uint8)


def recording_to_overlay_preview(paths,
                                 lenses=7,
                                 radius_range=np.arange(260,310),
                                 min_distance=200,
                                 mean=True,
                                 fps=10,
                                 vmin=0,
                                 vmax=100,
                                 absolute_limits=False,
                                 df=False,
                                 write_video=True,
                                 write_file=True, 
                                 verbose=True):
    print("Loading PSF and calculating circle masks...")if verbose else None
    with h5py.File(paths.psf, "r") as f:
        # masks = cp.asarray(f["ml_masks"])
        # cx,cy,radii = cp.asarray(f["ml_cx"]), cp.asarray(f["ml_cy"]), cp.asarray(f["ml_radii"])
        # mean_psf= cp.asarray(f["psf"]).mean(axis=0).get()
        circle_mask = cp.asarray(f["circle_mask"]).get()
        
    cx,cy,radii, masks = get_lenses(circle_mask, 
                                    lenses=lenses, 
                                    radius_range=radius_range, 
                                    min_distance=min_distance)
    padxy = int(radii.max())
    print(f"setting up preview, saving {"video" if write_video else ""}{"file" if write_file else ""}...") if verbose else None
    with h5py.File(paths.raw, "r") as f:
        fn_vid = paths.pn_outrec + f"/overlay_preview{"_df"if df else ""}.mp4" 
        fn_file = paths.pn_outrec + f"/overlay{"_df"if df else ""}.h5"
        with h5py.File(fn_file, "w") as f_overlay:
            if write_file:
                f_overlay.create_dataset("data", shape=(f["data"].shape[0], padxy*2, padxy*2), dtype=np.float32)
            if write_video:
                writer = AVWriter(fn_vid, width=padxy*2, height=padxy*2, 
                                 codec='h264', fps=fps, bit_rate=2 * 8e6, 
                                 pix_fmt='yuv420p', out_fmt='gray')
            masks = cp.asarray(masks)
            if df:
                avg_loop = tqdm(range(f["data"].shape[0]), desc="Calculating mean overlay") if verbose else range(f["data"].shape[0])
                avg_sub_img = cp.zeros(shape=(padxy*2, padxy*2))
                for i in avg_loop:
                    img = f["data"][i]
                    sub_img = img_to_overlay_preview(img, cx, cy, masks, padxy, mean=mean)
                    delta = sub_img - avg_sub_img
                    avg_sub_img += delta / (i + 1)
            main_loop =tqdm(range(f["data"].shape[0]),desc="Calculating overlay") if verbose else range(f["data"].shape[0])
            for i in main_loop:
                img = f["data"][i]
                sub_img = img_to_overlay_preview(img, cx, cy, masks, padxy, mean=mean)
                if df:
                    sub_img = sub_img - avg_sub_img
                sub_img = get_clipped_array(sub_img, vmin=vmin, vmax=vmax, absolute_limits=absolute_limits).get()
                
                if write_video:
                    writer.write(sub_img.astype(np.uint8))
                if write_file:
                    f_overlay["data"][i,:,:] = sub_img.astype(np.float32)

    return fn_vid, fn_file
                    
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

def img_to_overlay_preview(img,cx,cy,masks,padxy, mean=True):
    sub_img= cp.zeros(shape=(padxy*2, padxy*2))
    for i in range(masks.shape[0]):
        sub_img +=(cp.asarray(img)*masks[i])[cy[i]-padxy:cy[i]+padxy,cx[i]-padxy:cx[i]+padxy]
    sub_img /= masks.shape[0] if mean else 1
    return sub_img



