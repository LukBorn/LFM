import os
import numpy as np
import cupy as cp
from tqdm.auto import tqdm
import h5py
from daio.h5 import lazyh5
from i_o import AsyncH5Writer, VolumeReader
import cupyx
import matplotlib.pyplot as plt
from video import get_clipped_array





class NeighborhoodCovMapper:
    """ Sequentially aggregates local covariance matrix of a volume when calling .step(). 
    When done, retrieve with .retrieve(). Called within the registration main loop.

    Args:
        tau (scalar): decay time constant for estimating the running background
    """

    def __init__(self, tau=60):
        self.init_frames = 2 * tau
        self.alpha = cp.float32(1 - 1 / tau)
        self.nn_kernel = cp.ones((3, 3, 3), dtype='float32')
        self.nn_kernel[1, 1, 1] = 0
        self.nn_kernel /= self.nn_kernel.sum()
        self.reset()

    def step(self, vol):
        vol = cp.array(vol, dtype='float32', copy=False)
        if self.k == 0:
            self.mu_running = vol
        else:
            self.mu_running = self.alpha * self.mu_running + (1 - self.alpha) * vol
        if self.k >= self.init_frames:
            #print(f"updating covariance map of shape: {self.cov_int.shape}")
            temp = (vol - self.mu_running)
            self.var_int = self.var_int + (temp * temp)
            temp *= cupyx.scipy.ndimage.convolve(temp, self.nn_kernel)
            self.cov_int = self.cov_int + temp
        self.k += 1

    def reset(self):
        self.mu_running = cp.array(0, 'float32')
        self.cov_int = cp.array(0, 'float32')
        self.var_int = cp.array(0, 'float32')
        self.k = 0

    def retrieve(self):
        if self.k <= self.init_frames:
            import warnings
            warnings.warn("Number of steps needs to be larger than init_frames")
        cov_map = (self.cov_int / (self.k - self.init_frames)).get()
        var_map = (self.var_int / (self.k - self.init_frames)).get()
        self.reset()
        return cov_map, var_map


class NeighborhoodCorrMapper:
    """ Sequentially aggregates local correlation of a volume when calling .step(). 
    When done, retrieve with .retrieve(). Called within the registration main loop.

    Args:
        tau (scalar): decay time constant for estimating the running background
    """

    def __init__(self, tau=60):
        self.init_frames = 2 * tau
        self.alpha = cp.float32(1 - 1 / tau)
        self.nn_kernel = cp.ones((3, 3, 3), dtype='float32')
        self.nn_kernel[1, 1, 1] = 0
        self.nn_kernel /= self.nn_kernel.sum()
        self.reset()

    def step(self, vol):
        vol = cp.array(vol, dtype='float32', copy=False)
        if self.k == 0:
            self.mu_running = vol
            self.mu2_running = vol**2
            self.corr_int = 0 * vol
        else:
            self.mu_running = self.alpha * self.mu_running + (1 - self.alpha) * vol
            self.mu2_running = self.alpha * self.mu2_running + (1 - self.alpha) * vol**2
        if self.k >= self.init_frames:
            temp = (vol - self.mu_running) / cp.sqrt(self.mu2_running - self.mu_running**2)
            temp *= cupyx.scipy.ndimage.convolve(temp, self.nn_kernel)
            self.corr_int += temp
        self.k += 1

    def reset(self):
        self.mu_running = cp.array(0, 'float32')
        self.mu2_running = cp.array(0, 'float32')
        self.corr_int = cp.array(0, 'float32')
        self.k = 0

    def retrieve(self):
        if self.k <= self.init_frames:
            import warnings
            warnings.warn("Number of steps needs to be larger than init_frames")
        corr_map = (self.corr_int / (self.k - self.init_frames)).get()
        self.reset()
        return corr_map

def ball_kernel(radius):
    """ Make a ball kernel

    Args:
        radius (list or array_like): list of radii, one per output dimension

    Return:
        array_like: ball kernel array (dtype: bool)
    """
    X, Y, Z = np.meshgrid(*[np.r_[-np.floor(radius[ii]):np.floor(radius[ii]) + 1] for ii in range(3)], indexing='ij')
    s = X * X / radius[0]**2 + Y * Y / radius[1]**2 + Z * Z / radius[2]**2
    footprint = (s <= 1)
    return footprint


def soft_ball(radius, k=5):
    """ Make a ball kernel with soft edges

    Args:
        radius (list or array_like): list of radii, one per output dimension
        k (float): logistic smoothing parameter

    Return:
        array_like: ball kernel array
    """
    X, Y, Z = np.meshgrid(*[np.r_[-np.ceil(radius[ii] * (1 + 1 / k)):np.ceil(radius[ii] * (1 + 1 / k)) + 1] for ii in range(3)], indexing='ij')
    d = X * X / radius[0]**2 + Y * Y / radius[1]**2 + Z * Z / radius[2]**2
    footprint = 1 / (1 + np.exp(-(1-d)*k))
    return footprint


def findmaxima_gpu(vol, radius, thresh=0):
    """ Find local maxima, given radius

    Args:
        vol (array_like): input array
        radius (list or array_like): list of radii, one per dimension

    Return:
        array_like: N-D binary array of maxima
        tuple: indices of peaks
        array_like: 1-D list of local maximum values
    """
    in_class = vol.__class__
    vol = cp.array(vol, 'float32', copy=False)
    kernel = cp.array(ball_kernel(radius))
    maxima_bool = (cupyx.scipy.ndimage.maximum_filter(vol, footprint=kernel) == vol)
    maxima_bool &= vol > thresh
    peaks = cp.nonzero(cp.array(maxima_bool))
    peak_vals = vol[peaks]
    if in_class == 'numpy.ndarray':
        maxima_bool = maxima_bool.get()
        peaks = tuple(i.get() for i in peaks)
        peak_vals = peak_vals.get()
    return maxima_bool, peaks, peak_vals

#consider moving to ndimage
def dogfilter_gpu(vol, sigma_low=1, sigma_high=4, mode='reflect'):
    """ Diffference of Gaussians filter

    Args:
        vol (array_like): data to be filtered
        sigma_low (scalar or sequence of scalar): standard deviations
        sigma_high (scalar or sequence of scalar): standard deviations
        mode (str): The array borders are handled according to the given mode
   
    Returns:
        (array_like): filtered data
    
    See also:
        cupyx.scipy.ndimage.gaussian_filter
        skimage.filters.difference_of_gaussians
    """
    in_module = vol.__class__.__module__
    vol = cp.asarray(vol, 'float32')
    out = cupyx.scipy.ndimage.gaussian_filter(vol, sigma_low, mode=mode)
    out -= cupyx.scipy.ndimage.gaussian_filter(vol, sigma_high, mode=mode)
    if in_module == 'numpy':
        out = out.get()
    return out


def segment_old(paths,
            dog_sigma_low=[1.5,0.5,0.5],
            dog_sigma_high=[4,4,4],
            search_min_radius=[1,1,1],
            search_min_brightness=0.003,
            soft_ball_kernel=[6,3.5,3.5],
            soft_ball_k=2,
            transpose = True, 
            save = True
            ):
    fn = os.path.expanduser(os.path.join(paths.pn_outrec, f"segmentation_dog{dog_sigma_low}-{dog_sigma_high}_minrad{search_min_radius}_tresh{search_min_brightness}_kernel{soft_ball_kernel}_k{soft_ball_k}"))
    assert os.path.exists(paths.registered), "recording must be registered"
    with h5py.File(paths.registered, 'r') as f:
        cov_map = cp.asarray(f["cov_map"])
    cov_map_f = dogfilter_gpu(cov_map, 
                              sigma_low=dog_sigma_low, 
                              sigma_high=dog_sigma_high)
    maxima_bool, peaks, peak_vals = findmaxima_gpu(cov_map_f, 
                                                   radius=search_min_radius, 
                                                   thresh=search_min_brightness)
    kernel = cp.array(soft_ball(soft_ball_kernel, soft_ball_k), 'float32')
    kernel /= kernel.sum()
    peaks_mip = create_projection_image_with_peaks(cov_map_f, 
                                                   peaks,
                                                   marker_radius=soft_ball_kernel,
                                                   marker_k=soft_ball_k,
                                                   marker_intensity=0.8,
                                                   vmax=2*search_min_brightness,
                                                   absolute_limits=True,
                                                   transpose=transpose)
    
    
    

    segmentation_result = dict(params = dict(dog_sigma_low=dog_sigma_low,
                                             dog_sigma_high=dog_sigma_high,
                                             search_min_radius=search_min_radius,
                                             search_min_brightness=search_min_brightness,
                                             soft_ball_kernel=soft_ball_kernel,
                                             soft_ball_k=soft_ball_k),
                               cov_map_f=cov_map_f, 
                               peaks=peaks, 
                               peak_vals=peak_vals)
    if save:
        plt.imsave(fn + '.png', peaks_mip)
        lazyh5(fn + '.h5').from_dict(segmentation_result)
        save_segmentation_result_tif(cov_map_f, 
                                     peaks, 
                                     output_fn=fn + '.tif',
                                     marker_radius=soft_ball_kernel,
                                     marker_k=soft_ball_k,
                                     vmax=2*search_min_brightness,
                                     absolute_limits=True,)

    return segmentation_result, peaks_mip

def extract_traces_old(paths,
                    dog_sigma_low=[1.5,1.5,1.5],
                    dog_sigma_high=[4,4,4],
                    min_radius=[3,3,3],
                    soft_ball_kernel=[6,3,3],
                    soft_ball_k=2,
                   
                    save=True,
                    ):
    assert os.path.exists(paths.registered), "recording must be registered"
    with h5py.File(paths.registered, 'r') as f:
        cov_map = cp.asarray(f["cov_map"])
    reader = VolumeReader(paths.registered, "data")
    cov_map_f = dogfilter_gpu(cov_map, sigma_low=dog_sigma_low, sigma_high=dog_sigma_high)
    maxima_bool, peaks, peak_vals = findmaxima_gpu(cov_map_f, min_radius = min_radius)

    kernel = cp.array(soft_ball(soft_ball_kernel, soft_ball_k=soft_ball_k), 'float32')
    kernel /= kernel.sum()
    traces = []

    
    for i_frame, vol in tqdm(reader):
        vol = cupyx.scipy.ndimage.convolve(cp.asarray(vol), kernel) 
        traces.append(vol[peaks].get())
    
    traces_results = dict(
        traces=np.array(traces),
        cov_vals=peak_vals.get(),
        indices=peaks,
        dataset_name=paths.dataset_name,
    )
    if save:
        with h5py.File(paths.traces, 'w') as f:
            for key, value in traces_results.items():
                f.create_dataset(key, data=value)
    return traces_results
import cupyx.scipy.ndimage
def extract_traces(paths, 
                   params_dict={},):
     
    
    # Initialize the video reader
    reader = VolumeReader(paths.registered, "data", prefetch=10)

    with h5py.File(paths.segmentation, "r") as f:
        segmentation = np.array(f["segmentation"])
    
    # Get all unique region labels, excluding 0 (background)
    labels_gpu = cp.asarray(segmentation)
    unique_labels = cp.unique(labels_gpu)
    unique_labels = unique_labels[unique_labels != 0]
    
    writer = AsyncH5Writer(paths.traces)
    writer.write_meta("params", params_dict)
    writer.create_dataset("traces", shape = (reader.get_shape("data")[0], unique_labels.shape[0]), dtype = np.float32)
    writer.write_dataset("segmentation", segmentation)
    
    
    for i_frame, vol in tqdm(reader, desc="Extracting mean traces"):
        vol_gpu = cp.asarray(vol)
        frame_means = cupyx.scipy.ndimage.mean(vol_gpu, labels=labels_gpu, index=unique_labels)
        writer.write("traces", frame_means.get(), i_frame)
    writer.close()


def extract_traces_voxels(paths,
                          voxel_size = [3,2,2],
                          params_dict={}):
    # Initialize the video reader
    reader = VolumeReader(paths.registered, "data", prefetch=10)

    with h5py.File(paths.segmentation, "r") as f:
        segmentation = np.array(f["segmentation"])

    labels_gpu = cp.asarray(segmentation)
    unique_labels = cp.unique(labels_gpu)
    unique_labels = unique_labels[unique_labels != 0]

    num_voxels = voxel_size[0]*voxel_size[1]*voxel_size[2]

    writer = AsyncH5Writer(paths.traces)
    writer.write_meta("params", params_dict)
    writer.create_dataset("traces", shape=(reader.get_shape("data")[0], unique_labels.shape[0]), dtype=np.float32)
    writer.write_dataset("segmentation", segmentation)

    # Precompute a flat label array for bincount
    flat_labels = labels_gpu.ravel()
    for i_frame, vol in tqdm(reader, desc="Extracting sum traces"):
        vol_gpu = cp.asarray(vol)
        flat_vol = vol_gpu.ravel()
        # bincount will sum values for each label
        sums = cp.bincount(flat_labels, weights=flat_vol, minlength=unique_labels.max() + 1)
        # Only keep nonzero labels (skip background)
        frame_means = sums[unique_labels]/num_voxels
        writer.write("traces", frame_means.get(), i_frame)
    writer.close()
    
def create_projection_image_with_peaks(volume, peaks,
                                      projection="max", 
                                      slice_idx=None,
                                      vmin=0,
                                      vmax=100,
                                      absolute_limits=False,
                                      pad=None,
                                      marker_radius=[2, 2, 2], 
                                      marker_k=5,
                                      marker_intensity=1.0,
                                      transpose=False,
                                      gpu=True):
    """
    Creates a 2D RGB image showing projections of a 3D volume with red markers at peak locations.
    """
    if isinstance(volume, cp.ndarray) and not gpu:
        volume = volume.get()
    volume = cp.asarray(volume) if gpu else volume
    
    # Store original dimensions before transpose
    orig_depth, orig_height, orig_width = volume.shape
    
    if transpose:
        volume = cp.transpose(volume, (0, 2, 1))  # Transpose to (depth, width, height)
    
    # Get dimensions after potential transpose
    depth, height, width = volume.shape

    if pad is None:
        pad = int(depth/10)

    # Calculate output dimensions with padding
    output_height = height + depth + 3 * pad
    output_width = width + depth + 3 * pad
    
    # Create grayscale projection first
    output = cp.zeros((output_height, output_width), dtype=volume.dtype) if gpu else np.zeros((output_height, output_width), dtype=volume.dtype)

    if projection == "max":
        projection_func = cp.max if gpu else np.max
    elif projection == "mean":
        projection_func = cp.mean if gpu else np.mean
    elif projection == "sum":
        projection_func = cp.sum if gpu else np.sum
    elif projection == "slice":
        if slice_idx is None:
            slice_idx = [depth // 2, height // 2, width // 2]
        def projection_func(vol, axis):
            return vol[slice_idx[0], :, :] if axis == 0 else vol[:, slice_idx[1], :] if axis == 1 else vol[:, :, slice_idx[2]]
    else:
        raise ValueError(f"Unknown projection type: {projection}")

    # XY projection (center)
    xy_proj = projection_func(volume, axis=0)
    output[pad:pad + height, pad:pad + width] = xy_proj

    # XZ projection (bottom)
    xz_proj = projection_func(volume, axis=1)
    output[pad + height + pad:pad + height + pad + depth, pad:pad + width] = xz_proj

    # YZ projection (right side) 
    yz_proj = projection_func(volume, axis=2).T
    output[pad:pad + height, pad + width + pad:pad + width + pad + depth] = yz_proj

    # Clip and normalize base projection
    output = get_clipped_array(output, vmin=vmin, vmax=vmax, absolute_limits=absolute_limits, gpu=gpu)
    output = output.get() if gpu else output
    
    # Convert to RGB
    rgb_output = np.stack([output, output, output], axis=-1).astype(np.float64)
    
    # Create marker kernel
    marker_kernel = soft_ball(marker_radius, k=marker_k)
    marker_kernel = marker_kernel / marker_kernel.max() * marker_intensity
    
    # Extract peak coordinates - these are in ORIGINAL coordinate system
    z_coords, y_coords, x_coords = peaks
    
    # Transform peak coordinates if transposed
    if transpose:
        # Original peaks are (z, y, x) but after transpose volume is (z, x, y)
        # So we need to swap y and x coordinates for the transposed volume
        y_coords_adj = x_coords  # original x becomes new y
        x_coords_adj = y_coords  # original y becomes new x
        z_coords_adj = z_coords  # z stays the same
    else:
        z_coords_adj = z_coords
        y_coords_adj = y_coords  
        x_coords_adj = x_coords
    
    # Create empty marker projections
    xy_markers = np.zeros((height, width), dtype=np.float64)
    xz_markers = np.zeros((depth, width), dtype=np.float64)
    yz_markers = np.zeros((height, depth), dtype=np.float64)
    
    # Add markers for each peak using adjusted coordinates
    for i in range(len(z_coords_adj)):
        z, y, x = int(z_coords_adj[i]), int(y_coords_adj[i]), int(x_coords_adj[i])
        
        # Get marker bounds
        kz, ky, kx = marker_kernel.shape
        z_start, z_end = max(0, z - kz//2), min(depth, z + kz//2 + 1)
        y_start, y_end = max(0, y - ky//2), min(height, y + ky//2 + 1)
        x_start, x_end = max(0, x - kx//2), min(width, x + kx//2 + 1)
        
        # Skip if the region is completely outside the volume
        if z_start >= z_end or y_start >= y_end or x_start >= x_end:
            continue
            
        # Calculate kernel region indices
        kernel_z_start = max(0, kz//2 - (z - z_start))
        kernel_z_end = kernel_z_start + (z_end - z_start)
        kernel_y_start = max(0, ky//2 - (y - y_start))
        kernel_y_end = kernel_y_start + (y_end - y_start)
        kernel_x_start = max(0, kx//2 - (x - x_start))
        kernel_x_end = kernel_x_start + (x_end - x_start)
        
        # Extract the valid kernel region
        kernel_region = marker_kernel[
            kernel_z_start:kernel_z_end,
            kernel_y_start:kernel_y_end,
            kernel_x_start:kernel_x_end
        ]
        
        # Skip if kernel region is empty
        if kernel_region.size == 0:
            continue
        
        # Project marker to each plane
        # XY projection (discard z dimension)
        if kernel_region.shape[0] > 0:
            xy_marker_proj = np.max(kernel_region, axis=0)
            xy_markers[y_start:y_end, x_start:x_end] = np.maximum(
                xy_markers[y_start:y_end, x_start:x_end], xy_marker_proj)
        
        # XZ projection (discard y dimension)  
        if kernel_region.shape[1] > 0:
            xz_marker_proj = np.max(kernel_region, axis=1)
            xz_markers[z_start:z_end, x_start:x_end] = np.maximum(
                xz_markers[z_start:z_end, x_start:x_end], xz_marker_proj)
        
        # YZ projection (discard x dimension)
        if kernel_region.shape[2] > 0:
            yz_marker_proj = np.max(kernel_region, axis=2).T
            yz_markers[y_start:y_end, z_start:z_end] = np.maximum(
                yz_markers[y_start:y_end, z_start:z_end], yz_marker_proj)
    
    # Scale markers to match projection intensity
    max_intensity = np.max(rgb_output) * 0.8  # Use 80% of max for visibility
    
    # Overlay markers on RGB projections
    # XY projection (center)
    mask_xy = xy_markers > 0.1
    if np.any(mask_xy):
        rgb_output[pad:pad + height, pad:pad + width, 0][mask_xy] += xy_markers[mask_xy] * max_intensity
        rgb_output[pad:pad + height, pad:pad + width, 1:][mask_xy] *= 0.5  # Dim other channels
    
    # XZ projection (bottom)
    mask_xz = xz_markers > 0.1
    if np.any(mask_xz):
        rgb_output[pad + height + pad:pad + height + pad + depth, pad:pad + width, 0][mask_xz] += xz_markers[mask_xz] * max_intensity
        rgb_output[pad + height + pad:pad + height + pad + depth, pad:pad + width, 1:][mask_xz] *= 0.5
    
    # YZ projection (right side)
    mask_yz = yz_markers > 0.1
    if np.any(mask_yz):
        rgb_output[pad:pad + height, pad + width + pad:pad + width + pad + depth, 0][mask_yz] += yz_markers[mask_yz] * max_intensity
        rgb_output[pad:pad + height, pad + width + pad:pad + width + pad + depth, 1:][mask_yz] *= 0.5
    
    # Clip and convert to uint8
    max_val = np.max(rgb_output)
    if max_val > 255:
        rgb_output = rgb_output * (255 / max_val)
    
    return np.clip(rgb_output, 0, 255).astype(np.uint8)

def save_segmentation_result_tif(volume, 
                                 peaks, 
                                 output_fn = None,
                                 marker_radius=[3,3,3], 
                                 marker_k=5, 
                                 vmax=100,
                                 absolute_limits=True, threshold=100, transpose = True):
    """
    Create a grayscale mask volume with soft-ball kernels at peak locations.
    Output is scaled to 0-255 (uint8).
    """
    from signal_extraction import soft_ball
    import numpy as np
    import tifffile

    mask = np.zeros_like(volume, dtype=np.float32)
    kernel = soft_ball(marker_radius, k=marker_k)
    kernel = kernel / kernel.max()  # Normalize to 1

    z_coords, y_coords, x_coords = peaks
    kz, ky, kx = kernel.shape
    dz, dy, dx = kz//2, ky//2, kx//2

    for z, y, x in zip(z_coords, y_coords, x_coords):
        z, y, x = int(z), int(y), int(x)
        zs, ze = max(0, z-dz), min(volume.shape[0], z+dz+1)
        ys, ye = max(0, y-dy), min(volume.shape[1], y+dy+1)
        xs, xe = max(0, x-dx), min(volume.shape[2], x+dx+1)

        ks, ke = max(0, dz-(z-zs)), dz+(ze-z)
        ls, le = max(0, dy-(y-ys)), dy+(ye-y)
        ms, me = max(0, dx-(x-xs)), dx+(xe-x)

        mask[zs:ze, ys:ye, xs:xe] = np.maximum(
            mask[zs:ze, ys:ye, xs:xe],
            kernel[ks:ke, ls:le, ms:me]
        )

    # Scale to 0-255 and convert to uint8
    mask = (mask / mask.max() * 255).astype(np.uint8) if mask.max() > 0 else mask.astype(np.uint8)
    v = get_clipped_array(volume, vmax=vmax, absolute_limits=absolute_limits)

    # Create RGB
    rgb = np.stack([v, v, v], axis=-1)

    # # Overlay: where mask > threshold, set to (R=mask, G=0, B=0)
    red_mask = mask > threshold
    rgb[red_mask, 0] = mask[red_mask]
    rgb[red_mask, 1] = 0
    rgb[red_mask, 2] = 0
    if transpose:
        rgb = rgb.transpose(0,2,1,3)
    if output_fn is not None:
        assert ".tif" in output_fn, "outputpath must be a tiff file"
        tifffile.imwrite(output_fn, rgb.get(), imagej=True)
    return rgb

def stimulus_correlation(traces, stim):
    #calculates the Pearsons R squared for every trace with every stimulus
    if stim.ndim == 1:
        stim = stim[:, None]
    stim_padded = np.zeros((traces.shape[0], stim.shape[1]))
    stim_padded[:stim.shape[0]] = stim

    # Fill NaNs in traces with column mean
    traces = np.where(np.isnan(traces), np.nanmean(traces, axis=0), traces)

    traces_std = traces.std(0)
    traces_std[traces_std == 0] = 1
    traces_z = (traces - traces.mean(0)) / traces_std

    stim_std = stim_padded.std(0)
    stim_std[stim_std == 0] = 1
    stim_z = (stim_padded - stim_padded.mean(0)) / stim_std

    corr = (np.dot(stim_z.T, traces_z) / traces.shape[0])
    sorted_idx = np.argsort(corr, axis=1)[:, ::-1]
    return corr, sorted_idx

def peristimulus_histogram(traces, stim, pad=10, align_to='onset'):
    """
    traces: (time, cells)
    stim: (time, n_stimuli), bool
    pad: int, number of samples before and after event
    align_to: 'onset' or 'offset'
    Returns: (n_stimuli, cells, 2*pad+max_len, 2) [mean, std]
    """
    assert stim.dtype == bool

    if stim.ndim == 1:
        stim = stim[:, None]
    stim_padded = np.zeros((traces.shape[0], stim.shape[1]))
    stim_padded[:stim.shape[0]] = stim
    stim = stim_padded

    n_time, n_cells = traces.shape
    n_stimuli = stim.shape[1]

    # Find all stimulus event indices and lengths
    stim_diff = np.diff(stim.astype(int), axis=0, prepend=0)
    onsets = [np.where(stim_diff[:, i] == 1)[0] for i in range(n_stimuli)]
    offsets = [np.where(stim_diff[:, i] == -1)[0] for i in range(n_stimuli)]

    # Calculate individual stimulus lengths
    stim_lengths = []
    for i in range(n_stimuli):
        # Pair onsets and offsets
        o, f = onsets[i], offsets[i]
        # If stimulus is on at the end, add end as offset
        if len(f) < len(o):
            f = np.append(f, n_time)
        stim_lengths.extend(f - o)
    max_len = np.max(stim_lengths)

    # Prepare output
    out = np.full((n_stimuli, n_cells, 2*pad+max_len, 2), np.nan)

    for stim_idx in tqdm(range(n_stimuli), desc="Calculating correlation with stimulus"):
        o, f = onsets[stim_idx], offsets[stim_idx]
        if len(f) < len(o):
            f = np.append(f, n_time)
        for cell in tqdm(range(n_cells), desc="for cells",leave=False):
            aligned = []
            for onset, offset in zip(o, f):
                if align_to == 'onset':
                    start = onset - pad
                    end = onset + max_len + pad
                else:  # offset
                    start = offset - max_len - pad
                    end = offset + pad
                # Bounds check
                if start < 0 or end > n_time:
                    continue
                aligned.append(seg)
            if aligned:
                arr = np.stack(aligned)
                out[stim_idx, cell, :, 0] = arr.mean(0)
                out[stim_idx, cell, :, 1] = arr.std(0)
    return out




















