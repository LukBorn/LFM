import numpy as np
import cupy as cp
from tqdm.auto import tqdm
import h5py
from daio.h5 import lazyh5
from i_o import AsyncH5Writer, VolumeReader
import cupyx

# detect a way of where the traces should be extracted from -> covariance map while registering
# define a kernel to extract 
# extract teh sum(?) fluorescence from the kernel across time

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
    maxima_bool &= vol > 0
    peaks = cp.nonzero(cp.array(maxima_bool))
    peak_vals = vol[peaks]
    if in_class == 'numpy.ndarray':
        maxima_bool = maxima_bool.get()
        peaks = tuple(i.get() for i in peaks)
        peak_vals = peak_vals.get()
    return maxima_bool, peaks, peak_vals


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
