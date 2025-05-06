import numpy as np
import cupy as cp
import scipy.signal
from pyqtgraph.examples.logAxis import plotdata
from tqdm.auto import tqdm
from numpy.fft import fft2 as np_fft2, ifft2 as np_ifft2, fftshift as np_fftshift, ifftshift as np_ifftshift
from cupy.fft import fft2 as cp_fft2, ifft2
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

from lfm.util import create_projection_image


def reconstruct_vol_from_img_gpu(img,
                                 psf,
                                 n_iter=30,
                                 ratio=0.5,
                                 xy_pad=201,
                                 roi_size=300,
                                 verbose=True,
                                 pad=10
                                 ):
    from cupy.fft import fft2, ifft2, fftshift, ifftshift

    size_y = psf.shape[1] + 2 * xy_pad
    size_x = psf.shape[2] + 2 * xy_pad
    size_z = psf.shape[0]

    if verbose:
        print("Initializing memory")

    OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

    for i in range(size_z):
        OTF[i, :, :] = fft2(ifftshift(cp.pad(psf[i, :, :], ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))

    img_padded = cp.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')

    temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
    obj_recon = cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
    img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
    ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
    plot_mip = np.zeros(shape=(n_iter, 2*roi_size + size_z + 3 * pad, 2*roi_size + size_z + 3 * pad), dtype=cp.float32)

    for it in tqdm(range(n_iter)):
        img_est.fill(0)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            img_est += cp.maximum(cp.real(ifft2(OTF[z, :, :] * fft2(temp_obj))), 0)

        # temp_4 = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
        #             img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
        # ratio_img[img_est > (cp.max(img_est) / 200)].fill(cp.mean(temp_4, dtype=cp.float32))
        # # ratio_img.fill(cp.mean(temp_4, dtype=cp.float32))
        # # ratio_img *= (img_est > (cp.max(img_est) / 200)).astype(cp.float32)
        ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            temp = temp_obj * (cp.maximum(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF[z, :, :]))), 0))
            obj_recon[z, :, :] = temp[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]
        plot_mip[it,:,:] = create_projection_image(obj_recon.get(),np.max,pad)

    return obj_recon, plot_mip

def reconstruct_volume_cpu(img,
                           psf,
                           obj_0=None,
                           n_iter=30,
                           roi_size=600,
                           verbose=True,
                           plot=False,
                           ):
    """
    img: LFM image: 2d array
    psf: point spread function of single fluorscent bead on different z slices,
         should be normalized to sum(psf) = 1
    obj_0: reconstructed object for first iteration. defaults to np.ones if None
    n_iter: number of iterations
    roi_size: xy shape of reconstructed volume

    """

    if verbose:
        print("Initializing memory")
    size_z = psf.shape[0]
    assert psf.shape[1] == psf.shape[2]
    xy_pad = (psf.shape[1]-roi_size)//2

    assert np.isclose(psf.sum(axis=(1,2)),1.0, atol=1e-5).all(), "PSF not normalized, normalize using / psf.sum"
    psf_flipped = psf[:,::-1,::-1]
    # psf_flipped = np.zeros_like(psf)
    # for z in range(size_z):
        # psf_flipped = np.fft.ifft(np.fft.fft2(psf[:, :, z]).conj().T)

    obj_recon = obj_0 if obj_0 is not None else np.ones((size_z,roi_size, roi_size), dtype=np.float32)

    img_estimate_temp = np.zeros_like(img)
    slice_temp = np.zeros_like(img)
    losses = np.zeros(n_iter)
    if plot:
        pad = 10
        plot_mip = np.zeros(shape = (roi_size+size_z+3*pad, roi_size+size_z+3*pad))


    if verbose:
        print("finished initializing memory")
        loop = tqdm(range(n_iter))
    else:
        loop = range(n_iter)
    for it in loop:
        img_estimate_temp.fill(0)
        loss = 0
        for z in range(size_z):
            slice_temp[xy_pad:xy_pad+roi_size,xy_pad:xy_pad+roi_size] = obj_recon[z,:, :]
            forward_pass = fftconvolve(slice_temp, psf[z,:,:],  mode="same")
            img_estimate_temp += forward_pass

        img_ratio_temp = (img / img_estimate_temp + np.finfo(img.dtype).eps)

        for z in range(size_z):
            back_pass = fftconvolve(img_ratio_temp, psf_flipped[z,:,:], mode="same")
            obj_recon[z,:,:] *= back_pass[xy_pad:xy_pad+roi_size,xy_pad:xy_pad+roi_size]
            loss += np.mean(back_pass[xy_pad:xy_pad+roi_size,xy_pad:xy_pad+roi_size])

        losses[it] = loss/size_z

        if plot:
            plot_mip[pad:pad+roi_size,pad:pad+roi_size] = obj_recon.max(axis=0)
            plot_mip[2*pad+roi_size:2*pad+roi_size+size_z, pad:pad+roi_size] = obj_recon.max(axis=1)
            plot_mip[pad:pad+roi_size,2*pad+roi_size:2*pad+roi_size+size_z] = obj_recon.max(axis=2).T
            plt.imshow(plot_mip, cmap='binary')


    return obj_recon, losses

def reconstruct_volume_gpu(img,
                           psf,
                           obj_0=None,
                           n_iter=30,
                           roi_size=600,
                           verbose=True,
                           plot=False,
                           pad = 10
                           ):
    """
    img: LFM image: 2d array
    psf: point spread function of single fluorscent bead on different z slices,
         should be normalized to sum(psf) = 1
    obj_0: reconstructed object for first iteration. defaults to np.ones if None
    n_iter: number of iterations
    roi_size: xy shape of reconstructed volume

    """
    from cupyx.scipy.signal import fftconvolve
    if verbose:
        print("Initializing memory")
    size_z = psf.shape[0]
    y_pad = (psf.shape[1]-roi_size)//2
    x_pad = (psf.shape[2]-roi_size)//2

    img = cp.array(img)
    psf = cp.array(psf)

    assert cp.isclose(psf.sum(axis=(1,2)),1.0, atol=1e-5).all(), "PSF not normalized, normalize using / psf.sum"
    psf_flipped = psf[:,::-1,::-1]
    # psf_flipped = np.zeros_like(psf)
    # for z in range(size_z):
        # psf_flipped = np.fft.ifft(np.fft.fft2(psf[:, :, z]).conj().T)

    obj_recon = obj_0 if obj_0 is not None else cp.ones((size_z,roi_size, roi_size), dtype=cp.float32)

    img_estimate_temp = cp.zeros_like(img)
    slice_temp = cp.zeros_like(img)
    losses = cp.zeros(n_iter)
    if plot:

        plot_mip = np.zeros(shape = (n_iter,roi_size + size_z + 3*pad,roi_size + size_z + 3*pad), dtype=cp.float32)
    else:
        plot_mip = None

    if verbose:
        print("Finished initializing memory")
        loop = tqdm(range(n_iter), "Main loop")
        forward_projection = tqdm(range(size_z),"Forward Projection",leave=False)
        back_projection = tqdm(range(size_z),"Back Projection",leave=False)
    else:
        loop = range(n_iter)
        forward_projection = range(size_z)
        back_projection = range(size_z)

    for it in loop:
        img_estimate_temp.fill(0)
        loss = 0
        for z in forward_projection:
            slice_temp[y_pad:y_pad+roi_size,x_pad:x_pad+roi_size] = obj_recon[z,:, :]
            forward_pass = fftconvolve(slice_temp, psf[z,:,:],  mode="same")
            img_estimate_temp += forward_pass

        img_ratio_temp = (img / img_estimate_temp + cp.finfo(img.dtype).eps)

        for z in back_projection:
            back_pass = fftconvolve(img_ratio_temp, psf_flipped[z,:,:], mode="same")
            obj_recon[z,:,:] *= back_pass[y_pad:y_pad+roi_size,x_pad:x_pad+roi_size]
            loss += cp.mean(back_pass[y_pad:y_pad+roi_size,x_pad:x_pad+roi_size])

        losses[it] = loss/size_z

        if plot:
            plot_mip[it,:,:] = create_projection_image(obj_recon.get(),np.max,pad)

    return obj_recon, losses, plot_mip




def generate_random_gaussians_3d(shape,
                                 sparseness=0.01,  # fraction of voxels that contain Gaussians
                                 intensity_dist=(50, 200),  # (min, max) for uniform distribution
                                 sigma_dist=(2, 5),  # (min, max) for standard deviations
                                 seed=None):
    """
    Generate a 3D volume with randomly placed 3D Gaussian kernels.

    Parameters:
    -----------
    shape: tuple
        Shape of the output volume (depth, height, width)
    sparseness: float
        Fraction of voxels that contain Gaussian kernels (0-1)
    intensity_dist: tuple
        (min, max) for uniform intensity distribution
    sigma_dist: tuple
        (min, max) for standard deviation of Gaussians
    seed: int
        Random seed for reproducibility

    Returns:
    --------
    volume: cp.ndarray
        3D volume with Gaussian kernels
    """
    if seed is not None:
        np.random.seed(seed)
        cp.random.seed(seed)

    # Create empty volume
    volume = cp.zeros(shape, dtype=cp.float32)
    depth, height, width = shape

    # Calculate number of Gaussians based on sparseness
    total_voxels = depth * height * width
    n_gaussians = int(total_voxels * sparseness)

    # Generate random positions
    positions = np.random.randint(n_gaussians, 3) * np.array([depth, height, width])

    # Generate random intensities and sigmas
    intensities = np.random.uniform(intensity_dist[0], intensity_dist[1], n_gaussians)
    sigmas = np.random.uniform(sigma_dist[0], sigma_dist[1], (n_gaussians, 3))

    # Create coordinate grids
    z_indices, y_indices, x_indices = cp.mgrid[:depth, :height, :width]

    # Add each Gaussian to the volume
    for i in range(n_gaussians):
        z_pos, y_pos, x_pos = positions[i]
        sigma_z, sigma_y, sigma_x = sigmas[i]
        intensity = intensities[i]

        # Calculate 3D Gaussian
        gaussian = intensity * cp.exp(
            -((z_indices - z_pos) ** 2 / (2 * sigma_z ** 2) +
              (y_indices - y_pos) ** 2 / (2 * sigma_y ** 2) +
              (x_indices - x_pos) ** 2 / (2 * sigma_x ** 2))
        )

        # Add to volume
        volume += gaussian

    return volume
