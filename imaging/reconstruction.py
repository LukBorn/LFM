import numpy as np
import cupy as cp
import scipy.signal
from pyqtgraph.examples.logAxis import plotdata
from tqdm.auto import tqdm
from numpy.fft import fft2 as np_fft2, ifft2 as np_ifft2, fftshift as np_fftshift, ifftshift as np_ifftshift
from cupy.fft import fft2 as cp_fft2, ifft2 as cp_ifft2, fftshift as cp_fftshift, ifftshift as cp_ifftshift

import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve



def reconstruct_vol_from_img(img,
                             psf1,
                             psf2,
                             n_iter=30,
                             ratio=0.5,
                             xy_pad=201,
                             roi_size=300,
                             verbose=True,
                             gpu=False
                             ):
    if gpu:
        xp = cp
        fft2, ifft2, fftshift, ifftshift = cp_fft2, cp_ifft2, cp_fftshift, cp_ifftshift
    else:
        xp = np
        fft2, ifft2, fftshift, ifftshift = np_fft2, np_ifft2, np_fftshift, np_ifftshift

    size_x = psf1.shape[0] + 2 * xy_pad
    size_y = psf1.shape[1] + 2 * xy_pad
    size_z = psf1.shape[2]

    if verbose:
        print("Initializing memory")

    OTF = xp.zeros((size_x, size_y, size_z), dtype=xp.complex64)


    for i in range(size_z):
        OTF[:, :, i] = fft2(ifftshift(xp.pad(psf1[:, :, i], ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))

    size_add = round((size_x / ratio - size_x) / 2)
    size_sub = round(size_x * (1 - ratio) / 2) + size_add

    temp_1 = xp.zeros((size_x + 2 * size_add, size_y + 2 * size_add), dtype=xp.complex64)
    temp_2 = xp.zeros_like(temp_1, dtype=xp.float32)
    temp_3 = xp.zeros((size_x, size_y), dtype=xp.float32)

    img_padded = xp.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')

    temp_obj = xp.zeros((size_x, size_y), dtype=xp.float32)
    obj_recon = xp.ones((2 * roi_size, 2 * roi_size, size_z), dtype=xp.float32)
    img_est = xp.zeros((size_x, size_y), dtype=xp.float32)
    ratio_img = xp.zeros((size_x, size_y), dtype=xp.float32)

    for it in tqdm(range(n_iter)):
        img_est.fill(0)

        for z in range(size_z):
            temp_obj[size_x // 2 - roi_size: size_x // 2 + roi_size,
            size_y // 2 - roi_size: size_y // 2 + roi_size] = obj_recon[:, :, z]
            temp_1[size_add:-size_add, size_add:-size_add] = fftshift(fft2(temp_obj))
            temp_2 = xp.abs(ifft2(ifftshift(temp_1)))
            img_est += xp.maximum(xp.real(ifft2(OTF_A[:, :, z] * fft2(temp_obj))), 0)
            img_est += xp.maximum(xp.real(ifft2(OTF_B[:, :, z] * fft2(temp_2[size_add:-size_add, size_add:-size_add]))),
                                  0)

        temp_4 = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + xp.finfo(xp.float32).eps)
        ratio_img.fill(xp.mean(temp_4, dtype=xp.float32))
        ratio_img *= (img_est > (xp.max(img_est) / 200)).astype(xp.float32)
        ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + xp.finfo(xp.float32).eps)

        temp_2.fill(0)

        for z in range(size_z):
            temp_1[size_add:-size_add, size_add:-size_add] = fftshift(fft2(ratio_img) * xp.conj(OTF_B[:, :, z]))
            temp_2[size_sub:-size_sub, size_sub:-size_sub] = xp.abs(
                ifft2(ifftshift(temp_1[size_sub:-size_sub, size_sub:-size_sub])))
            temp_obj[size_x // 2 - roi_size: size_x // 2 + roi_size,
                    size_y // 2 - roi_size: size_y // 2 + roi_size] = obj_recon[:, :, z]
            temp_3 = temp_obj * (xp.maximum(xp.real(ifft2(fft2(ratio_img) * xp.conj(OTF_A[:, :, z]))), 0) + temp_2[
                                                                                                            size_add:-size_add,
                                                                                                            size_add:-size_add]) / 2
            obj_recon[:, :, z] = temp_3[size_x // 2 - roi_size: size_x // 2 + roi_size,
                                 size_y // 2 - roi_size: size_y // 2 + roi_size]



    return obj_recon

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
    assert psf.shape[1] == psf.shape[2]
    xy_pad = (psf.shape[1]-roi_size)//2

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
        pad = 10
        plot_mip = cp.zeros(shape = (n_iter,roi_size+size_z+3*pad, roi_size+size_z+3*pad))

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
            slice_temp[xy_pad:xy_pad+roi_size,xy_pad:xy_pad+roi_size] = obj_recon[z,:, :]
            forward_pass = fftconvolve(slice_temp, psf[z,:,:],  mode="same")
            img_estimate_temp += forward_pass

        img_ratio_temp = (img / img_estimate_temp + cp.finfo(img.dtype).eps)

        for z in back_projection:
            back_pass = fftconvolve(img_ratio_temp, psf_flipped[z,:,:], mode="same")
            obj_recon[z,:,:] *= back_pass[xy_pad:xy_pad+roi_size,xy_pad:xy_pad+roi_size]
            loss += cp.mean(back_pass[xy_pad:xy_pad+roi_size,xy_pad:xy_pad+roi_size])

        losses[it] = loss/size_z

        if plot:
            plot_mip[it,pad:pad+roi_size,pad:pad+roi_size] = obj_recon.max(axis=0)
            plot_mip[it,2*pad+roi_size:2*pad+roi_size+size_z, pad:pad+roi_size] = obj_recon.max(axis=1)
            plot_mip[it,pad:pad+roi_size,2*pad+roi_size:2*pad+roi_size+size_z] = obj_recon.max(axis=2).T
            # plt.imshow(plot_mip[it].get(), cmap='binary')


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
