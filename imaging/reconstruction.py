import numpy as np
import cupy as cp
from cupy.fft import fft2, ifft2, ifftshift
from tqdm.auto import tqdm
from util import create_projection_image
import h5py
from multiprocessing import Pool, cpu_count
from functools import partial
import tempfile
from daio.h5 import lazyh5
import os, pathlib, socket, glob
import threading, queue

class Paths():
    def __init__(self, 
                dataset_name, 
                psf_name,
                pn_rec='',
                pn_psfs='', 
                pn_out='', 
                pn_scratch='', 
                url_home='', verbosity=0, expand=True, create_dirs=True):
        expand = lambda p: str(pathlib.Path(p).expanduser()) if expand else lambda p:str(p)
        
        
        # paths
        self.hostname = socket.gethostname()
        self.dataset_name = dataset_name
        self.pn_rec = expand(pathlib.Path(pn_rec, dataset_name))
        self.pn_out = expand(pathlib.Path(pn_out))
        self.pn_outrec = expand(pathlib.Path(self.pn_out, dataset_name))
        scratch = pn_out if not len(pn_scratch) else pn_scratch
        self.pn_scratch = expand(pathlib.Path(scratch, dataset_name))
        self.psf_name = psf_name
        self.pn_psfs = expand(pathlib.Path(pn_psfs))
        self.pn_psf = expand(pathlib.Path(self.pn_psfs, psf_name))
        
        
        # create directories
        if create_dirs:
            pathlib.Path(self.pn_outrec).mkdir(parents=True, exist_ok=True)

        # files
        self.psf_orig = os.path.join(self.pn_psf, 'psf.h5')
        self.psf = os.path.join(self.pn_psf, 'psf_filtered.h5')
        self.raw = os.path.join(self.pn_rec, 'data.h5')
        self.deconvolved = os.path.join(self.pn_outrec, 'deconvolved.h5')        
        self.otf = os.path.join(self.pn_outrec, f'OTF_{psf_name}.h5')
        #URLs
        self.url_home = url_home
        self.out_url = self.pn_outrec.replace(expand('~'), url_home)         

def reconstruct_vols_from_imgs(paths,
                               n_img=None,
                               xy_pad=201,
                               roi_size=200,
                               max_iter=30,
                               loss_threshold = 0,
                               OTF_subtract_bg=True,
                               OTF_normalize=True,
                               img_subtract_bg=False,
                               img_mask=True,
                               max_io_threads=5,
                               verbose=True,
                                ):

    # Load and preprocess PSF
    otf_path = os.path.join(paths.pn_scratch + f"/OTF_{paths.psf_name}.npy")
    if os.path.exists(otf_path):
        print("Loading OTF from disk") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            bg = cp.array(f["bg"]).astype(cp.float32)
            crop = cp.array(f["crop"])
            if img_mask:
                mask = cp.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
        OTF = cp.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Loading PSF, Calculating OTF") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            bg = cp.array(f["bg"]).astype(cp.float32)
            crop = cp.array(f["crop"])
            if img_mask:
                mask = cp.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
            psf = cp.array(f["psf"])
  
        size_z = psf.shape[0]
        size_y = psf.shape[1] + 2 * xy_pad
        size_x = psf.shape[2] + 2 * xy_pad
        #calculate OTF
        OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

        for z in tqdm(range(size_z)):
            slice_processed = cp.asarray(psf[z,:,:]).astype(cp.float32)
            if OTF_subtract_bg:
                slice_processed -= bg
            if OTF_normalize:
                slice_processed /= slice_processed.sum()
            OTF[z, :, :] = fft2(ifftshift(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
            # assert slice_processed.sum() = 1, "OTF not normalized"
        cp.save(otf_path, OTF)
        
        del psf


    print("Loading Images") if verbose else None
    with h5py.File(paths.raw, 'r') as f:
        if n_img is None:
            data = np.array(f["data"])
            n_img = data.shape[0]
        else:
            data = np.array(f["data"][:n_img])
            
    
    print("Creating output dataset") if verbose else None  
    with h5py.File(paths.deconvolved, 'w') as f:
        # Create dataset for the reconstructed volume     
        dset = f.create_dataset("data", shape=(n_img, size_z, 2*roi_size, 2*roi_size), dtype=np.float32)        
        losses = f.create_dataset("losses", shape=(n_img,), dtype=np.float32)
        n_iters = f.create_dataset("n_iters", shape=(n_img,), dtype=np.int32)
        grp = f.create_group("deconvolution_params")
        grp.attrs["roi_size"] = roi_size
        grp.attrs["xy_pad"] = xy_pad
        grp.attrs["max_iter"] = max_iter
        grp.attrs["loss_threshold"] = loss_threshold
        grp.attrs["OTF_subtract_bg"] = OTF_subtract_bg
        grp.attrs["OTF_normalize"] = OTF_normalize
        grp.attrs["img_subtract_bg"] = img_subtract_bg
        grp.attrs["img_mask"] = img_mask

        #create multithreaded IO
        result_queue = queue.Queue(maxsize=max_io_threads)
        def io_worker(dset):
            while True:
                item = result_queue.get()
                if item is None:
                    break
                it, obj_recon = item
                dset[it, :, :, :] = obj_recon
                dset.flush()
                result_queue.task_done()

        io_thread = threading.Thread(target=io_worker, args=(dset,))
        io_thread.start()


        print("initializing memory") if verbose else None
        # Preallocate memory for the reconstructed volume
        obj_recon = cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
        temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
        img_padded = cp.zeros((size_y, size_x), dtype=cp.float32)        
        img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
        ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)

        for it in tqdm(range(n_img), desc="Reconstructing volumes:"):
            img = cp.array(data[it, :,:]).astype(cp.float32)[crop[0]:crop[1], crop[2]:crop[3]]
            if img_subtract_bg:
                img -= bg
            if img_mask:
                img = img * mask
            img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            for n_iter in tqdm(range(max_iter), leave=False):
                img_est.fill(0)
                #forward pass
                for z in range(size_z):
                    # reusing obj_recon from previous iteration bc they are probably very similar
                    temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                             size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
                    img_est += cp.maximum(cp.real(ifft2(OTF[z, :, :] * fft2(temp_obj))), 0)
                #calculate ratio
                ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                        img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                #backward pass
                for z in range(size_z):
                    temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                             size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
                    temp = temp_obj * (cp.maximum(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF[z, :, :]))), 0))
                    obj_recon[z, :, :] = temp[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                              size_x // 2 - roi_size: size_x // 2 + roi_size]
                #calculate loss
                loss = cp.mean(cp.abs(cp.log(ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad])))
                if loss < loss_threshold:
                    break
 
            # Save volume
            result_queue.put((it, obj_recon.get() if gpu else obj_recon))
            # Save loss and n_iter
            losses[it] = loss.get()
            n_iters[it] = n_iter

    # Stop the IO thread
    result_queue.put(None)
    io_thread.join()
    print("Deconvolution finished") if verbose else None


def reconstruct_vol_from_img(img,
                             psf=None,
                             obj_0=None,
                             otf_path="",
                             params = dict(n_iter=30,
                                           xy_pad=201,
                                           roi_size=300,
                                           psf_downsample=1,
                                           threshold = 0,),
                             verbose=True,
                             plot=False,
                             pad=10,
                             ):

    n_iter, xy_pad, roi_size, psf_downsample, threshold = params.values() 
    
    size_y = psf.shape[1] + 2 * xy_pad
    size_x = psf.shape[2] + 2 * xy_pad
    size_z = psf.shape[0]

    if psf==None and len(otf_path)==0:
        raise ValueError("No PSF or OTF provided")
    elif psf==None and len(otf_path)>0:
        print("Loading OTF from disk") if verbose else None
        OTF = cp.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Calculating OTF") if verbose else None
        OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

        for i in range(0,size_z,psf_downsample):
            OTF[i, :, :] = fft2(ifftshift(cp.pad(psf[i, :, :], ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
        if len(otf_path)>0:
            cp.save(otf_path, OTF)


    print("Initializing memory") if verbose else None
    img_padded = cp.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')

    obj_recon = obj_0 if obj_0 is not None else cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
    losses = cp.zeros(n_iter, dtype=cp.float32)if loss else None
    plot_mip = cp.zeros(shape=(n_iter, 2*roi_size + size_z + 3 * pad, 2*roi_size + size_z + 3 * pad), dtype=cp.float32) if plot else None

    temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
    img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
    ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)

    for it in tqdm(range(n_iter)):
        img_est.fill(0)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            img_est += cp.maximum(cp.real(ifft2(OTF[z, :, :] * fft2(temp_obj))), 0)
        est_ims[it, :, :] = img_est
        ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            temp = temp_obj * (cp.maximum(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF[z, :, :]))), 0))
            obj_recon[z, :, :] = temp[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]

        if plot:
            plot_mip[it,:,:] = create_projection_image(obj_recon,cp.max,pad)

        ratio_ = ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
        calc_loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
        losses[it] = calc_loss
        if calc_loss < threshold:
            losses = losses[:it]
            plot_mip = plot_mip[:it]
            break

    return obj_recon, plot_mip, losses



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
