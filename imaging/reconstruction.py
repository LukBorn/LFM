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
        #URLs
        self.url_home = url_home
        self.out_url = self.pn_outrec.replace(expand('~'), url_home)         

def reconstruct_vols_from_imgs(paths,
                               img_iter=None,
                               params= dict(max_iter=30,
                                            xy_pad=201,
                                            roi_size=300,
                                            loss_threshold = 0,
                                            psf_downsample=1,
                                            OTF_subtract_bg=True,
                                            OTF_normalize=True,
                                            img_subtract_bg=False,
                                            img_mask=True,),
                               max_io_threads=5,
                               verbose=True,
                                ):
    max_iter, xy_pad, roi_size, loss_threshold, psf_downsample, OTF_subtract_bg, OTF_normalize, img_subtract_bg, img_mask = params.values()

    # Load and preprocess PSF

    otf_path = os.path.join(paths.pn_scratch + f"/OTF_{paths.psf_name}{"_-bg" if OTF_subtract_bg else ""}{"_norm" if OTF_normalize else ""}.npy")
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
  
        size_z = psf.shape[0]/psf_downsample
        size_y = psf.shape[1] + 2 * xy_pad
        size_x = psf.shape[2] + 2 * xy_pad
        #calculate OTF
        OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

        for z in tqdm(range(0,psf.shape[0],psf_downsample), desc=f"Calculating OTF: (downsampling PSF by{psf_downsample})"):
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
        if img_iter is None:
            data = np.array(f["data"])
            n_img = data.shape[0]
        else:
            assert len(img_iter) == 3, "n_img must be a tuple of (start, stop, step)"
            data = np.array(f["data"][n_img[0]:n_img[1]:n_img[2]])
            n_img = len(range(n_img[0], n_img[1], n_img[2]))

            
    
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
                # reusing obj_recon from previous frame bc they are probably very similar
                #forward pass
                for z in range(size_z):
                    #padding the obj_slice to center of temp_obj
                    temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                             size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
                    # adding the contribution of each slice to the image estimate
                    img_est += cp.maximum(cp.real(ifft2(OTF[z, :, :] * fft2(temp_obj))), 0)
                #calculate ratio
                ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                        img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                #backward pass
                for z in range(size_z):
                    #padding the obj_slice to center of temp_obj
                    temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                             size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
                    #updating the object estimate 
                    temp = temp_obj * (cp.maximum(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF[z, :, :]))), 0))
                    obj_recon[z, :, :] = temp[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                              size_x // 2 - roi_size: size_x // 2 + roi_size]
                #calculate loss
                loss = cp.mean(cp.abs(cp.log(ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad])))
                if loss < loss_threshold:
                    break
 
            # Save volume
            result_queue.put((it, obj_recon.get()))
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
                             bg=None,
                             circle_mask=None,
                             otf_path="",
                             params=dict(max_iter=30,
                                         xy_pad=201,
                                         roi_size=300,
                                         loss_threshold = 0,
                                         psf_downsample=1,
                                         OTF_subtract_bg=True,
                                         OTF_normalize=True,
                                         img_subtract_bg=False,
                                         img_mask=True,),
                             verbose=True,
                             plot=False,
                             pad=10,
                             ):

    max_iter, xy_pad, roi_size, loss_threshold, psf_downsample, OTF_subtract_bg, OTF_normalize, img_subtract_bg, img_mask = params.values()
    
    

    if psf==None and len(otf_path)==0:
        raise ValueError("No PSF or OTF provided")
    elif psf==None and len(otf_path)>0:
        print("Loading OTF from disk") if verbose else None
        OTF = cp.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Calculating OTF") if verbose else None
        size_y = psf.shape[1] + 2 * xy_pad
        size_x = psf.shape[2] + 2 * xy_pad
        size_z = psf.shape[0]/psf_downsample
        OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

        for i in range(0,size_z,psf_downsample):
            slice_processed = cp.asarray(psf[i,:,:]).astype(cp.float32)
            if OTF_subtract_bg:
                slice_processed -= bg
            if OTF_normalize:
                slice_processed /= slice_processed.sum()
            OTF[i, :, :] = fft2(ifftshift(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
        if len(otf_path)>0:
            cp.save(otf_path, OTF)


    print("Initializing memory") if verbose else None
    if img_subtract_bg:
        assert bg is not None, "bg must be provided if img_subtract_bg is True"
        img -= bg
    if img_mask:
        assert circle_mask is not None, "circle_mask must be provided if img_mask is True"
        img *= circle_mask
    img_padded = cp.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')

    obj_recon = obj_0 if obj_0 is not None else cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
    losses = cp.zeros(max_iter, dtype=cp.float32)if loss else None
    plot_mip = cp.zeros(shape=(max_iter, 2*roi_size + size_z + 3 * pad, 2*roi_size + size_z + 3 * pad), dtype=cp.float32) if plot else None

    temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
    img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
    ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)

    for it in tqdm(range(max_iter)):
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
        if calc_loss < loss_threshold:
            losses = losses[:it]
            plot_mip = plot_mip[:it]
            break

    return obj_recon, plot_mip, losses



