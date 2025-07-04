import numpy as np
import cupy as cp
from cupy.fft import fft2, ifft2, ifftshift
from tqdm.auto import tqdm
import h5py, json
from multiprocessing import Pool, cpu_count
from functools import partial
import tempfile
from daio.h5 import lazyh5
import os, pathlib, socket, glob, traceback
import threading, queue, time
from video import AVWriter, create_projection_image
from slurm import SlowProgressLogger
from i_o import AsyncH5Writer, volume_reader


def get_OTF(paths,
            OTF_normalize=True,
            OTF_clip=False,
            psf_downsample=None,
            xy_pad=201,
            verbose=True
            ):
    otf_path = os.path.join(paths.pn_scratch + f"/OTF_{paths.psf_name}_pad{xy_pad}{"_clip" if OTF_clip else ""}{"_norm" if OTF_normalize else ""}{f'_dwn{psf_downsample}' if psf_downsample is not None else ''}.npy")
    if psf_downsample is None:
        psf_downsample = [0, None, 1]
    if os.path.exists(otf_path):
        print("Loading OTF from disk") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            crop = list(f["crop"])
            mask = np.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]    
            zpos = np.array(f["z_positions"])
            zpos = zpos[psf_downsample[0]:psf_downsample[1]:psf_downsample[2]] if psf_downsample is not None else zpos
        OTF = np.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Loading PSF, Calculating OTF") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            crop = list(f["crop"])
            mask = np.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
            zpos = np.array(f["z_positions"])
            zpos = zpos[psf_downsample[0]:psf_downsample[1]:psf_downsample[2]] if psf_downsample is not None else zpos
            psf = cp.array(f["psf"])
  
        size_y = psf.shape[1] + 2 * xy_pad
        size_x = psf.shape[2] + 2 * xy_pad
        if psf_downsample[1] is None:
            psf_downsample[1] = psf.shape[0]  
        else:
            psf_downsample[1] = psf[:psf_downsample[1],:,:].shape[0]
        size_z = len(range(*psf_downsample))

        #calculate OTF
        OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

        for i,z in enumerate(tqdm(range(*psf_downsample), desc=f"Calculating OTF: (downsampling PSF by {psf_downsample[2]})")):
            slice_processed = cp.asarray(psf[z,:,:]).astype(cp.float32)
            if OTF_clip:
                slice_processed = cp.clip(slice_processed, 0, None)
            if OTF_normalize:
                slice_processed /= slice_processed.sum()
            OTF[i, :, :] = fft2(ifftshift(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
            # OTF[z, :, :] = fft2(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant'))
            # assert slice_processed.sum() = 1, "OTF not normalized"
        cp.save(otf_path, OTF)
        OTF = OTF.get()
        del psf 

    return OTF, size_z, size_y, size_x, crop, mask, zpos, otf_path



class PrevVolumeManager:
    def __init__(self, size_z, roi_size, reuse_prev_vol):
        self.lock = threading.Lock()
        self.volume = cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
        self.index= -1
        self.reuse_prev_vol = reuse_prev_vol
    
    def update(self, volume, idx):
        if self.reuse_prev_vol:
            if idx > self.index:
                with self.lock:
                    self.volume = volume
                    self.index = idx

    def get(self):   
        with self.lock:
            return self.volume


def reconstruct_vols_from_imgs_parallel2(paths,
                                        img_idx=None,
                                        max_iter=30,
                                        xy_pad=201,
                                        roi_size=300,
                                        loss_threshold = 0,
                                        reuse_prev_vol=False,
                                        psf_downsample=None,
                                        OTF_normalize=False,
                                        OTF_clip=False,
                                        crop=None,
                                        img_subtract_bg=False,
                                        img_mask=True,
                                        fully_batched=False,
                                        write_mip_video=True,
                                        vmin=0,
                                        vmax=100,
                                        absolute_limits=False,
                                        force_new=False,
                                        verbose=True,
                                        ):
    """
    Reconstructs volumes from images using the Richardson-Lucy algorithm.
    based on the MATLAB code from Cong et al. (2017) "Rapid whole brain imaging of neural activity in freely behaving larval zebrafish (Danio rerio)"
    https://doi.org/10.7554/eLife.28158
    original matlab code in "LFM/imaging/XLFM reconstruction"
    
    Args:
    paths (Paths): Paths object containing the paths to the input and output files.

    img_idx (tuple): Tuple of (start, stop, step) for the images to be processed. If None, all images will be processed.

    max_iter (int): Maximum number of iterations for the Richardson-Lucy algorithm.
    xy_pad (int): Padding size in the x and y dimensions when generating otf.
    roi_size (int): Size of the region of interest (ROI) in the x and y dimensions.
    
    loss_threshold (float): Threshold for the loss function to stop the iteration 
                            mean(abs(log(every non-zero pixel of the ratio of original image to reconstructed image)))
    reuse_prev_vol (bool): Whether to reuse the final previous volume estimate for the next image.
    
    psf_downsample (tuple): Tuple of (start, stop, step) for downsampling the PSF. If None, no downsampling is applied.
    OTF_normalize (bool): Whether to normalize the OTF.
    OTF_clip (bool): Whether to clip the OTF to non-negative values.
    
    crop (tuple): Tuple of (start_y, stop_y, start_x, stop_x) for cropping the images. If None, the crop from PSF is used, which might cause errors
    
    img_subtract_bg (bool): Whether to subtract the background from the images.
    img_mask (bool): Whether to mask the cropped images with the circle mask from the PSF.
    
    fully_batched (bool): Whether to use fully batched deconvolution, where there are no loops over zslices when convolving object and OTF
                          Uses more memory, but is faster(?), reverts to non-batched deconvolution if out of memory error occurs. 
    
    write_mip_video (bool): Whether to write a video of the maximum intensity projections of the reconstructed volumes.
        vmin (int): vmin for the video projection.
        vmax (int): vmax for the video projection.
        absolute_limits (bool): absolute_limits for the video projection.

    force_new (bool): Whether to overwrite existing files.

    verbose (bool): Whether to print progress messages.
    

    Reading and writing are seperated into seperate threads, so that reading, writing and deconvolution can happen in parallel
    The function uses multiple GPUs in parallel if available, with each GPU process using its own thread, each GPU processes a different image.
    
    Be sure to only use one instance of the function if running it on cluster (ntasks=1,nodes=1), otherwise there will be I/O errors
    """  
    start_time = time.time()

    # Load or preprocess PSF
    OTF, size_z, size_y, size_x, _crop, mask, zpos, otf_path = get_OTF(paths,
                                                                       OTF_normalize=OTF_normalize,
                                                                       OTF_clip=OTF_clip,
                                                                       psf_downsample=psf_downsample,
                                                                       xy_pad=xy_pad,
                                                                       verbose=verbose,
                                                                        )
    
    crop = _crop if crop is None else crop

    fps = json.load(open(paths.meta))["acquisition"]["fps"]
    led_pwr = json.load(open(paths.meta))["acquisition"]["led_percent"]

    if paths.bg != "" and img_subtract_bg:
        bg_data = lazyh5(paths.bg)
        if (fps, led_pwr) != (bg_data["fps"], bg_data["led_percent"]):
            raise Warning(f"Background data fps ({bg_data['fps']}) and led power ({bg_data['led_percent']}) do not match acquisition data fps ({fps}) and led power ({led_pwr}). This may lead to incorrect background subtraction.")
        bg = np.array(bg_data["data"]).astype(np.uint8)
    elif paths.bg == "" and img_subtract_bg:
        raise ValueError("no background file specified")


    print("Setting up I/O queues") if verbose else None
    #determine the read indexes
    with h5py.File(paths.raw, 'r') as f:
        if img_idx is None:
            img_idx = (0, f["data"].shape[0], 1)
            save_fn = paths.deconvolved
        else:
            assert len(img_idx) == 3, "img_idx must be a tuple of (start, stop, step)"
            read_idx = range(img_idx[0], img_idx[1], img_idx[2])
            save_fn = paths.deconvolved[:-3]+ f"_frames{img_idx[0]}-{img_idx[1]}.h5"


    # handle existing files
    if os.path.exists(save_fn) and not force_new:
        try:
            with h5py.File(save_fn, 'r') as f:
                params = f['deconvolution_params']
                param_match = (
                    params.attrs.get('OTF', -1) == otf_path and
                    params.attrs.get('roi_size', -1) == roi_size and
                    params.attrs.get('max_iter', -1) == max_iter and
                    params.attrs.get('loss_threshold', -1) == loss_threshold and
                    params.attrs.get('reuse_prev_vol', -1) == reuse_prev_vol and
                    params.attrs.get('img_subtract_bg', -1) == img_subtract_bg and
                    params.attrs.get('img_mask', -1) == img_mask
                    )
            if param_match:
                processed_indices = f['processed_indices'][:]
                print(f"Resuming from existing file. {len(processed_indices)}/{n_img} volumes already processed.") if verbose else None
                read_idx = [i for i in read_idx if i not in processed_indices]

            else:
                print("Parameters do not match, overwriting existing file.") if verbose else None
                os.remove(save_fn)
        except (OSError, KeyError):
            print("File exists but is not compatible, overwriting existing file.") if verbose else None
            os.remove(save_fn)
    elif os.path.exists(save_fn) and force_new:
        print(f"Overwriting existing file: {save_fn}") if verbose else None
        os.remove(save_fn)
    else:
        print(f"Writing into new file: {save_fn}") if verbose else None

    n_gpus = get_available_gpus()
    stop_event = threading.Event()

    # set up reading queue
    reader = volume_reader(paths.raw, 'data', i_frames=read_idx, prefetch=2*n_gpus+1)

    writer = AsyncH5Writer(save_fn)
    writer.write_meta('deconvolution_params', {'OTF': otf_path,
                                                'roi_size': roi_size,
                                                'max_iter': max_iter,
                                                'xy_pad': xy_pad,
                                                'loss_threshold': loss_threshold,
                                                'reuse_prev_vol': reuse_prev_vol,
                                                'img_subtract_bg': img_subtract_bg,
                                                'img_mask': img_mask,
                                                'psf_downsample': psf_downsample,
                                                'OTF_normalize': OTF_normalize,
                                                'OTF_clip': OTF_clip,
                                                'crop': crop,
                                                'vmin': vmin,
                                                'vmax': vmax,
                                                'absolute_limits': absolute_limits
                                                })
    writer.create_dataset('data', shape=(reader.len, size_z, 2 * roi_size, 2 * roi_size), dtype=np.float32)
    writer.create_dataset('losses', shape=(reader.len, max_iter), dtype=np.float32)
    writer.create_dataset('n_iter', shape=(reader.len,), dtype=np.int32)
    processed_indeces = []
    
    if write_mip_video:
        fn_vid = paths.deconvolved[:-3] + f"_f{}_mip_vmin{vmin}_vmax{vmax}{"_al" if absolute_limits else ""}.mp4"
        video_writer = AVWriter2(fn_vid,fps=fps)

    #setup previous volume manager
    prev_volume_manager = PrevVolumeManager(size_z, roi_size, reuse_prev_vol)

    # Define the GPU worker class
    class GPUWorker:
        def __init__(self, gpu_id, fully_batched=False):
            self.gpu_id = gpu_id
            cp.cuda.Device(gpu_id).use()
            
            self.OTF = cp.asarray(OTF)
            self.mask = cp.asarray(mask) if img_mask else None
            self.bg = cp.asarray(bg) if img_subtract_bg else None
            self.fully_batched = fully_batched
            self.init_memory()

        def init_memory(self):
            if not self.fully_batched:
                print(f"GPU {self.gpu_id}: Initializing Memory") if verbose else None
                self.temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)                
                self.temp = cp.zeros((size_y, size_x), dtype=cp.float32)
            else:
                print(f"GPU {self.gpu_id}: Initializing Memory for fully batched deconvolution") if verbose else None
                self.temp_obj = cp.zeros((size_z, size_y, size_x), dtype=cp.float32)
            self.img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.img_padded = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.losses = cp.zeros(max_iter, dtype=cp.float32)

        def deconvolve(self, it, img):
            if img_subtract_bg: img -= self.bg
            if img_mask: img *= self.mask
            self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            self.obj_recon = cp.asarray(prev_volume_manager.get()).copy()

            loop = tqdm(range(max_iter), desc=f"GPU {self.gpu_id}: Deconvolving image {it+1}/{n_img}", leave=False) if verbose else range(max_iter)

            for iter in loop:
                self.img_est.fill(0)
                #forward pass
                for z in range(size_z):
                    #padding the obj_slice to center of temp_obj
                    self.temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                  size_x // 2 - roi_size: size_x // 2 + roi_size] = self.obj_recon[z, :, :]
                    # adding the contribution of each slice to the image estimate
                    self.img_est += cp.maximum(cp.real(ifft2(self.OTF[z, :, :] * fft2(self.temp_obj))), 0)
                #calculate ratio
                self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                        self.img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                #backward pass
                for z in range(size_z):
                    #padding the obj_slice to center of temp_obj
                    self.temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                  size_x // 2 - roi_size: size_x // 2 + roi_size] = self.obj_recon[z, :, :]
                    #updating the object estimate 
                    self.temp = self.temp_obj * (cp.maximum(cp.real(ifft2(fft2(self.ratio_img) * cp.conj(self.OTF[z, :, :]))), 0))
                    self.obj_recon[z, :, :] = self.temp[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                                        size_x // 2 - roi_size: size_x // 2 + roi_size]
                #calculate loss
                ratio_ = self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
                loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
                self.losses[iter] = loss
                if loss < loss_threshold:
                    break      
                
            return self.obj_recon.get(), self.losses.get(), iter
        
        def deconvolve_batch(self, it, img):
            if img_subtract_bg: img -= bg
            if img_mask: img *= self.mask
            self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            self.obj_recon = cp.asarray(prev_volume_manager.get()).copy()

            loop = tqdm(range(max_iter), desc=f"GPU {self.gpu_id}: Deconvolving image {it+1}/{reader.len}", leave=False) if verbose else range(max_iter)

            for iter in loop:
                self.img_est.fill(0)
                self.temp_obj.fill(0)

                self.temp_obj[:,size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = self.obj_recon[:, :, :]

                self.img_est = cp.clip(cp.real(ifft2(self.OTF * fft2(self.temp_obj, axes=(1,2)), axes=(1,2))), 0, None).sum(axis=0)
                
                self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    self.img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                
                self.temp_obj *= cp.clip(cp.real(ifft2(fft2(self.ratio_img) * cp.conj(self.OTF), axes=(1,2))), 0, None)
                
                self.obj_recon[:, :, :] = self.temp_obj[:, size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]
                ratio_ = self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
                loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
                if loss < loss_threshold:
                    break      
                
            return self.obj_recon.get(), self.losses.get(), iter

    # Define the GPU worker loop
    failed_gpus = 0
    def gpu_worker_loop(gpu_id):
        worker = GPUWorker(gpu_id, fully_batched=fully_batched)
        try:
            while not stop_event.is_set():
                item = reader.get_next_prefetched(timeout=60)
                if item is None:
                    break
                frame_n, img = item
                if not worker.fully_batched:
                    obj_recon, losses_arr, n_iter = worker.deconvolve(frame_n, cp.array(img))
                else:
                    try:
                        obj_recon, losses_arr, n_iter = worker.deconvolve_batch(idx, cp.array(img))
                    except cp.cuda.memory.OutOfMemoryError:
                        print(f"GPU{gpu_id} ran out of memory, switching to nonbatched deconvolution")
                        worker.fully_batched = False
                        worker.init_memory()
                        obj_recon, losses_arr, n_iter = worker.deconvolve(idx, cp.array(img))
                writer.write('data', obj_recon, frame_n)
                writer.write('losses', losses_arr, frame_n)
                writer.write('n_iter', n_iter, frame_n)
                if write_mip_video:
                    mip = create_projection_image(obj_recon, 
                                                text=str(frame_n), 
                                                vmin=vmin,vmax=vmax,
                                                scalebar=200, zpos=zpos,
                                                text_size=1,
                                                absolute_limits=absolute_limits)
                    video_writer.write(mip)
                prev_volume_manager.update(obj_recon, idx)
        except Exception as e:
            print(f"Error in GPU worker {gpu_id}: \n{traceback.format_exc()}")
            nonlocal failed_gpus
            failed_gpus += 1
            if failed_gpus == n_gpus:
                print("all GPU workers failed, stopping all workers.")
            else:
                print(f"Putting frame {idx} back to read queue")
                reader.put_back((frame_n, img))
                
    # Start GPU worker threads
    gpu_threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(target=gpu_worker_loop, args=(gpu_id,))
        t.start()
        gpu_threads.append(t)

    # Stop the IO threads
    for t in gpu_threads:
        t.join()
    stop_event.set()
    reader.close()
    writer.close()
    if write_mip_video:
        video_writer.close()
    print(f"Deconvolution finished in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}") if verbose else None

    kwargs = dict(max_iter=max_iter,
                  xy_pad=xy_pad,
                  roi_size=roi_size,
                  loss_threshold=loss_threshold,
                  reuse_prev_vol=reuse_prev_vol,
                  psf_downsample=psf_downsample,
                  OTF_normalize=OTF_normalize,
                  OTF_clip=OTF_clip,
                  crop=crop,
                  img_subtract_bg=img_subtract_bg,
                  img_mask=img_mask,
                  fully_batched=fully_batched,
                  vmin=vmin,
                  vmax=vmax,
                  absolute_limits=absolute_limits,)
    
    return kwargs, save_fn, fn_vid if write_mip_video else None


def reconstruct_vols_from_imgs_parallel(paths,
                                        img_idx=None,
                                        max_iter=30,
                                        xy_pad=201,
                                        roi_size=300,
                                        loss_threshold = 0,
                                        reuse_prev_vol=False,
                                        psf_downsample=None,
                                        OTF_normalize=False,
                                        OTF_clip=False,
                                        crop=None,
                                        img_subtract_bg=False,
                                        img_mask=True,
                                        fully_batched=False,
                                        write_mip_video=True,
                                        vmin=0,
                                        vmax=100,
                                        absolute_limits=False,
                                        force_new=False,
                                        verbose=True,
                                        ):
    """
    Reconstructs volumes from images using the Richardson-Lucy algorithm.
    based on the MATLAB code from Cong et al. (2017) "Rapid whole brain imaging of neural activity in freely behaving larval zebrafish (Danio rerio)"
    https://doi.org/10.7554/eLife.28158
    original matlab code in "LFM/imaging/XLFM reconstruction"
    
    Args:
    paths (Paths): Paths object containing the paths to the input and output files.

    img_idx (tuple): Tuple of (start, stop, step) for the images to be processed. If None, all images will be processed.

    max_iter (int): Maximum number of iterations for the Richardson-Lucy algorithm.
    xy_pad (int): Padding size in the x and y dimensions when generating otf.
    roi_size (int): Size of the region of interest (ROI) in the x and y dimensions.
    
    loss_threshold (float): Threshold for the loss function to stop the iteration 
                            mean(abs(log(every non-zero pixel of the ratio of original image to reconstructed image)))
    reuse_prev_vol (bool): Whether to reuse the final previous volume estimate for the next image.
    
    psf_downsample (tuple): Tuple of (start, stop, step) for downsampling the PSF. If None, no downsampling is applied.
    OTF_normalize (bool): Whether to normalize the OTF.
    OTF_clip (bool): Whether to clip the OTF to non-negative values.
    
    crop (tuple): Tuple of (start_y, stop_y, start_x, stop_x) for cropping the images. If None, the crop from PSF is used, which might cause errors
    
    img_subtract_bg (bool): Whether to subtract the background from the images.
    img_mask (bool): Whether to mask the cropped images with the circle mask from the PSF.
    
    fully_batched (bool): Whether to use fully batched deconvolution, where there are no loops over zslices when convolving object and OTF
                          Uses more memory, but is faster(?), reverts to non-batched deconvolution if out of memory error occurs. 
    
    write_mip_video (bool): Whether to write a video of the maximum intensity projections of the reconstructed volumes.
        vmin (int): vmin for the video projection.
        vmax (int): vmax for the video projection.
        absolute_limits (bool): absolute_limits for the video projection.

    force_new (bool): Whether to overwrite existing files.

    verbose (bool): Whether to print progress messages.
    

    Reading and writing are seperated into seperate threads, so that reading, writing and deconvolution can happen in parallel
    The function uses multiple GPUs in parallel if available, with each GPU process using its own thread, each GPU processes a different image.
    
    Be sure to only use one instance of the function if running it on cluster (ntasks=1,nodes=1), otherwise there will be I/O errors
    """  
    start_time = time.time()

    # Load or preprocess PSF
    OTF, size_z, size_y, size_x, _crop, mask, zpos, otf_path = get_OTF(paths,
                                                                       OTF_normalize=OTF_normalize,
                                                                       OTF_clip=OTF_clip,
                                                                       psf_downsample=psf_downsample,
                                                                       xy_pad=xy_pad,
                                                                       verbose=verbose,
                                                                        )
    
    crop = _crop if crop is None else crop

    fps = json.load(open(paths.meta))["acquisition"]["fps"]
    led_pwr = json.load(open(paths.meta))["acquisition"]["led_percent"]

    if paths.bg != "" and img_subtract_bg:
        bg_data = lazyh5(paths.bg)
        if (fps, led_pwr) != (bg_data["fps"], bg_data["led_percent"]):
            raise Warning(f"Background data fps ({bg_data['fps']}) and led power ({bg_data['led_percent']}) do not match acquisition data fps ({fps}) and led power ({led_pwr}). This may lead to incorrect background subtraction.")
        bg = np.array(bg_data["data"]).astype(np.uint8)
    elif paths.bg == "" and img_subtract_bg:
        raise ValueError("no background file specified")


    print("Setting up I/O queues") if verbose else None
    #determine the read indexes
    with h5py.File(paths.raw, 'r') as f:
        if img_idx is None:
            n_img = f["data"].shape[0]
            read_idx = range(n_img)
            save_fn = paths.deconvolved
        else:
            assert len(img_idx) == 3, "n_img must be a tuple of (start, stop, step)"
            read_idx = range(img_idx[0], img_idx[1], img_idx[2])
            n_img = len(read_idx)
            save_fn = paths.deconvolved[:-3]+ f"_frames{img_idx[0]}-{img_idx[1]}.h5"


    # handle existing files
    if os.path.exists(save_fn) and not force_new:
        try:
            with h5py.File(save_fn, 'r') as f:
                params = f['deconvolution_params']
                param_match = (
                    params.attrs.get('OTF', -1) == otf_path and
                    params.attrs.get('roi_size', -1) == roi_size and
                    params.attrs.get('max_iter', -1) == max_iter and
                    params.attrs.get('loss_threshold', -1) == loss_threshold and
                    params.attrs.get('reuse_prev_vol', -1) == reuse_prev_vol and
                    params.attrs.get('img_subtract_bg', -1) == img_subtract_bg and
                    params.attrs.get('img_mask', -1) == img_mask
                    )
            if param_match:
                processed_indices = f['processed_indices'][:]
                print(f"Resuming from existing file. {len(processed_indices)}/{n_img} volumes already processed.") if verbose else None
                read_idx = [i for i in read_idx if i not in processed_indices]
                n_img = len(read_idx)
            else:
                print("Parameters do not match, overwriting existing file.") if verbose else None
                os.remove(save_fn)
        except (OSError, KeyError):
            print("File exists but is not compatible, overwriting existing file.") if verbose else None
            os.remove(save_fn)
    elif os.path.exists(save_fn) and force_new:
        print(f"Overwriting existing file: {save_fn}") if verbose else None
        os.remove(save_fn)
    else:
        print(f"Writing into new file: {save_fn}") if verbose else None

    n_gpus = get_available_gpus()
    stop_event = threading.Event()

    # set up reading queue
    read_queue = queue.Queue(maxsize=n_gpus*2+1)
    def reader_worker(stop_event):
        try:
            with h5py.File(paths.raw, 'r') as f:
                read_loop = tqdm(read_idx, desc="Reader") if verbose else read_idx
                for it, frame_n in enumerate(read_loop):
                    if stop_event.is_set():
                        break
                    read_queue.put((it,f["data"][frame_n, crop[0]:crop[1], crop[2]:crop[3]],frame_n))
                    #print("Read frame ", it+1, "/", n_img) if verbose else None
            for i in range(n_gpus):
                read_queue.put(None)
        except Exception as e:
            print(f"Error in reader worker:\n{traceback.format_exc()}")
            stop_event.set()
            for i in range(n_gpus):
                read_queue.put(None)
    reader_thread = threading.Thread(target=reader_worker, args=(stop_event,))
    reader_thread.start()

    # set up writing queue
    write_queue = queue.Queue(maxsize=n_gpus*2+1)
    def writer_worker(stop_event):
        progress= SlowProgressLogger(n_img, description="Writer", update_interval=60) if verbose else None
        with h5py.File(save_fn, 'w') as f:
            # Create dataset for the reconstructed volume     
            dset = f.create_dataset("data", shape=(n_img, size_z, 2*roi_size, 2*roi_size), dtype=np.float32)        
            losses = f.create_dataset("losses", shape=(n_img,max_iter), dtype=np.float32)
            n_iters = f.create_dataset("n_iters", shape=(n_img,), dtype=np.int32)
            zpos = f.create_dataset("zpos", data=zpos, dtype=np.float32)
            grp = f.create_group("deconvolution_params")
            grp.attrs["roi_size"] = roi_size
            grp.attrs["xy_pad"] = xy_pad
            grp.attrs["max_iter"] = max_iter
            grp.attrs["loss_threshold"] = loss_threshold
            grp.attrs["OTF"] = otf_path
            grp.attrs["reuse_prev_vol"] = reuse_prev_vol
            grp.attrs["img_subtract_bg"] = img_subtract_bg
            grp.attrs["img_mask"] = img_mask
            processed_indices = []

            try:
                while not stop_event.is_set():
                    item = write_queue.get()
                    if item is None:
                        break
                    it, obj_recon, loss, n_iter = item
                    dset[it, :, :, :] = obj_recon
                    losses[it,:] = loss
                    n_iters[it] = n_iter
                    processed_indices.append(it)
                    dset.flush()
                    write_queue.task_done()
                    progress.update()
                    #print(f"Written frame {it+1}/{n_img} to disk") if verbose else None
                f.create_dataset("processed_indices", data=processed_indices)
                f.flush()
            except Exception as e:
                print(f"Error in writer worker:\n{traceback.format_exc()}")
                stop_event.set()
                f.create_dataset("processed_indices", data=processed_indices)
                f.flush()
    writer_thread = threading.Thread(target=writer_worker, args=(stop_event,))
    writer_thread.start()

    video_writer_queue = queue.Queue(maxsize=n_gpus*2+1)
    # set up video writer
    def video_writer_worker(stop_event):
        next_idx = 0
        buffer = {}
        video_writer = None
        try:
            while not stop_event.is_set():
                item = video_writer_queue.get()
                if item is None:
                    break
                idx, mip = item
                if video_writer is None:
                    try:
                        print(f"Initializing video writer:")
                        print(f"  Output file: {fn_vid}")
                        print(f"  Frame shape: {mip.shape}")
                        print(f"  FPS: {fps}")
                        print(f"  Directory exists: {os.path.exists(os.path.dirname(fn_vid))}")
                        print(f"  Directory writable: {os.access(os.path.dirname(fn_vid), os.W_OK)}")
                        
                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(fn_vid), exist_ok=True)
                        
                        # Try different codec settings
                        video_writer = AVWriter(fn_vid, 
                                                height=mip.shape[0],
                                                width=mip.shape[1],
                                                fps=fps,
                                                codec='mpeg4',
                                                pix_fmt='yuv420p',)  
                        print("Video writer initialized successfully")
                    except Exception as e:
                        print(f"Video writer initialization failed: {e}")
                        print(f"Error type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                        # Try fallback options
                        try:
                            print("Trying fallback video settings...")
                            video_writer = AVWriter(fn_vid, 
                                                    height=mip.shape[0],
                                                    width=mip.shape[1],
                                                    fps=10,  # Very low fps
                                                    codec='mpeg4',  # Different codec
                                                    pix_fmt='yuv420p')
                            print("Fallback video writer initialized")
                        except Exception as e2:
                            print(f"Fallback also failed: {e2}")
                            # Consume remaining items without processing
                            while not stop_event.is_set():
                                try:
                                    item = video_writer_queue.get()
                                    if item is None:
                                        break
                                    video_writer_queue.task_done()
                                except:
                                    break
                            return
                
                buffer[idx] = mip
                while next_idx in buffer:
                    try:
                        print(f"Writing frame {next_idx}, shape: {buffer[next_idx].shape}, dtype: {buffer[next_idx].dtype}")
                        video_writer.write(buffer[next_idx].astype(np.uint8))
                        del buffer[next_idx]
                        next_idx += 1
                    except Exception as e:
                        print(f"Frame write error: {e}")
                        print(f"Error type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                        print("Disabling video output")
                        buffer.clear()
                        while not stop_event.is_set():
                            try:
                                item = video_writer_queue.get()
                                if item is None:
                                    break
                                video_writer_queue.task_done()
                            except:
                                break
                        return
                video_writer_queue.task_done()
        except Exception as e:
            print(f"Video writer worker error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception as e:
                    print(f"Error closing video writer: {e}")

        video_writer_thread = threading.Thread(target=video_writer_worker, args=(stop_event,))
        video_writer_thread.daemon = True  # Make it daemon so it won't block shutdown
        video_writer_thread.start()
    
    #setup previous volume manager
    prev_volume_manager = PrevVolumeManager(size_z, roi_size, reuse_prev_vol)

    # Define the GPU worker class
    class GPUWorker:
        def __init__(self, gpu_id, fully_batched=False):
            self.gpu_id = gpu_id
            cp.cuda.Device(gpu_id).use()
            
            self.OTF = cp.asarray(OTF)
            self.mask = cp.asarray(mask) if img_mask else None
            self.bg = cp.asarray(bg) if img_subtract_bg else None
            self.fully_batched = fully_batched
            self.init_memory()

        def init_memory(self):
            if not self.fully_batched:
                print(f"GPU {self.gpu_id}: Initializing Memory") if verbose else None
                self.temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)                
                self.temp = cp.zeros((size_y, size_x), dtype=cp.float32)
            else:
                print(f"GPU {self.gpu_id}: Initializing Memory for fully batched deconvolution") if verbose else None
                self.temp_obj = cp.zeros((size_z, size_y, size_x), dtype=cp.float32)
            self.img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.img_padded = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.losses = cp.zeros(max_iter, dtype=cp.float32)

        def deconvolve(self, it, img):
            if img_subtract_bg: img -= self.bg
            if img_mask: img *= self.mask
            self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            self.obj_recon = cp.asarray(prev_volume_manager.get()).copy()

            loop = tqdm(range(max_iter), desc=f"GPU {self.gpu_id}: Deconvolving image {it+1}/{n_img}", leave=False) if verbose else range(max_iter)

            for iter in loop:
                self.img_est.fill(0)
                #forward pass
                for z in range(size_z):
                    #padding the obj_slice to center of temp_obj
                    self.temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                  size_x // 2 - roi_size: size_x // 2 + roi_size] = self.obj_recon[z, :, :]
                    # adding the contribution of each slice to the image estimate
                    self.img_est += cp.maximum(cp.real(ifft2(self.OTF[z, :, :] * fft2(self.temp_obj))), 0)
                #calculate ratio
                self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                        self.img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                #backward pass
                for z in range(size_z):
                    #padding the obj_slice to center of temp_obj
                    self.temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                  size_x // 2 - roi_size: size_x // 2 + roi_size] = self.obj_recon[z, :, :]
                    #updating the object estimate 
                    self.temp = self.temp_obj * (cp.maximum(cp.real(ifft2(fft2(self.ratio_img) * cp.conj(self.OTF[z, :, :]))), 0))
                    self.obj_recon[z, :, :] = self.temp[size_y // 2 - roi_size: size_y // 2 + roi_size,
                                                        size_x // 2 - roi_size: size_x // 2 + roi_size]
                #calculate loss
                ratio_ = self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
                loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
                self.losses[iter] = loss
                if loss < loss_threshold:
                    break      
                
            return self.obj_recon.get(), self.losses.get(), iter
        
        def deconvolve_batch(self, it, img):
            if img_subtract_bg: img -= bg
            if img_mask: img *= self.mask
            self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            self.obj_recon = cp.asarray(prev_volume_manager.get()).copy()

            loop = tqdm(range(max_iter), desc=f"GPU {self.gpu_id}: Deconvolving image {it+1}/{n_img}", leave=False) if verbose else range(max_iter)

            for iter in loop:
                self.img_est.fill(0)
                self.temp_obj.fill(0)

                self.temp_obj[:,size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = self.obj_recon[:, :, :]

                self.img_est = cp.clip(cp.real(ifft2(self.OTF * fft2(self.temp_obj, axes=(1,2)), axes=(1,2))), 0, None).sum(axis=0)
                
                self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    self.img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                
                self.temp_obj *= cp.clip(cp.real(ifft2(fft2(self.ratio_img) * cp.conj(self.OTF), axes=(1,2))), 0, None)
                
                self.obj_recon[:, :, :] = self.temp_obj[:, size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]
                ratio_ = self.ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
                loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
                if loss < loss_threshold:
                    break      
                
            return self.obj_recon.get(), self.losses.get(), iter

    # Define the GPU worker loop
    failed_gpus = 0
    def gpu_worker_loop(gpu_id):
        worker = GPUWorker(gpu_id, fully_batched=fully_batched)
        try:
            while not stop_event.is_set():
                item = read_queue.get()
                if item is None:
                    break
                idx, img, frame_n = item
                if not worker.fully_batched:
                    obj_recon, losses_arr, n_iter = worker.deconvolve(idx, cp.array(img))
                else:
                    try:
                        obj_recon, losses_arr, n_iter = worker.deconvolve_batch(idx, cp.array(img))
                    except cp.cuda.memory.OutOfMemoryError:
                        print(f"GPU{gpu_id} ran out of memory, switching to nonbatched deconvolution")
                        worker.fully_batched = False
                        worker.init_memory()
                        obj_recon, losses_arr, n_iter = worker.deconvolve(idx, cp.array(img))
                write_queue.put((idx, obj_recon, losses_arr, n_iter))
                if write_mip_video:
                    mip = create_projection_image(obj_recon, 
                                                text=str(frame_n), 
                                                vmin=vmin,vmax=vmax,
                                                scalebar=200, zpos=zpos,
                                                text_size=1,
                                                absolute_limits=absolute_limits)
                    video_writer_queue.put((idx, mip))
                prev_volume_manager.update(obj_recon, idx)
                read_queue.task_done()
        except Exception as e:
            print(f"Error in GPU worker {gpu_id}: \n{traceback.format_exc()}")
            nonlocal failed_gpus
            failed_gpus += 1
            if failed_gpus == n_gpus:
                print("all GPU workers failed, stopping all workers.")
                stop_event.set()
                # Clear the read queue
                try: [read_queue.get_nowait() for _ in range(read_queue.qsize())] 
                except queue.Empty: pass
            else:
                print(f"Putting frame {idx} back to read queue")
                read_queue.put((idx, img, frame_n))
                
    # Start GPU worker threads
    gpu_threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(target=gpu_worker_loop, args=(gpu_id,))
        t.start()
        gpu_threads.append(t)


    # Stop the IO threads
    for t in gpu_threads:
        t.join()
    stop_event.set()
    write_queue.put(None)
    reader_thread.join(timeout=60)
    writer_thread.join(timeout=300)
    if write_mip_video:
        video_writer_queue.put(None)
        video_writer_thread.join(timeout=60)
    print(f"Deconvolution finished in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}") if verbose else None

    kwargs = dict(max_iter=max_iter,
                  xy_pad=xy_pad,
                  roi_size=roi_size,
                  loss_threshold=loss_threshold,
                  reuse_prev_vol=reuse_prev_vol,
                  psf_downsample=psf_downsample,
                  OTF_normalize=OTF_normalize,
                  OTF_clip=OTF_clip,
                  crop=crop,
                  img_subtract_bg=img_subtract_bg,
                  img_mask=img_mask,
                  fully_batched=fully_batched,
                  vmin=vmin,
                  vmax=vmax,
                  absolute_limits=absolute_limits,)
    
    return kwargs, save_fn, fn_vid if write_mip_video else None

def reconstruct_vols_from_imgs(paths,
                               img_idx=None,
                               max_iter=30,
                               xy_pad=201,
                               roi_size=300,
                               loss_threshold = 0,
                               reuse_prev_vol=False,
                               psf_downsample=None,
                               OTF_normalize=False,
                               OTF_clip=False,
                               crop=None,
                               img_subtract_bg=False,
                               img_mask=True,
                               fully_batched=False,
                               plot_decon=True,
                                   vmin=0,
                                   vmax=100,
                                   absolute_limits=False,
                               verbose=True):
    """
    for reconstructing a small amount of images, no I/O, everything kept in memory
    for large amounts of images, use reconstruct_vols_from_imgs_parallel
    for more precise documentation see reconstruct_vols_from_imgs_parallel

    plot_decon: wether to save an mip at every iteration of the deconvolution
    """
    assert len(img_idx) == 3, "n_img must be a tuple of (start, stop, step)"
    read_idx = range(img_idx[0], img_idx[1], img_idx[2])
    n_img = len(read_idx)

    OTF, size_z, size_y, size_x, _crop, mask, zpos, otf_path = get_OTF(paths,
                                                                       OTF_normalize=OTF_normalize,
                                                                       OTF_clip=OTF_clip,
                                                                       psf_downsample=psf_downsample,
                                                                       xy_pad=xy_pad,
                                                                       verbose=verbose,
                                                                        )
    
    crop = _crop if crop is None else crop

    fps = json.load(open(paths.meta))["acquisition"]["fps"]
    led_pwr = json.load(open(paths.meta))["acquisition"]["led_percent"]

    if paths.bg != "" and img_subtract_bg:
        bg_data = lazyh5(paths.bg)
        if (fps, led_pwr) != (bg_data["fps"], bg_data["led_percent"]):
            raise Warning(f"Background data fps ({bg_data['fps']}) and led power ({bg_data['led_percent']}) do not match acquisition data fps ({fps}) and led power ({led_pwr}). This may lead to incorrect background subtraction.")
        bg = bg_data["data"]
    
    objs = np.zeros(shape=(n_img, size_z, 2*roi_size, 2*roi_size), dtype=np.float32)
    mip_shape = 2*roi_size+(3*int(zpos.shape[0]/10))+zpos.shape[0]
    plots_mip = np.zeros(shape=(n_img, max_iter, mip_shape, mip_shape), dtype=np.float32)
    losses = np.zeros((n_img, max_iter), dtype=np.float32)
    
    OTF = cp.asarray(OTF)
    mask = cp.asarray(mask) if img_mask else None
    bg = cp.asarray(bg).astype(cp.uint8) if img_subtract_bg else None
    img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
    ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
    img_padded = cp.zeros((size_y, size_x), dtype=cp.float32)
    obj_recon= cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
    if not fully_batched:
        temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
        temp = cp.zeros((size_y, size_x), dtype=cp.float32)
    else:
        temp_obj = cp.zeros((size_z, size_y, size_x), dtype=cp.float32)

    for it, frame_n in enumerate(tqdm(read_idx, desc="Reconstructing volumes")):
        with h5py.File(paths.raw, 'r') as f:
            img = cp.asarray(f["data"][frame_n, crop[0]:crop[1], crop[2]:crop[3]])
        if img_subtract_bg: img -= bg
        if img_mask: img *= mask
        img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img
        if not reuse_prev_vol:
                obj_recon.fill(1)
        loop = tqdm(range(max_iter), desc=f"Deconvolving image {it+1}/{n_img}", leave=False) if verbose else range(max_iter)
        for iter in loop:
            img_est.fill(0)
            if not fully_batched:
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
            else:
                # fully batched deconvolution
                temp_obj[:,size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[:, :, :]
                img_est = cp.clip(cp.real(ifft2(OTF * fft2(temp_obj, axes=(1,2)), axes=(1,2))), 0, None).sum(axis=0)
                ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
                temp_obj *= cp.clip(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF), axes=(1,2))), 0, None)
                obj_recon[:, :, :] = temp_obj[:, size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]
            
            if plot_decon:
                plots_mip[it, iter,:,:] = create_projection_image(obj_recon, 
                                                                text=str(frame_n), 
                                                                vmin=vmin, vmax=vmax,absolute_limits=absolute_limits,
                                                                scalebar=200, zpos=zpos,text_size=3,)

           #calculate loss
            ratio_ = ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
            loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
            losses[it,iter] = loss
            if loss < loss_threshold:
                break
        
        # Store the reconstructed object 
        objs[it, :, :, :] = obj_recon.get()

    kwargs = dict(max_iter=max_iter,
                  xy_pad=xy_pad,
                  roi_size=roi_size,
                  loss_threshold=loss_threshold,
                  reuse_prev_vol=reuse_prev_vol,
                  psf_downsample=psf_downsample,
                  OTF_normalize=OTF_normalize,
                  OTF_clip=OTF_clip,
                  crop=crop,
                  img_subtract_bg=img_subtract_bg,
                  img_mask=img_mask,
                  fully_batched=fully_batched,
                  vmin=vmin,
                  vmax=vmax,
                  absolute_limits=absolute_limits,)

    return objs, plots_mip, losses, kwargs

def get_available_gpus():
    """Robustly detect available GPUs with comprehensive error handling"""
    try:
         # Reset CUDA context before detection
        try:
            cp.cuda.runtime.deviceReset()
            # Add a small delay to let GPU reset
            import time
            time.sleep(1)
        except:
            pass
        
        # Force synchronization
        try:
            cp.cuda.runtime.deviceSynchronize()
        except:
            pass
        
        # Set environment variables for better GPU handling
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        # First check if CUDA is available at all
        print("Checking CUDA availability...")
        
        # Method 1: Check nvidia-smi
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '-L'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                nvidia_gpu_count = len(gpu_lines)
                print(f"nvidia-smi reports {nvidia_gpu_count} GPUs")
            else:
                nvidia_gpu_count = 0
                print(f"nvidia-smi failed with return code {result.returncode}")
        except Exception as e:
            nvidia_gpu_count = 0
            print(f"nvidia-smi check failed: {e}")
        
        # Method 2: Try to initialize CUDA context
        try:
            print("Attempting CUDA context initialization...")
            
            # Try to reset any existing context
            try:
                cp.cuda.runtime.deviceReset()
            except:
                pass
            
            # Set CUDA device order
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # Try to get device count
            cupy_gpu_count = cp.cuda.runtime.getDeviceCount()
            print(f"CuPy reports {cupy_gpu_count} GPUs")
            
            # Test each GPU
            working_gpus = []
            for i in range(cupy_gpu_count):
                try:
                    with cp.cuda.Device(i):
                        # Try a simple operation
                        test_array = cp.ones(10, dtype=cp.float32)
                        result = test_array.sum()
                        print(f"GPU {i}: Working (test sum: {result})")
                        working_gpus.append(i)
                except Exception as e:
                    print(f"GPU {i}: Failed - {e}")
            
            cupy_gpu_count = len(working_gpus)
            
        except Exception as e:
            print(f"CuPy GPU detection failed: {e}")
            cupy_gpu_count = 0
        
        # Use the minimum of both methods
        n_gpus = min(nvidia_gpu_count, cupy_gpu_count) if nvidia_gpu_count > 0 and cupy_gpu_count > 0 else max(nvidia_gpu_count, cupy_gpu_count)
        
        if n_gpus == 0:
            print("No working GPUs detected. Available options:")
            print("1. Run on CPU (not recommended)")
            print("2. Check SLURM GPU allocation")
            print("3. Contact system administrator")
            raise RuntimeError("No GPUs detected or accessible")
        
        print(f"Using {n_gpus} GPU(s)")
        return n_gpus
        
    except Exception as e:
        print(f"GPU detection completely failed: {e}")
        return 1  # Fallback