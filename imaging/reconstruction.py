import numpy as np
import cupy as cp
from cupy.fft import fft2, ifft2, ifftshift
from tqdm.auto import tqdm
import h5py
from multiprocessing import Pool, cpu_count
from functools import partial
import tempfile
from daio.h5 import lazyh5
import os, pathlib, socket, glob
import threading, queue
from video import AVWriter, create_projection_image


def reconstruct_vols_from_imgs_parallel(paths,
                                        img_idx=None,
                                        max_iter=30,
                                        xy_pad=201,
                                        roi_size=300,
                                        loss_threshold = 0,
                                        psf_downsample=None,
                                        OTF_normalize=False,
                                        OTF_clip=False,
                                        img_subtract_bg=False,
                                        img_mask=True,
                                        img_clip=True,
                                        reuse_prev_vol=False,
                                        write_mip_video=True,
                                        vmin=0,
                                        vmax=100,
                                        fps=10,
                                        absolute_limits=False,
                                        force_new=False,
                                        verbose=True,
                                        ):
    
    # Load and preprocess PSF
    otf_path = os.path.join(paths.pn_scratch + f"/OTF_{paths.psf_name}{"_clip" if OTF_clip else None}{"_norm" if OTF_normalize else ""}{f'_dwn{psf_downsample}' if psf_downsample is not None}.npy")
    if os.path.exists(otf_path):
        print("Loading OTF from disk") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            crop = list(f["crop"])
            if img_mask:
                mask = np.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
        OTF = np.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Loading PSF, Calculating OTF") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            crop = list(f["crop"])
            if img_mask:
                mask = np.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
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

        for i,z in enumerate(tqdm(range(*psf_downsample), desc=f"Calculating OTF: (downsampling PSF by{psf_downsample[2]})")):
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


    print("Setting up I/O queues") if verbose else None
    bg = cp.load(paths.bg) if img_subtract_bg else None
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

    if os.path.exists(save_fn) and not force_new:
        with h5py.File(save_fn, 'r') as f:
            params = f['deconvolution_params']
            param_match = (
                params.attrs.get('roi_size', -1) == roi_size and
                params.attrs.get('xy_pad', -1) == xy_pad and
                params.attrs.get('max_iter', -1) == max_iter and
                params.attrs.get('loss_threshold', -1) == loss_threshold and
                params.attrs.get('OTF', -1) == otf_path and
                params.attrs.get('reuse_prev_vol', -1) == reuse_prev_vol and
                params.attrs.get('img_subtract_bg', -1) == img_subtract_bg and
                params.attrs.get('img_mask', -1) == img_mask
                )
            if param_match:
                processed_indices = f['processed_indices'][:]
            if processed_indices:
                print(f"Resuming from existing file. {len(processed_indices)}/{n_img} volumes already processed.") if verbose else None
                read_idx = [i for i in read_idx if i not in processed_indices]
                n_img = len(read_idx)
            else:
                print("Parameters do not match, overwriting existing file.") if verbose else None
    else:
        print(f"Writing into new file: {save_fn}") if verbose else None

    # set up reading queue
    n_gpus = cp.cuda.runtime.getDeviceCount()
    stop_event = threading.Event()

    read_queue = queue.Queue(maxsize=n_gpus*2+1)
    def reader_worker(stop_event):
        with h5py.File(paths.raw, 'r') as f:
            read_loop = tqdm(read_idx, desc="Reader") if verbose else read_idx
            for it, frame_n in enumerate(read_loop):
                if stop_event.is_set():
                    break
                read_queue.put((it,f["data"][frame_n, crop[0]:crop[1], crop[2]:crop[3]],frame_n))
        for i in range(n_gpus):
            read_queue.put(None)
    reader_thread = threading.Thread(target=reader_worker, args=(stop_event,))
    reader_thread.start()

    # set up writing queue
    write_queue = queue.Queue(maxsize=n_gpus*2+1)
    def writer_worker(stop_event):
        with h5py.File(save_fn, 'w') as f:
            # Create dataset for the reconstructed volume     
            dset = f.create_dataset("data", shape=(n_img, size_z, 2*roi_size, 2*roi_size), dtype=np.float32)        
            losses = f.create_dataset("losses", shape=(n_img,max_iter), dtype=np.float32)
            n_iters = f.create_dataset("n_iters", shape=(n_img,), dtype=np.int32)
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
            f.create_dataset("processed_indices", data=processed_indices)
    writer_thread = threading.Thread(target=writer_worker, args=(stop_event,))
    writer_thread.start()

    if write_mip_video:
        
        video_writer_queue = queue.Queue(maxsize=n_gpus*2+1) 
        fn_vid = save_fn.replace('.h5', f"_mip_vmin{vmin}_vmax{vmax}{'_al' if absolute_limits else ''}.mp4")
        def video_writer_worker(stop_event):
            next_idx = 0
            buffer = {}
            video_writer=None
            while not stop_event.is_set():
                idx, mip = video_writer_queue.get()
                if mip is None:
                    break
                if video_writer is None:
                    video_writer = AVWriter(fn_vid, 
                                            height=mip.shape[0],
                                            width=mip.shape[1],
                                            fps=fps,
                                            pix_fmt='yuv420p',
                                            out_fmt='gray',)
                buffer[idx] = mip
                while next_idx in buffer:
                    video_writer.write(mip.astype(np.uint8))
                    del buffer[next_idx]
                    next_idx += 1
                video_writer_queue.task_done()
        video_writer_thread = threading.Thread(target=video_writer_worker, args=(stop_event,))
        video_writer_thread.start()
        

    class PrevVolumeManager:
        def __init__(self):
            self.lock = threading.Lock()
            self.volume = cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
            self.index= -1
        
        def update(self, volume, idx):
            if reuse_prev_vol:
                if idx > self.index:
                    with self.lock:
                        self.volume = volume
                        self.index = idx

        def get(self):   
            with self.lock:
                return self.volume
        
    prev_volume_manager = PrevVolumeManager()
    

    class GPUWorker:
        def __init__(self, gpu_id):
            self.gpu_id = gpu_id
            cp.cuda.Device(gpu_id).use()
            print(f"GPU {gpu_id}: Initializing Memory") if verbose else None
            self.OTF = cp.asarray(OTF)
            self.mask = cp.asarray(mask) if img_mask else None

            self.temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.img_padded = cp.zeros((size_y, size_x), dtype=cp.float32)        
            self.img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.temp = cp.zeros((size_y, size_x), dtype=cp.float32)
            self.losses = cp.zeros(max_iter, dtype=cp.float32)

        def deconvolve(self, it, img):
            if img_subtract_bg: img -= bg
            if img_mask: img *= self.mask
            self.img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            self.obj_recon = cp.asarray(prev_volume_manager.get()).copy()

            loop = tqdm(range(max_iter), desc=f"GPU {self.gpu_id}: Deconvolving image {it+1}/{n_img}", leave=False) if verbose else range(max_iter)

            for iter in loop:
                self.img_est.fill(0)
                # reusing obj_recon from previous frame bc they are probably very similar
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
                

    def gpu_worker_loop(gpu_id):
        worker = GPUWorker(gpu_id)
        while True:
            item = read_queue.get()
            if item is None:
                break
            idx, img, frame_n = item
            obj_recon, losses_arr, n_iter = worker.deconvolve(idx, cp.array(img))
            mip = create_projection_image(obj_recon, 
                                          text=str(frame_n), 
                                          vmin=vmin,vmax=vmax,
                                          absolute_limits=absolute_limits) if write_mip_video else None
            write_queue.put((idx, obj_recon, losses_arr, n_iter))
            if write_mip_video:
                video_writer_queue.put((idx, mip))
            prev_volume_manager.update(obj_recon, idx)
            read_queue.task_done()

    gpu_threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(target=gpu_worker_loop, args=(gpu_id,))
        t.start()
        gpu_threads.append(t)

    # Wait for all GPU workers to finish
    for t in gpu_threads:
        t.join()
    # Stop the IO thread
    stop_event.set()
    write_queue.put(None)
    reader_thread.join()
    writer_thread.join()
    if write_mip_video:
        video_writer_queue.put(None)
        video_writer_thread.join()
    print("Deconvolution finished") if verbose else None

    kwargs = dict(max_iter=max_iter,
                  xy_pad=xy_pad,
                  roi_size=roi_size,
                  loss_threshold=loss_threshold,
                  psf_downsample=psf_downsample,
                  OTF_normalize=OTF_normalize,
                  OTF_clip=OTF_clip,
                  img_subtract_bg=img_subtract_bg,
                  img_mask=img_mask,
                  img_clip=img_clip,
                  reuse_prev_vol=reuse_prev_vol,)
    
    return kwargs, save_fn, fn_vid if write_mip_video else None



def reconstruct_vols_from_imgs(paths,
                               img_idx=None,
                               max_iter=30,
                               xy_pad=201,
                               roi_size=300,
                               loss_threshold = 0,
                               psf_downsample=1,
                               OTF_normalize=False,
                               OTF_clip=False,
                               img_subtract_bg=False,
                               img_mask=True,
                               img_clip=True,
                               reuse_prev_vol=True,
                               max_io_threads=2,
                               verbose=True,
                               ):
    """
    Reconstructs volumes from images using the Richardson-Lucy algorithm.
    based on the MATLAB code from Cong et al. (2017) "Rapid whole brain imaging of neural activity in freely behaving larval zebrafish (Danio rerio)"
    https://doi.org/10.7554/eLife.28158
    original matlab code in "LFM/imaging/XLFM reconstruction"
    
    Args:
    paths (Paths): Paths object containing the paths to the input and output files.
    img_iter (tuple): Tuple of (start, stop, step) for the images to be processed. If None, all images will be processed.

    max_io_threads (int): Number of threads to use for IO operations. Default is 5.
    """    

    # Load and preprocess PSF
    otf_path = os.path.join(paths.pn_scratch + f"/OTF_{paths.psf_name}{"_clip" if OTF_clip else None}{"_norm" if OTF_normalize else ""}.npy")
    if os.path.exists(otf_path):
        print("Loading OTF from disk") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            crop = cp.array(f["crop"])
            if img_mask:
                mask = cp.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
        OTF = cp.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Loading PSF, Calculating OTF") if verbose else None
        with h5py.File(paths.psf, 'r') as f:
            crop = cp.array(f["crop"])
            if img_mask:
                mask = cp.array(f["circle_mask"])[crop[0]:crop[1], crop[2]:crop[3]]
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

        for i,z in enumerate(tqdm(range(*psf_downsample), desc=f"Calculating OTF: (downsampling PSF by{psf_downsample[2]})")):
            slice_processed = cp.asarray(psf[z,:,:]).astype(cp.float32)
            if OTF_clip:
                slice_processed = cp.clip(slice_processed, 0, None)
            if OTF_normalize:
                slice_processed /= slice_processed.sum()
            OTF[i, :, :] = fft2(ifftshift(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
            # OTF[z, :, :] = fft2(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant'))
            # assert slice_processed.sum() = 1, "OTF not normalized"
        cp.save(otf_path, OTF)
        del psf


    print("Loading Images") if verbose else None
    bg = cp.load(paths.bg) if img_subtract_bg else None
    with h5py.File(paths.raw, 'r') as f:
        if img_idx is None:
            data = np.array(f["data"])
            n_img = data.shape[0]
            save_fn = paths.deconvolved
        else:
            assert len(img_idx) == 3, "n_img must be a tuple of (start, stop, step)"
            data = np.array(f["data"][img_idx[0]:img_idx[1]:img_idx[2]])
            n_img = len(range(img_idx[0], img_idx[1], img_idx[2]))
            save_fn = paths.deconvolved[:-3]+ f"_frames{img_idx[0]}-{img_idx[1]}.h5"
            
    
    print("Creating output dataset") if verbose else None  
    with h5py.File(save_fn, 'w') as f:
        # Create dataset for the reconstructed volume     
        dset = f.create_dataset("data", shape=(n_img, size_z, 2*roi_size, 2*roi_size), dtype=np.float32)        
        losses = f.create_dataset("losses", shape=(n_img, max_iter), dtype=np.float32)
        n_iters = f.create_dataset("n_iters", shape=(n_img,), dtype=np.int32)
        grp = f.create_group("deconvolution_params")
        grp.attrs["roi_size"] = roi_size
        grp.attrs["xy_pad"] = xy_pad
        grp.attrs["max_iter"] = max_iter
        grp.attrs["loss_threshold"] = loss_threshold
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


        print("Initializing Memory") if verbose else None
        # Preallocate memory for the reconstructed volume
        obj_recon = cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
        temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
        img_padded = cp.zeros((size_y, size_x), dtype=cp.float32)        
        img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
        ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
        loss = cp.zeros(max_iter, dtype=cp.float32)

        for it in tqdm(range(n_img), desc="Reconstructing volumes"):
            loss.fill(0)
            img = cp.array(data[it, :,:]).astype(cp.float32)[crop[0]:crop[1], crop[2]:crop[3]]
            if img_subtract_bg:
                img -= bg
            if img_clip:
                img = cp.clip(img, 0, None)
            if img_mask:
                img = img * mask
            img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] = img

            for n_iter in tqdm(range(max_iter), leave=False, desc=f"Deconvolving image {it+1}"):
                img_est.fill(0)
                if not reuse_prev_vol:# otherwise reusing obj_recon from previous frame bc they are probably very similar
                    obj_recon.fill(0)
                
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
                ratio_ = ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
                calc_loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
                if calc_loss < loss_threshold:
                    loss[n_iter] = calc_loss
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

    kwargs = dict(max_iter=max_iter,
                  xy_pad=xy_pad,
                  roi_size=roi_size,
                  loss_threshold=loss_threshold,
                  psf_downsample=psf_downsample,
                  OTF_normalize=OTF_normalize,
                  OTF_clip=OTF_clip,
                  img_subtract_bg=img_subtract_bg,
                  img_mask=img_mask,
                  img_clip=img_clip,
                  reuse_prev_vol=reuse_prev_vol,)
    
    return kwargs, save_fn

def reconstruct_vol_from_img(img,
                             psf=None,
                             obj_0=None,
                             bg=None,
                             circle_mask=None,
                             otf_path="",
                             max_iter=30,
                             xy_pad=201,
                             roi_size=300,
                             loss_threshold = 0,
                             psf_downsample=[0,None,1],
                             OTF_normalize=False,
                             OTF_clip=False,
                             img_subtract_bg=False,
                             img_mask=True,
                             img_clip=True,
                             verbose=True,
                             plot=False,
                             pad=10,
                             ):


    if psf is None and len(otf_path)==0:
        raise ValueError("No PSF or OTF provided")
    elif psf is None and len(otf_path)>0:
        print("Loading OTF from disk") if verbose else None
        OTF = cp.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Calculating OTF") if verbose else None
        size_y = psf.shape[1] + 2 * xy_pad
        size_x = psf.shape[2] + 2 * xy_pad
        if psf_downsample[1] is None:
            psf_downsample[1] = psf.shape[0]  
        else:
            psf_downsample[1] = psf[:psf_downsample[1],:,:].shape[0]    
        size_z = len(range(*psf_downsample))

        OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)

        for i,z in enumerate(tqdm(range(*psf_downsample))):
            slice_processed = cp.asarray(psf[z,:,:]).astype(cp.float32)
            if OTF_clip:
                slice_processed = cp.clip(slice_processed, 0, None)
            if OTF_normalize:
                slice_processed /= slice_processed.sum()
            OTF[i, :, :] = fft2(ifftshift(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
            # OTF[z, :, :] = fft2(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant'))
        if len(otf_path)>0:
            cp.save(otf_path, OTF)


    print("Initializing memory") if verbose else None
    if img_subtract_bg:
        assert bg is not None, "bg must be provided if img_subtract_bg is True"
        img -= bg
    if img_clip:
        img = cp.clip(img, 0, None)
    if img_mask:
        assert circle_mask is not None, "circle_mask must be provided if img_mask is True"
        img *= circle_mask
    img_padded = cp.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')

    obj_recon = obj_0 if obj_0 is not None else cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
    losses = cp.zeros(max_iter, dtype=cp.float32)
    plot_mip = cp.zeros(shape=(max_iter, 2*roi_size + size_z + 3 * pad, 2*roi_size + size_z + 3 * pad), dtype=cp.float32) if plot else None

    temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
    img_est = cp.zeros((size_y, size_x), dtype=cp.float32)
    ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)

    for it in tqdm(range(max_iter)):
        img_est.fill(0)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            img_est += cp.maximum(cp.real(ifft2(OTF[z, :, :] * fft2(temp_obj))), 0)
        ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            temp = temp_obj * (cp.maximum(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF[z, :, :]))), 0))
            obj_recon[z, :, :] = temp[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]

        if plot:
            plot_mip[it,:,:] = create_projection_image(obj_recon,pad=pad)

        ratio_ = ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
        calc_loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
        losses[it] = calc_loss
        if calc_loss < loss_threshold:
            losses = losses[:it]
            plot_mip = plot_mip[:it]
            break
    
    kwargs = dict(max_iter=max_iter,
                  xy_pad=xy_pad,
                  roi_size=roi_size,
                  loss_threshold=loss_threshold,
                  psf_downsample=psf_downsample,
                  OTF_normalize=OTF_normalize,
                  OTF_clip=OTF_clip,
                  img_subtract_bg=img_subtract_bg,
                  img_mask=img_mask,
                  img_clip=img_clip,)

    return obj_recon, plot_mip, losses, kwargs


def reconstruct_vol_from_img1(img,
                             psf=None,
                             obj_0=None,
                             bg=None,
                             circle_mask=None,
                             otf_path="",
                             max_iter=30,
                             xy_pad=201,
                             roi_size=300,
                             loss_threshold = 0,
                             psf_downsample=1,
                             OTF_subtract_bg=True,
                             OTF_normalize=True,
                             OTF_clip=False,
                             img_subtract_bg=False,
                             img_mask=True,
                             img_clip=True,
                             verbose=True,
                             plot=False,
                             pad=10,
                             ):

    
    

    if psf is None and len(otf_path)==0:
        raise ValueError("No PSF or OTF provided")
    elif psf is None and len(otf_path)>0:
        print("Loading OTF from disk") if verbose else None
        OTF = cp.load(otf_path) 
        size_z, size_y, size_x  = OTF.shape
    else:
        print("Calculating OTF") if verbose else None
        size_y = psf.shape[1] + 2 * xy_pad
        size_x = psf.shape[2] + 2 * xy_pad
        size_z = int(psf.shape[0]/psf_downsample)

        # print(f"PSF zspacing: {np.abs(np.diff(psf["z_positions"][::psf_downsample])).mean()*1000} um") if verbose else None	
        psf = cp.asarray(psf[::psf_downsample,:,:]).astype(cp.float32)
        if OTF_subtract_bg:
            assert bg is not None, "bg must be provided if OTF_subtract_bg is True"
            psf -= bg 
        if OTF_clip:
            psf = cp.clip(psf, 0, None)
        if OTF_normalize:
            assert psf is not None, "psf must be provided if OTF_normalize is True"
            psf /= psf.sum(axis=(1, 2), keepdims=True)
        
        psf = cp.pad(psf, ((0, 0), (xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')
        OTF = fft2(ifftshift(psf, axes=(1,2)), axes=(1,2))

        # psf = cp.asarray(psf).astype(cp.float32)
        # OTF = cp.zeros((size_z, size_y, size_x), dtype=cp.complex64)
        # for z in range(0,size_z,psf_downsample):
        #     slice_processed = cp.asarray(psf[z,:,:]).astype(cp.float32)
        #     if OTF_subtract_bg:
        #         slice_processed -= bg
        #     if OTF_clip:
        #         slice_processed = cp.clip(slice_processed, 0, None)
        #     if OTF_normalize:
        #         slice_processed /= slice_processed.sum()
        #     OTF[z, :, :] = fft2(ifftshift(cp.pad(slice_processed, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')))
        # if len(otf_path)>0:
        #     cp.save(otf_path, OTF)


    print("Initializing memory") if verbose else None
    if img_subtract_bg:
        assert bg is not None, "bg must be provided if img_subtract_bg is True"
        img -= bg
    if img_clip:
        img = cp.clip(img, 0, None)
    if img_mask:
        assert circle_mask is not None, "circle_mask must be provided if img_mask is True"
        img *= circle_mask
    img_padded = cp.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant')

    obj_recon = obj_0 if obj_0 is not None else cp.ones((size_z, 2 * roi_size, 2 * roi_size), dtype=cp.float32)
    losses = cp.zeros(max_iter, dtype=cp.float32)
    plot_mip = cp.zeros(shape=(max_iter, 2*roi_size + size_z + 3 * pad, 2*roi_size + size_z + 3 * pad), dtype=cp.float32) if plot else None
    #temp_obj = cp.zeros((size_z, size_y, size_x), dtype=cp.float32)
    temp_obj = cp.zeros((size_y, size_x), dtype=cp.float32)
    ratio_img = cp.zeros((size_y, size_x), dtype=cp.float32)
    img_est = cp.zeros((size_y, size_x), dtype=cp.float32)

    for it in tqdm(range(max_iter)):
        img_est.fill(0)
        # temp_obj.fill(0)

        # temp_obj[:,size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[:, :, :]
        # img_est = cp.clip(cp.real(ifft2(OTF * fft2(temp_obj), axes=(1,2)), axes=(1,2)), 0, None).sum(axis=0)

        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            img_est += cp.maximum(cp.real(ifft2(OTF[z, :, :] * fft2(temp_obj))), 0)
        ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad] / (
                    img_est[xy_pad:-xy_pad, xy_pad:-xy_pad] + cp.finfo(cp.float32).eps)
        # temp_obj[:,size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[:, :, :]
        # temp_obj *= cp.clip(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF), axes=(1,2))), 0, None)
        # obj_recon[:, :, :] = temp_obj[:, size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]
        for z in range(size_z):
            temp_obj[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size] = obj_recon[z, :, :]
            temp = temp_obj * (cp.maximum(cp.real(ifft2(fft2(ratio_img) * cp.conj(OTF[z, :, :]))), 0))
            obj_recon[z, :, :] = temp[size_y // 2 - roi_size: size_y // 2 + roi_size, size_x // 2 - roi_size: size_x // 2 + roi_size]

        if plot:
            plot_mip[it,:,:] = create_projection_image(obj_recon,pad=pad)

        ratio_ = ratio_img[xy_pad:-xy_pad, xy_pad:-xy_pad]
        calc_loss = cp.mean(cp.abs(cp.log(ratio_[ratio_>0])))
        losses[it] = calc_loss
        if calc_loss < loss_threshold:
            losses = losses[:it]
            plot_mip = plot_mip[:it]
            break

    return obj_recon, plot_mip, losses
