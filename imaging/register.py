import numpy as np
import cupy as cp
import h5py 
import json
from tqdm.auto import tqdm
from video import create_projection_image, AVWriter2
from i_o import VolumeReader, AsyncH5Writer, get_available_gpus
import warpfield
from warpfield import Recipe, register_volumes
import threading
import time, os
from skimage.metrics import structural_similarity as ssim
from cupyx.scipy.ndimage import median_filter, maximum_filter
from signal_extraction import NeighborhoodCovMapper, NeighborhoodCorrMapper

def average_volumes(paths,
                    ref_idx =[100,120,1],
                    preprocess = lambda x: x,
                    vmax=100,
                    vmin=0,
                    absolute_limits=False,
                    transpose=False,
                    fps=None,
                    **kwargs,
                    ):
    assert len(ref_idx) ==3, "ref_idx must be tuple of [start:stop:step] to index into the data with"
    idx = range(*ref_idx)

    if fps is None:
        fps = json.load(open(paths.meta))["acquisition"]["fps"]
    with h5py.File(paths.deconvolved, 'r') as f:
        try:
            zpos = np.array(f['zpos'])
        except Exception as e:
            print(f"Error reading zpos from metadata: {e}, setting zpos to None")
            zpos = None

    reader = VolumeReader(paths.deconvolved, key='data', i_frames=idx)
    averager = Averager()
    video_fn = paths.pn_outrec+ f'/reference_f{ref_idx}.mp4'
    video_writer=AVWriter2(video_fn,
                           fps = int(json.load(open(paths.meta))["acquisition"]["fps"]if fps is None else fps),
                           expected_indeces=idx,)
    for frame_n, im in tqdm(reader, desc="Averaging"):
        data = preprocess(im, **kwargs)
        averager.step(data)
        mip = create_projection_image(data,
                                      vmax=vmax, vmin=vmin, absolute_limits=absolute_limits,
                                      zpos=zpos, scalebar = 200, text=f"f{frame_n}",transpose=transpose)
        video_writer.write(mip, frame_n)
    ref_vol = averager.retrieve()
    video_writer.close()
    return ref_vol, video_fn




def mini_registration(paths,
                      idx,
                      ref_vol,
                      recipe,
                      fn_addendum='',
                      preprocess=lambda x: x,
                      vmax=100,
                      vmin=0,
                      absolute_limits=False,
                      transpose=False,
                      fps=None,
                      verbose=False
                      ):
    assert len(idx) == 3, "idx must be tuple of [start:stop:step] to index into the data with"
    if fps is None:
        fps = json.load(open(paths.meta))["acquisition"]["fps"]
    with h5py.File(paths.deconvolved, 'r') as f:
        try:
            zpos = np.array(f['zpos'])
        except Exception as e:
            print(f"Error reading zpos from metadata: {e}, setting zpos to None")
            zpos = None
    with h5py.File(paths.reg_mask, 'r') as f:
        try:
            reg_mask = cp.asarray(f['mask_3d']).transpose(0,2,1)
        except Exception as e:
            print(f"Error reading reg_mask from metadata: {e}, setting reg_mask to None")
            reg_mask = 1.0

    reader = VolumeReader(paths.deconvolved, key='data', i_frames=range(*idx))

    video_fn = paths.pn_outrec + f'/mini_registration_f{idx}_vmin{vmin}_vmax{vmax}{"_al" if absolute_limits else ""}.mp4'
    video_writer = AVWriter2(video_fn,fps=fps, expected_indeces=range(*idx),)

    video_reg_fn = paths.pn_outrec + f'/mini_registration_registered{"_" if fn_addendum else ""}{fn_addendum}_f{idx}_vmin{vmin}_vmax{vmax}{"_al" if absolute_limits else ""}.mp4'
    video_reg_writer = AVWriter2(video_reg_fn, fps=fps, expected_indeces=range(*idx),)

    metrics = {'correlation': np.zeros(len(reader), dtype=np.float32),
               'mse': np.zeros(len(reader), dtype=np.float32),
               'ssim': np.zeros(len(reader), dtype=np.float32),
               'r': np.zeros(len(reader), dtype=np.float32),
               'dmf': np.zeros(len(reader), dtype=np.float32)}
    print(metrics.keys())

    ref_vol = cp.asarray(ref_vol)
    warpfields = []
    for i, (frame_n, im) in enumerate(tqdm(reader, desc="Mini registration")):
        data = preprocess(im)
        data = cp.asarray(data)
        mip = create_projection_image(data,
                                      vmax=vmax, vmin=vmin, absolute_limits=absolute_limits,
                                      zpos=zpos, scalebar=200, text=f"f{frame_n}",transpose=transpose)
        video_writer.write(mip, frame_n)
        registered_vol, _warpfield, _ = register_volumes(ref_vol, data, recipe)
        registered_vol *= reg_mask
        mip_reg = create_projection_image(registered_vol,
                                      vmax=vmax, vmin=vmin, absolute_limits=absolute_limits,
                                      zpos=zpos, scalebar=200, text=f"f{frame_n}",text_size = 2,transpose=transpose)
        warpfields.append(_warpfield)
        corr, mse, ssim_val, r = calculate_metrics(ref_vol, registered_vol, verbose)
        metrics['correlation'][i] = corr
        metrics['mse'][i] = mse
        metrics['ssim'][i] = ssim_val
        metrics['r'][i] = r
        video_reg_writer.write(mip_reg, frame_n)
    video_writer.close()
    video_reg_writer.close()
    metrics['dmf'] = (maximum_filter(median_filter(cp.asarray(metrics['r']), 3), size=21) - cp.asarray(metrics['r'])).get()
    return video_fn, warpfields, metrics

def calculate_metrics(ref_vol, mov_vol, verbose=False):
    # Flatten volumes for metrics
    t0 = time.time()
    ref = cp.asarray(ref_vol).flatten()
    reg = cp.asarray(mov_vol).flatten()
    # Cross-correlation
    corr = cp.corrcoef(ref, reg)[0, 1]
    # MSE
    mse = cp.mean((ref - reg) ** 2)
    # SSIM (for 2D slices, e.g., middle slice) -- minimal CuPy implementation
    ref2d = cp.asarray(ref_vol[ref_vol.shape[0]//2]).astype(cp.float32)
    mov2d = cp.asarray(mov_vol[mov_vol.shape[0]//2]).astype(cp.float32)
    K1, K2 = 0.01, 0.03
    L = 1.0 if ref2d.max() <= 1.0 else 255.0
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = cp.mean(ref2d)
    mu2 = cp.mean(mov2d)
    sigma1_sq = cp.mean((ref2d - mu1)**2)
    sigma2_sq = cp.mean((mov2d - mu2)**2)
    sigma12 = cp.mean((ref2d - mu1)*(mov2d - mu2))
    ssim_val = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_val = float(ssim_val.get())

    if verbose:
        print(f"Metrics calculated in {time.time() - t0:.2f} seconds")
    return float(corr), -float(mse), float(ssim_val)



def register_recording(paths):

    with h5py.File(paths.reg_recipe, 'r') as f:
        recipe_yaml_path = str(f['recipe_path'][()].decode("utf-8"))
        recipe = Recipe.from_yaml(recipe_yaml_path)
        ref_vol = cp.asarray(f['ref_vol'])
        crop = np.asarray(f['crop'])
        vid_params = json.loads(f['vid_params'][()])
    
    with h5py.File(paths.reg_mask, 'r') as f:
        try:
            reg_mask = cp.asarray(f['mask_3d'])
        except Exception as e:
            print(f"Error reading reg_mask from metadata: {e}, setting reg_mask to None")
            reg_mask = 1.0

    
    reader = VolumeReader(paths.deconvolved, key='data', i_frames=None)
    if ref_vol.shape == reader.get_shape("data"):
        crop = (0, ref_vol.shape[1], 0, ref_vol.shape[2])
    
    testvol, testwarp, _ = register_volumes(ref_vol, ref_vol, recipe)

    writer = AsyncH5Writer(paths.registered)
    writer.create_dataset('data', shape=(reader.len, *testvol.shape), dtype=np.float32)
    writer.create_dataset('warpfields', shape=(reader.len, *testwarp.warp_field.shape), dtype=np.float32)
    writer.write_dataset('block_size', testwarp.block_size.get())
    writer.write_dataset('block_stride', testwarp.block_stride.get())
    writer.write_dataset('ref_vol', ref_vol.get())
    writer.write_meta('recipe', {"path": recipe_yaml_path})
    writer.create_dataset("metrics", shape=(reader.len, 3), dtype=np.float32)
    
    cov_mapper = NeighborhoodCovMapper(tau=recipe.cov_tau)
    corr_mapper = NeighborhoodCorrMapper(tau=recipe.cov_tau)
    corr_mapper_2 = NeighborhoodCorrMapper(tau=2)

    if vid_params["write_video"]:
        video_fn = paths.pn_outrec + f'/registered{"_T" if vid_params["transpose"] else ""}_vmin{vid_params["vid"]["vmin"]}-vmax{vid_params["vid"]["vmax"]}{"_al" if vid_params["vid"]["absolute_limits"] else ""}.mp4'
        video_writer = AVWriter2(video_fn, fps=vid_params["fps"], expected_indeces=range(reader.len), verbose=False)
        print(f"Writing video to {video_fn}")
    if vid_params["write_dff_video"]:
        dff_video_fn = paths.pn_outrec + f'/registered_dff{"_T" if vid_params["transpose"] else ""}_vmin{vid_params["dff"]["vmin"]}-vmax{vid_params["dff"]["vmax"]}{"_al" if vid_params["dff"]["absolute_limits"] else ""}.mp4'
        dff_video_writer = AVWriter2(dff_video_fn, fps=vid_params["fps"], expected_indeces=range(reader.len), verbose=False)
        average_vol = cp.ones_like(ref_vol, dtype=cp.float32) 
        print(f"Writing dff video to {dff_video_fn}")

    for frame_n, vol in tqdm(reader, desc="Registering"):
        vol = cp.asarray(vol)[:, crop[0]:crop[1], crop[2]:crop[3]]
        registered_vol, warpfield, _ = register_volumes(ref_vol, vol, recipe)
        registered_vol *= reg_mask
        
        writer.write("warpfields", warpfield.warp_field.get(), frame_n)
        writer.write("data", registered_vol.get(), frame_n)
        metrics = np.array(calculate_metrics(ref_vol, registered_vol, verbose=False))
        writer.write("metrics", metrics, frame_n)

        if metrics[0] > recipe.r_threshold:
            cov_mapper.step(registered_vol)
            corr_mapper.step(registered_vol)
            corr_mapper_2.step(registered_vol)

        if vid_params["write_video"]:
            mip = create_projection_image(registered_vol, 
                                          vmax=vid_params["vid"]["vmax"], vmin=vid_params["vid"]["vmin"], absolute_limits=vid_params["vid"]["absolute_limits"],
                                          zpos=vid_params["zpos"], scalebar=vid_params["scalebar"], text=f"f{frame_n}",transpose=vid_params["transpose"])
            video_writer.write(mip, frame_n)
        
        if vid_params["write_dff_video"]:
            average_vol = (1/vid_params["dff"]["tau"]) * registered_vol+ (1-1/vid_params["dff"]["tau"]) * average_vol
            dff_vol = (registered_vol - average_vol) / average_vol
            dff_mip = create_projection_image(dff_vol,
                                              vmax=vid_params["dff"]["vmax"], vmin=vid_params["dff"]["vmin"], absolute_limits=vid_params["dff"]["absolute_limits"],
                                              zpos=vid_params["zpos"], scalebar=vid_params["scalebar"], text=f"f{frame_n}",transpose=vid_params["transpose"])
            dff_video_writer.write(dff_mip, frame_n)
    cov_map, var_map = cov_mapper.retrieve()
    writer.write_dataset('cov_map', cov_map)
    writer.write_dataset('var_map', var_map)
    writer.write_dataset('corr_map', corr_mapper.retrieve().get())
    writer.write_dataset('corr2_map', corr_mapper_2.retrieve().get())
    writer.close()
    if vid_params["write_video"]:
        video_writer.close()
    if vid_params["write_dff_video"]:
        dff_video_writer.close()





def registered_volume_reader(paths, idx=None):
    vol_reader = VolumeReader(paths.deconvolved, key='data', i_frames=idx)
    warpfield_reader = VolumeReader(paths.registered, key='warpfields', i_frames=idx)
    volshape = vol_reader.get_shape("data")[1:]
    
    # Read metadata needed for WarpField construction
    with h5py.File(paths.registered, 'r') as f:
        block_size = f['block_size'][()]
        block_stride = f['block_stride'][()]
    
    warp_field = warpfield.register.WarpMap(
            warp_field=[],
            block_size= block_size,
            block_stride = block_stride,
            ref_shape = volshape,
            mov_shape = volshape
        )
    
    for (frame_n, vol), (_, warp_data) in zip(vol_reader, warpfield_reader):
        # Construct WarpField object from saved data
        
        warp_field.warp_field = cp.asarray(warp_data)
        # Apply warpfield to volume
        vol_cp = cp.asarray(vol)
        warped_vol = warp_field.apply(vol_cp)
        
        yield frame_n, warped_vol.get() 



def save_register_recipe(paths, recipe, ref_vol, crop, vid_params=None, eye_mask=None, cov_tau=60):
    recipe_yaml_path = paths.reg_recipe[:-3] + '.yaml'  
    recipe.to_yaml(recipe_yaml_path)
    with h5py.File(paths.reg_recipe, 'w') as f:
        f.create_dataset('recipe_path', data=recipe_yaml_path)
        f.create_dataset('ref_vol', data=ref_vol)
        f.create_dataset('cov_tau', data=cov_tau)
        if crop is None:
            crop = (0, ref_vol.shape[1], 0, ref_vol.shape[2])
        f.create_dataset('crop', data=crop)
        assert ref_vol.shape == (ref_vol.shape[0],crop[1]-crop[0],crop[3]-crop[2]), "crop must be compatible with ref_vol shape"
        
        if eye_mask is not None:
            f.create_dataset("eyemask", data=eye_mask)

        if vid_params is None:
            vid_params = {
                "write_video": False,
                "write_dff_video": False,
                "fps": 30,
                "vid": {
                    "vmax": 100,
                    "vmin": 0,
                    "absolute_limits": False
                },
                "dff": {
                    "tau": 10,
                    "vmax": 100,
                    "vmin": 0,
                    "absolute_limits": False
                },
                "zpos": None,
                "scalebar": 200,
                "transpose": False
            }
        f.create_dataset("vid_params", data=json.dumps(vid_params).encode('utf-8'))
        


class Averager:
    def __init__(self, shape=None):
        self.shape = shape
        if self.shape is not None:
            self.init_avg()
        self.n=0
    def init_avg(self):
        self.avg = cp.zeros(shape=self.shape, dtype=cp.float32)

    def step(self, vol):
        if self.shape is None:
            self.shape = vol.shape
            self.init_avg()
        assert vol.shape == self.avg.shape, "vol must have same shape as average"
        vol = cp.asarray(vol)
        self.n += 1
        self.avg += (vol - self.avg) / self.n
        
    def retrieve(self):
        return self.avg.get()


def extract_traces(paths):
    ...



# def register_recording_parallel(paths,):
#     start_time = time.time()
#     stop_event = threading.Event()

#     with h5py.File(paths.recipe, 'r') as f:
#         recipe = ...
#         ref_vol = np.asarray(f['ref_vol'])
#         preprocess = ...

#     reader = VolumeReader(paths.deconvolved, key='data', i_frames=None)
#     writer = AsyncH5Writer(paths.registered,)
#     writer.create_dataset('warpfields', shape=(reader.len, ...), dtype=np.float32)
#     video_fn = paths.pn_outrec + f'/registered.mp4'
#     video_writer = AVWriter2(video_fn, fps=fps, expected_indeces=reader.i_frames)

#     n_gpus = get_available_gpus()

#     class GPUWorker:
#         def __init__(self,gpu_id):
#             self.gpu_id = gpu_id
#             cp.cuda.Device(gpu_id).use()
#             self.recipe = recipe
#             self.ref_vol = cp.asarray(ref_vol)
#             self.preprocess = preprocess
#         def register(self, vol):
#             vol = cp.asarray(vol)
#             vol = self.preprocess(vol)
#             registered_vol, warpfield, _ = warpfield.register_volumes(self.ref_vol, vol, self.recipe)
#             return registered_vol.get(), warpfield

#     def gpu_worker_loop(gpu_id):
#         worker = GPUWorker(gpu_id)
#         try:
#             while not stop_event.is_set():
#                 frame_n, vol = reader.get_next()
#                 if vol is None:
#                     break
#                 registered_vol, warpfield = worker.register(vol)
#                 writer.write('warpfields', warpfield, frame_n)
#                 mip = create_projection_image(registered_vol, vmax=100, vmin=0, absolute_limits=False,
#                                               zpos=None, scalebar=200, text=f"f{frame_n}")
#                 video_writer.write(mip, frame_n)
#         except Exception as e:
#             print(f"Error in GPU worker {gpu_id}: {e}")
    
#     # Start GPU worker threads
#     gpu_threads = []
#     for gpu_id in range(n_gpus):
#         t = threading.Thread(target=gpu_worker_loop, args=(gpu_id,))
#         t.start()
#         gpu_threads.append(t)

#     # Stop the IO threads
#     for t in gpu_threads:
#         t.join()
#     stop_event.set()
#     reader.close()
#     writer.close()
#     video_writer.close()
#     print(f"Registration finished in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}")

