import numpy as np
import cupy as cp
import h5py 
import json
from tqdm.auto import tqdm
from video import create_projection_image, AVWriter2
from i_o import VolumeReader, AsyncH5Writer, get_available_gpus
import warpfield
import threading
import time



def average_volumes(paths,
                    ref_idx =[0,200,1],
                    preprocess = lambda x: x,
                    vmax=100,
                    vmin=0,
                    absolute_limits=False,
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
                                      zpos=zpos, scalebar = 200, text=f"f{frame_n}",)
        video_writer.write(mip, frame_n)
    ref_vol = averager.retrieve()
    video_writer.close()
    return ref_vol, video_fn


def registered_volume_reader(paths, idx=None):
    # assert that idx is a tuple of [start:stop:step] to index into the data with
    #assert that both the deconvolved and registered datasets exist with correct indexes
    ...


def mini_registration(paths,
                      idx,
                      ref_vol,
                      recipe,
                      preprocess=lambda x: x,
                      vmax=100,
                      vmin=0,
                      absolute_limits=False,
                      fps=None,
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
    reader = VolumeReader(paths.deconvolved, key='data', i_frames=range(*idx))
    video_fn = paths.pn_outrec + f'/mini_registration_f{idx}.mp4'
    video_writer = AVWriter2(video_fn,fps=fps, expected_indeces=range(*idx),)
    ref_vol = cp.asarray(ref_vol)
    warpfields = []
    for frame_n, im in tqdm(reader, desc="Mini registration"):
        data = preprocess(im)
        data = cp.asarray(data)
        registered_vol, _warpfield, _ = warpfield.register_volumes(ref_vol, data, recipe)
        mip = create_projection_image(registered_vol,
                                      vmax=vmax, vmin=vmin, absolute_limits=absolute_limits,
                                      zpos=zpos, scalebar=200, text=f"f{frame_n}",)
        # warpfields.append(_warpfield)
        video_writer.write(mip, frame_n)
    video_writer.close()
    return video_fn, warpfields


def register_recording_parallel(paths,):
    start_time = time.time()
    stop_event = threading.Event()

    with h5py.File(paths.recipe, 'r') as f:
        recipe = ...
        ref_vol = np.asarray(f['ref_vol'])
        preprocess = ...

    reader = VolumeReader(paths.deconvolved, key='data', i_frames=None)
    writer = AsyncH5Writer(paths.registered,)
    writer.create_dataset('warpfields', shape=(reader.len, ...), dtype=np.float32)
    video_fn = paths.pn_outrec + f'/registered.mp4'
    video_writer = AVWriter2(video_fn, fps=fps, expected_indeces=reader.i_frames)

    n_gpus = get_available_gpus()

    class GPUWorker:
        def __init__(self,gpu_id):
            self.gpu_id = gpu_id
            cp.cuda.Device(gpu_id).use()
            self.recipe = recipe
            self.ref_vol = cp.asarray(ref_vol)
            self.preprocess = preprocess
        def register(self, vol):
            vol = cp.asarray(vol)
            vol = self.preprocess(vol)
            registered_vol, warpfield, _ = warpfield.register_volumes(self.ref_vol, vol, self.recipe)
            return registered_vol.get(), warpfield

    def gpu_worker_loop(gpu_id):
        worker = GPUWorker(gpu_id)
        try:
            while not stop_event.is_set():
                frame_n, vol = reader.get_next()
                if vol is None:
                    break
                registered_vol, warpfield = worker.register(vol)
                writer.write('warpfields', warpfield, frame_n)
                mip = create_projection_image(registered_vol, vmax=100, vmin=0, absolute_limits=False,
                                              zpos=None, scalebar=200, text=f"f{frame_n}")
                video_writer.write(mip, frame_n)
        except Exception as e:
            print(f"Error in GPU worker {gpu_id}: {e}")
    
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
    video_writer.close()
    print(f"Registration finished in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}")


def get_eyemask():
    ...

def save_register_recipe():
    ...


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