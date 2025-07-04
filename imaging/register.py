import numpy as np
import cupy as cp
import h5py 
import json
from tqdm.auto import tqdm
from video import AVWriter, create_projection_image, AVWriter2
from i_o import volume_reader



def reference_volume(paths,
                     ref_idx =[0,200,1],
                     preprocess = lambda x: x,
                     vmax=100,
                     vmin=0,
                     absolute_limits=False,
                     fps=None,
                     verbose = True,
                     **kwargs,
                     ):
    assert len(ref_idx) ==3, "ref_idx must be tuple of [start:stop:step] to index into the data with"
    idx = range(*ref_idx)

    if fps is None:
        fps = json.load(open(paths.meta))["acquisition"]["fps"]
    with h5py.File(paths.deconvolved, 'r') as f:
        zpos = np.array(f['zpos'])

    reader = volume_reader(paths.deconvolved, key='data', i_frames=idx)
    averager = Averager(shape=reader.get_shape('data')[1:])
    video_fn = paths.pn_outrec+ f'/reference_f{ref_idx}.mp4'
    video_writer=AVWriter2(video_fn,
                           fps = int(json.load(open(paths.meta))["acquisition"]["fps"]if fps is None else fps),
                           expected_indices=idx,)
    for frame_n, im in tqdm(reader, desc="Averaging"):
        data = preprocess(im, **kwargs)
        averager.step(data)
        mip = create_projection_image(data,
                                      vmax=vmax, vmin=vmin, absolute_limits=absolute_limits,
                                      zpos=zpos, scalebar = 200, text=f"f{i}",)
        video_writer.write(mip)
    ref_vol = averager.retrieve()
    video_writer.close()
    return ref_vol, video_fn










class Averager:
    def __init__(self, shape):
        self.avg = cp.zeros(shape = shape)
        self.n=0
    def step(self, vol):
        assert vol.shape == self.avg.shape, "vol must have same shape as average"
        vol = cp.asarray(vol)
        self.n += 1
        self.avg += (vol - self.avg) / self.n
        
    def retrieve(self):
        return self.avg.get()