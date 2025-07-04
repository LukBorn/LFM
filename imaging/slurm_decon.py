from reconstruction import reconstruct_vols_from_imgs
from i_o import Paths
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os, glob, h5py
import importlib, pathlib
from daio.h5 import lazyh5
from video import create_projection_image, recording_to_overlay_preview, showvid, get_lenses


pn_psfs = r"~/hpc-rw/lfm/psfs"
pn_rec = r"~/hpc-r/lfm2025/recordings"
pn_bg = r"~/hpc-rw/lfm/bg"
url_home = r"/home/lubo12/"
pn_out = r"~/hpc-rw/lfm/results"

dataset_name = "20250701_1605_LB_ZF_v552_f6_40fps_1"
psf_name = "20250701_1216_PSF_LB_noap_1"
bg_name = "20250701_1651_LB_bg_40fps.npy"

paths = Paths(dataset_name=dataset_name,
              psf_name = psf_name,
              bg_name=bg_name,
              pn_bg= pn_bg,
              pn_rec = pn_rec,
              pn_psfs=pn_psfs,
              pn_out=pn_out,
              url_home=url_home,
              )

reconstruct_vols_from_imgs_parallel(paths,
                                    roi_size=550,
                                    loss_threshold=0,
                                    reuse_prev_vol = False,
                                    psf_downsample = [80,None,1],
                                    OTF_normalize=True,
                                    OTF_clip=False,
                                    img_subtract_bg=True,
                                    img_mask=True,
                                    plot_decon=True,
                                    fully_batched=False,
                                    vmin=0,
                                    vmax=70,
                                    absolute_limits=False)