from i_o import Paths
import numpy as np
import cupy as cp
from reconstruction import reconstruct_vols_from_imgs_parallel
from tqdm.auto import tqdm
import os
import glob
import h5py

pn_psfs = r"~/hpc-rw/lfm/psfs"
pn_rec = r"~/hpc-r/lfm2025/recordings"
pn_bg = r"~/hpc-rw/lfm/bg"
url_home = r"/home/lubo12/"
pn_out = r"~/hpc-rw/lfm/results"

dataset_name = "20250514_1635_LF_ZF552_f4_1_comp_40fps"
psf_name = "20250509_1646_PSF_LB_redFB_1_30K_wo_coverslip_1"
bg_name = "20250602_1753_LB_bg_200s_2fps.npy"

paths = Paths(dataset_name=dataset_name,
              psf_name = psf_name,
              bg_name=bg_name,
              pn_bg= pn_bg,
              pn_rec = pn_rec,
              pn_psfs=pn_psfs,
              pn_out=pn_out
              )

kwargs = {'max_iter': 30,
 'xy_pad': 201,
 'roi_size': 300,
 'loss_threshold': 0,
 'psf_downsample': [100, 400, 1],
 'OTF_normalize': True,
 'OTF_clip': False,
 'img_subtract_bg': False,
 'img_mask': True,
 'img_clip': False,
 'reuse_prev_vol': False}

reconstruct_vols_from_imgs_parallel(paths,write_mip_video=True,**kwargs)