import numpy as np
from tqdm.auto import tqdm
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def reconstruct_vol_from_img(img,
                             psf,
                             n_iter,
                             xy_pad,
                             roi_size,
                             verbose,
                             ):
    size_x, size_y = psf.shape[0]+2*xy_pad, psf.shape[1]+2*xy_pad
    size_z = psf.shape[2]
    print("Initalizing Memory") if verbose else None

    OTF = np.zeros((size_x, size_y, size_z),dtype=np.complex64)

    for i in range(size_z):
        OTF[:,:,i] = fft2(ifftshift(np.pad(psf[:,:,i], ((xy_pad,xy_pad), (xy_pad,xy_pad)), mode='constant', constant_values=0)))

    temp_1 = np.zeros((size_x, size_y, size_z), dtype=np.complex64)#todo the NxyAdd
    temp_2 = np.zeros((size_x, size_y, size_z), dtype=np.complex64)
    temp_3 = np.zeros((size_x, size_y), dtype=np.float32)

    img_padded = np.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant', constant_values=0)

    temp_obj_recon = np.zeros((size_x, size_y), dtype=np.float32)
    obj_recon = np.zeros((2*roi_size,2*roi_size,size_z), dtype=np.float32)
    img_estimate = np.zeros((size_x, size_y), dtype=np.float32)
    ratio = np.zeros((size_x, size_y), dtype=np.float32)
    print("Finished Initalizing Memory") if verbose else None

    for i in tqdm(range(n_iter)):
        img_estimate *= 0
        for j in range(size_z):
            temp_obj_recon[:,:] = obj_recon[:,:,j] #todo
            temp_1[xy_pad+1:-xy_pad] = fftshift(fft2(temp_obj_recon))
            temp_2=np.abs(ifft2(ifftshift(temp_1)))
            img_estimate = img_estimate + np.max()




