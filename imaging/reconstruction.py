import numpy as np
import cupy as cp
from tqdm.auto import tqdm
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def reconstruct_vol_from_img(img,
                             psf_1,
                             psf_2,
                             n_iter,
                             ratio,
                             xy_pad,
                             roi_size,
                             verbose,
                             ):
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    size_x, size_y = psf_1.shape[0]+2*xy_pad, psf_1.shape[1]+2*xy_pad
    size_z = psf_1.shape[2]
    print("Initalizing Memory") if verbose else None

    OTF_A = np.zeros((size_x, size_y, size_z), dtype=np.complex64)
    OTF_B = np.zeros((size_x, size_y, size_z), dtype=np.complex64)

    for i in range(size_z):
        OTF_A[:,:,i] = fft2(ifftshift(np.pad(psf_1[:,:,i], ((xy_pad,xy_pad), (xy_pad,xy_pad)), mode='constant', constant_values=0)))
        OTF_B[:,:,i] = fft2(ifftshift(np.pad(psf_2[:,:,i], ((xy_pad,xy_pad), (xy_pad,xy_pad)), mode='constant', constant_values=0)))

    size_x_add = round((size_x/ratio-size_x)/2)
    size_y_add = round((size_y/ratio-size_y)/2)
    size_x_sub = round(size_x*(1-ratio)/2)+size_x
    size_y_sub = round(size_y*(1-ratio)/2)+size_y

    temp_1 = np.zeros((size_x+2*size_x_add, size_y+2*size_y_add), dtype=np.complex64)
    temp_2 = np.zeros((size_x+2*size_x_add, size_y+2*size_y_add), dtype=np.complex64)
    temp_3 = np.zeros((size_x, size_y), dtype=np.float32)

    img_padded = np.pad(img, ((xy_pad, xy_pad), (xy_pad, xy_pad)), mode='constant', constant_values=0)

    temp_obj_recon = np.zeros((size_x, size_y), dtype=np.float32)
    obj_recon = np.zeros((2*roi_size,2*roi_size,size_z), dtype=np.float32)
    img_estimate = np.zeros((size_x, size_y), dtype=np.float32)
    # ratio = np.zeros((size_x, size_y), dtype=np.float32)
    print("Finished Initalizing Memory") if verbose else None

    for i in tqdm(range(n_iter)):
        img_estimate *= 0
        for j in range(size_z):
            temp_obj_recon[size_x/2-roi_size+1:,:] = obj_recon[:,:,j] #todo
            temp_1[xy_pad:-xy_pad] = fftshift(fft2(temp_obj_recon))
            temp_2 = np.abs(ifft2(ifftshift(temp_1)))
            img_estimate += np.maximum(np.real(ifft2(OTF_A[:,:,j])*fft2(temp_obj_recon)),0)
            img_estimate += np.maximum(np.real(ifft2(OTF_B[:,:,j])*fft2(temp_2[size_x_add:-size_x_add,size_y_add:-size_y_add])),0)

        temp_4 = img_padded[xy_pad:-xy_pad, xy_pad:-xy_pad]/(img_estimate[xy_pad:-xy_pad, xy_pad:-xy_pad] + np.finfo(np.float32).eps)
        ratio = np.full_like(temp_4, np.mean(temp_4,dtype=np.float32),dtype=np.float32)
        ratio *= (img_estimate > (np.max(img_estimate)/200)).astype(np.float32)
        ratio[xy_pad:-xy_pad, xy_pad:-xy_pad] = img_estimate[xy_pad:-xy_pad, xy_pad:-xy_pad] / (img_estimate[xy_pad:-xy_pad, xy_pad:-xy_pad] + np.finfo(np.float32).eps)
        temp_2.fill(0)

        for j in range(size_z):
            temp_1[size]