The folder includes:
 
XLFM_reconstruction_CPU.mat: Matlab program for XLFM reconstruction using CPU computing

XLFM_reconstruction_GPU.mat: Matlab program for XLFM reconstruction using GPU computing

GPU version runs much faster than CPU version. The minimum memory requirement is 32G for CPU version. It takes 3~5 minutes for one iteration depends on computer configuration. The minimum GPU onboard memory requirement is 6G.It takes 15s for one iteration when using nVidia Titan X 12G. To get satisfying results, it usually takes 30 iterations.

The input parameters include:
(1) PSFs.mat: It's a measured PSFs raw data stored in Matlab format; In this mat file, there should be two PSFs: PSF1 and PSF2. They are corresponding PSFs measured from group A and B microlenses, as shown in Supplementary Figure 3 & 4. 
(2) test.tif: It's a camera captured raw image to be reconstructed in 3D, as shown in Supplementary Figure 9.
(3) RatioAB:  It's the magnification ratio between Group A and B microlenses
(4) ItN:      It's the iteration number
(5) ROISize:  It's the size of region of interest you want to reconstruct

Having PSFs.mat and test.tif ready, the code is ready to run with above parameters set correctly.