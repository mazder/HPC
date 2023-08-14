#ifndef __CONVOLUTION_CUH__
#define __CONVOLUTION_CUH__

__global__ void convolution_2D_device_global(int* matrix, int* outmatrix, int H, int W, int* mask_d, int MASKDIM, int MASK_OFFSET);
__global__ void convolution_device(int* matrix, int* outmatrix, int H, int W, int* mask, int MASKD);
#endif
