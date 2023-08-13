#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

#include "functions.hpp"

__global__ void convolution_2D_device_global(int* matrix, int* outmatrix, int H, int W, int* mask_d);
__global__ void convolution_device(int* matrix, int* outmatrix, int H, int W, int* mask, int MASKD);
#endif
