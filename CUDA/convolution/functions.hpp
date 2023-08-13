#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#define MASKDIM 3
#define MASK_OFFSET (MASKDIM/2)

// convolution
void convolution_2D_host(int* matrix, int* outmatrix, int H, int W, int* mask);
#endif
