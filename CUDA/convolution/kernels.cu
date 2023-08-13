#include<iostream>
#include "kernels.cuh"

__device__ void show_matrix(int* matrix, int H, int W){
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            printf("%d ",matrix[i*W+j]);
        }
        printf("\n");
    }
}

__global__ void convolution_2D_device_global(int* matrix, int* outmatrix, int H, int W, int* mask_d){

    int row_H = blockIdx.y*blockDim.y+threadIdx.y;
    int col_W = blockIdx.x*blockDim.x+threadIdx.x;

    int r_offset=row_H-MASK_OFFSET;
    int c_offset=col_W-MASK_OFFSET;
    int temp=0;

    // boundary
    if(row_H<H && col_W<W){
        for(int i=0; i<MASKDIM; i++){
            for(int j=0; j<MASKDIM; j++){
                if((r_offset+i)>=0 && (r_offset+i)<H){
                    if((c_offset+j)>=0 && (c_offset+j)<W){
                        temp+=matrix[(r_offset+i)*W+(c_offset+j)]*mask_d[i*MASKDIM+j];
                    }
                }
            }
        }
        outmatrix[row_H*W+col_W]=temp;
    }
}
