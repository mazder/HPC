#include<iostream>
#include "convolution.cuh"

__device__ void show_matrix(int* matrix, int H, int W){
    printf("=============\n");
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            printf("%d ",matrix[i*W+j]);
        }
        printf("\n");
    }
}

__global__ void convolution_2D_device_global(int* matrix, int* outmatrix, int H, int W, int* mask_d, int MASKDIM, int MASK_OFFSET){

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

__global__ void convolution_device(int* matrix, int* outmatrix, int H, int W, int* mask, int MASKD){


    int row_H = blockIdx.y*blockDim.y+threadIdx.y;
    int col_W = blockIdx.x*blockDim.x+threadIdx.x;

    int temp=0;
    int r_offset;
    int c_offset;
    int new_H=(H-MASKD+1);
    int new_W=(W-MASKD+1);

    // boundary
    if(row_H<new_H && col_W<new_W){
        for(int i=row_H; i<MASKD+row_H; i++){
            r_offset=i-row_H;
            for(int j=col_W; j<MASKD+col_W; j++){
                c_offset=j-col_W;
                temp+=mask[r_offset*MASKD+c_offset]*matrix[i*W+j];
            }
        }
        outmatrix[row_H*new_W+col_W]=temp;
        //printf("%d %d %d %d \n", row_H, col_W, row_H*new_W+col_W, temp);
    }
/*
    if(row_H==0 && col_W==0){
        show_matrix(mask, MASKD, MASKD);
        show_matrix(matrix, H, W);
        show_matrix(outmatrix, new_H, new_W);
    }
*/
}
