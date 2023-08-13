#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include "util.hpp"
#include "kernels.cuh"
#include "functions.hpp"

__constant__ int mask_dd[MASKDIM*MASKDIM];

__global__ void convolution_2D_device_constant(int* matrix, int* outmatrix, int H, int W){

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
                        temp+=matrix[(r_offset+i)*W+(c_offset+j)]*mask_dd[i*MASKDIM+j];
                    }
                }
            }
        }
        outmatrix[row_H*W+col_W]=temp;
    }
}


void convolution(){


    // matrix dimention (2 ^ 10 x 2 ^ 12)
    //int MATRIX_H=5;
    //int MATRIX_W=7;
    int MATRIX_H = 1 << 10;
    int MATRIX_W = 1 << 12;

    // matrix and mask size in bytes
    size_t matrix_bytes = MATRIX_H*MATRIX_W*sizeof(int);
    size_t mask_bytes = MASKDIM*MASKDIM*sizeof(int);

    // allocate and initialize matrix and mask in host
    //std::vector<int> matrix(MATRIX_H, MATRIX_W);
    //init_matrix(matrix.data(), MATRIX_H, MATRIX_W);
    //int *matrix=malloc(MATRIX_H*MATRIX_W*sizeof(int));
    int *matrix_h=new int[MATRIX_H*MATRIX_W];
    init_matrix(matrix_h, MATRIX_H, MATRIX_W);
    //std::cout<<"=====Matrix"<<std::endl;
    //print_matrix(matrix_h, MATRIX_H, MATRIX_W);

    int *outmatrix_h=new int[MATRIX_H*MATRIX_W];
    int *outmatrix_devicetohost=new int[MATRIX_H*MATRIX_W];

    int *mask_h=new int[MASKDIM*MASKDIM];
    init_matrix(mask_h, MASKDIM, MASKDIM);
    //std::cout<<"=====Mask"<<std::endl;
    //print_matrix(mask_h, MASKDIM, MASKDIM);

    auto t1=std::chrono::high_resolution_clock::now();
    convolution_2D_host(matrix_h, outmatrix_h, MATRIX_H, MATRIX_W, mask_h);
    auto t2=std::chrono::high_resolution_clock::now();
    auto duration=std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

    //std::cout<<"Execution Time::> "<<duration<<"µs"<<std::endl;
    std::cout<<"Host Execution Time::> "<<duration<<"\xc2\xb5s"<<std::endl;

    //std::cout<<"=====Host OutMatrix"<<std::endl;
    //print_matrix(outmatrix_h, MATRIX_H, MATRIX_W);

    // device pointer for matrix and outmatrix
    int *matrix_d;
    int *mask_d;
    int *outmatrix_d;

    // allocate in device non-unified memory
    cudaMalloc(&matrix_d, matrix_bytes);
    cudaMalloc(&outmatrix_d, matrix_bytes);
    cudaMalloc(&mask_d, mask_bytes);

    // copy matrix and mask in device global memory
    cudaMemcpy(matrix_d, matrix_h, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(mask_d, mask_h, mask_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask_dd, mask_h, mask_bytes);


    // calculate grid dimension
    int NTHREADS = 8;

    dim3 dim_block(NTHREADS,NTHREADS);
    dim3 dim_grid( (MATRIX_W+NTHREADS-1/NTHREADS), (MATRIX_H+NTHREADS-1/NTHREADS) );
    t1=std::chrono::high_resolution_clock::now();
    //convolution_2D_device_global<<<dim_grid, dim_block>>>(matrix_d, outmatrix_d, MATRIX_H, MATRIX_W, mask_d);
    convolution_2D_device_constant<<<dim_grid, dim_block>>>(matrix_d, outmatrix_d, MATRIX_H, MATRIX_W);

    cudaDeviceSynchronize();
    t2=std::chrono::high_resolution_clock::now();
    duration=std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    std::cout<<"Device Execution Time::> "<<duration<<"\xc2\xb5s"<<std::endl;

    cudaMemcpy(outmatrix_devicetohost, outmatrix_d, matrix_bytes, cudaMemcpyDeviceToHost);

    //std::cout<<"=====Device OutMatrix"<<std::endl;
    //print_matrix(outmatrix_devicetohost, MATRIX_H, MATRIX_W);

    if(check_equivalent(outmatrix_h, MATRIX_H, MATRIX_W, outmatrix_devicetohost)){
        std::cout<<"Success"<<std::endl;
    }
    else{
        std::cout<<"Failed"<<std::endl;
    }

    // free host memory
    delete[] matrix_h;
    delete[] mask_h;
    delete[] outmatrix_h;
    delete[] outmatrix_devicetohost;

    // free cuda memory

    cudaFree(matrix_d);
    cudaFree(outmatrix_d);
    cudaFree(mask_d);

}

int main(int argc, char *argv[])
{

    std::cout<<"=================Call Convolution======================="<<std::endl;
    convolution();


    return 0;
}
