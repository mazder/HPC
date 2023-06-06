#include<iostream>
#include "kernels.cuh"

__global__ void max_device(int *arr_d, unsigned int N, int* mx){

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x; // total number of threads work in parllel

    __shared__ int arr_cache[256];

    int tmp = -1;

    for(unsigned int i=tid; i<N; i+=stride){
        tmp = max(tmp,arr_d[i]);
    } 
    arr_cache[threadIdx.x]=tmp;
    __syncthreads();

    // reduction
    for(unsigned int s=blockDim.x/2; s>0; s/=2){
        if(threadIdx.x <s){
            arr_cache[threadIdx.x] = max(arr_cache[threadIdx.x], arr_cache[threadIdx.x+s]);
        }
        __syncthreads();
    }
    
    if(threadIdx.x==0){
        atomicMax(mx,arr_cache[0]);
    }

}