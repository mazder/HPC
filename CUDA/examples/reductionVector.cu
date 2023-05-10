#include<iostream>
#include<omp.h>
#include<cassert>

#define SHEMSIZE 256

__global__ void sumVector(int *v, int* vin){

    __shared__ int part_sum[SHEMSIZE];
    int tid = (blockIdx.x*blockDim.x)+threadIdx.x;

    part_sum[tid]=v[tid];
    __syncthreads();

    for(int stride=0; stride<blockDim.x; stride*=2 ){
        if(threadIdx.x%(2*stride)==0){
            part_sum[threadIdx.x]+=part_sum[threadIdx.x+stride];
        }
    }
    __syncthreads();

    if(threadIdx.x==0){
        vin[blockIdx.x] = part_sum[0];
    }
}

void initVector(int* a,  int N){
    for(int i=0; i<N; i++){
        a[i]=1;
    }
}

// vector add
void vectorReductionNative(){

    // Number of elements
    int N = 1<<16;

    //size_t bytes = N*sizeof(int);

    int *a_h, *b_h, *c_h; // host pointers
    int *a_d, *b_d, *c_d; // device pointers

    //allocate in host memory // usually pagable allocation

    a_h = (int *) malloc(N*sizeof(int));
    b_h = (int *) malloc(N*sizeof(int));
    c_h = (int *) malloc(N*sizeof(int));

    // initialize a_h and b_h
    initVector(a_h, N);

    // allocate in device
    cudaMalloc(&a_d, N*sizeof(int));
    cudaMalloc(&b_d, N*sizeof(int));
    cudaMalloc(&c_d, N*sizeof(int));

    // Host to Device
    cudaMemcpy(a_d, a_h, N*sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256; // Dg, number of threads per block
    int numBlocks = (N)/blockSize; // Db, number of blocks

    sumVector<<<numBlocks,blockSize>>>(a_d, b_d);

    cudaDeviceSynchronize();

    sumVector<<<1,blockSize>>>(b_d, c_d);

    cudaDeviceSynchronize();

    // Device to Host
    cudaMemcpy(c_h, c_d, N*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout<<c_h[0]<<std::endl;
    // deallocate host memory
    free(a_h);
    free(b_h);
    free(c_h);

    // deallocate device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}


int main(int argc, char* argv[])
{

    vectorReductionNative();

    return 0;
}

