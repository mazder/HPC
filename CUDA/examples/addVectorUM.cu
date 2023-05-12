#include<iostream>
#include<omp.h>
#include<cassert>

void printV(int *a, int *b, int *c, int N ){
    for(int i=0; i<10; i++){
        printf("%d %d %d\n",a[i],b[i],c[i]);
    }
}

void initVector(int* a, int* b, int N){
    for(int i=0; i<N; i++){
        a[i]=i%2;
        b[i]=i%2;
    }
}

void check(int* a, int* b, int* c, int N){
    for(int i=0; i<N; i++){
        assert(c[i]==a[i]+b[i]);
    }
    std::cout<<"Verification Success"<<std::endl;
}

__global__ void vectorAdd(int *A, int* B, int* C, int N){

    int tid = (blockIdx.x*blockDim.x)+threadIdx.x;

    if (tid<N){
        C[tid]=A[tid]+B[tid];
    }
}

// vector add
void vectorAddUM(){

    // Number of elements 2^16 (65536 elements)
    int N = 1<<16;

    //size_t bytes = N*sizeof(int);

    int *a_u, *b_u, *c_u; // host & device pointers



    // allocate in device
    cudaMallocManaged(&a_u, N*sizeof(int));
    cudaMallocManaged(&b_u, N*sizeof(int));
    cudaMallocManaged(&c_u, N*sizeof(int));

    // initialize a_u and b_u
    initVector(a_u, b_u, N);

    int blockSize = 256; // Dg, number of threads per block
    int numBlocks = (N+blockSize -1)/blockSize; // Db, number of blocks
    vectorAdd<<<numBlocks,blockSize>>>(a_u, b_u, c_u, N);

    cudaDeviceSynchronize();

    check(a_u, b_u, c_u, N);

    // deallocate memory
    cudaFree(a_u);
    cudaFree(b_u);
    cudaFree(c_u);
    std::cout<<"Completed Successfully"<<std::endl;
}


int main(int argc, char* argv[])
{

    vectorAddUM();

    return 0;
}
