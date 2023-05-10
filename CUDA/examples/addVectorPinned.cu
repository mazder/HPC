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
void vectorAddNative(){

    // Number of elements
    int N = 17; //1<<16;

    //size_t bytes = N*sizeof(int);

    int *a_h, *b_h, *c_h; // host pointers
    int *a_d, *b_d, *c_d; // device pointers

    // allocate in host memory // usually pagable allocation
    /*
    a_h = (int *) malloc(N*sizeof(int));
    b_h = (int *) malloc(N*sizeof(int));
    c_h = (int *) malloc(N*sizeof(int));
    */
    cudaMallocHost(&a_h, N*sizeof(int));
    cudaMallocHost(&b_h, N*sizeof(int));
    cudaMallocHost(&c_h, N*sizeof(int));

    // initialize a_h and b_h
    initVector(a_h, b_h, N);

    // allocate in device
    cudaMalloc(&a_d, N*sizeof(int));
    cudaMalloc(&b_d, N*sizeof(int));
    cudaMalloc(&c_d, N*sizeof(int));

    // Host to Device
    cudaMemcpy(a_d, a_h, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, N*sizeof(int), cudaMemcpyHostToDevice);


    //<<< Dg, Db, Ns, S >>> blockDim()

    int blockSize = 256; // Dg, number of threads per block
    int numBlocks = (N+blockSize -1)/blockSize; // Db, number of blocks
    vectorAdd<<<numBlocks,blockSize>>>(a_d, b_d, c_d, N);

    cudaDeviceSynchronize();

    // Device to Host
    cudaMemcpy(c_h, c_d, N*sizeof(int), cudaMemcpyDeviceToHost);

    check(a_h, b_h, c_h, N);

    // deallocate host memory
    cudaFreeHost(a_h);
    cudaFreeHost(b_h);
    cudaFreeHost(c_h);

    // deallocate device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}


int main(int argc, char* argv[])
{
    //hello();

    vectorAddNative();

    return 0;
}

