#include<iostream>
#include<omp.h>

void printV(int *a, int *b, int *c, int N ){
    for(int i=0; i<10; i++){
        printf("%d %d %d\n",a[i],b[i],c[i]);
    }
}

__global__ void vectorAdd(int *A, int* B, int* C, int N){

    int tid = (blockIdx.x*blockDim.x)+threadIdx.x;

    if (tid<N){
        C[tid]=A[tid]+B[tid];
    }
}

__global__ void cuda_hello(){

    int tid = threadIdx.x;
    printf("Hello %d\n",tid);

}

void hello(){

    dim3 BLOCKSIZE(5,1,1);
    dim3 NUMBLOCKS(1,1,1);

    cuda_hello<<<NUMBLOCKS,BLOCKSIZE>>>();

    cudaDeviceSynchronize();

}

// vector add
void vectorAddNative(){

    // Number of elements
    int N = 17; //1<<16;

    //size_t bytes = N*sizeof(int);

    int *a_h, *b_h, *c_h; // host pointers
    int *a_d, *b_d, *c_d; // device pointers

    // allocate in host memory
    a_h = (int *) malloc(N*sizeof(int));
    b_h = (int *) malloc(N*sizeof(int));
    c_h = (int *) malloc(N*sizeof(int));

    // initialize a_h and b_h
    for(int i=0; i<N; i++){
        a_h[i]=i%2;
        b_h[i]=i%2;
        c_h[i]=i%2;
    }
    printV(a_h, b_h, c_h, N);
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

    printV(a_h, b_h, c_h, N);

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
    //hello();

    vectorAddNative();

    return 0;
}

