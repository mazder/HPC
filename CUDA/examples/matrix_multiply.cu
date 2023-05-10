#include<iostream>
#include<omp.h>
#include<cassert>


#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32

// A[M, N]
// B[N, K]
// C[M, K]
/*

-------------->Y direction
- | | . | |
- | | . | |
- ----------
- | | . | |
- | | . | |
v

X  direction

*/




void matrixMultiplyHost(int *A, int* B, int* C, int M, int W, int N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            /*
            int c=0;
            for(int w=0; w<W; w++){
                c+=A[m*W+w]*B[w*N+n];
            }
            C[m*N+n]=c;
            */
            for(int w=0; w<W; w++){
                C[m*N+n]+=A[m*W+w]*B[w*N+n];
            }
        }
    }
}


__global__ void matrixMultiplyDeviceNative(int *A, int* B, int* C, int M, int W, int N){

    int row = (blockIdx.y*blockDim.y)+threadIdx.y;
    int col = (blockIdx.x*blockDim.x)+threadIdx.x;

    // set boundary constraints
    if(row<M && col<N){
        C[row*N+col]=0;
        for(int w=0; w<W; w++){
            C[row*N+col]+=A[row*W+w]*B[w*N+col];
        }
    }
}

// initialize matrices A, B, C for C[M,N] = A[M, W] * B[W, N]

void initMatrix(int *A, int* B, int* C, int* CC, int M, int W, int N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            for(int w=0; w<W; w++){
                A[m*W+w]=2;
                B[w*N+n]=2;
            }
            C[m*N+n]=0;
            CC[m*N+n]=0;
        }
    }
}


void printMatrix(int* A, int M, int N){
    for(int m=0; m<10; m++){
        for(int n=0; n<10; n++){
            std::cout<<A[m*N+n]<<' ';
        }
        std::cout<<"\n";
    }
}


// matrix multiplication
void MM(){

    // Number of elements
    int M = 1<<8;  // 256
    int W = 1<<10; // 1024
    int N = 1<<12; // 4096

    std::cout<<M<<" "<<W<<" "<<N<<std::endl;

    int *a_h, *b_h, *c_h, *c; // host pointers
    int *a_d, *b_d, *c_d; // device pointers

    // allocate in host memory // usually pagable allocation
    a_h = (int *) malloc(M*W*sizeof(int));
    b_h = (int *) malloc(W*N*sizeof(int));
    c_h = (int *) malloc(M*N*sizeof(int));
      c = (int *) malloc(M*N*sizeof(int));

    // initialize a_h and b_h
    initMatrix(a_h, b_h, c_h, c, M, W, N);

/*
    std::cout<<"-------------A-------------"<<std::endl;
    printMatrix(a_h, M, W);
    std::cout<<"-------------B-------------"<<std::endl;
    printMatrix(b_h, W, N);
    std::cout<<"-------------C-------------"<<std::endl;
    printMatrix(c_h, M, N);
*/
    matrixMultiplyHost(a_h, b_h, c_h, M, W, N);
    //std::cout<<"-------------C-------------"<<std::endl;
    //printMatrix(c_h, M, N);


    // allocate into device
    cudaMalloc(&a_d, M*W*sizeof(int));
    cudaMalloc(&b_d, W*N*sizeof(int));
    cudaMalloc(&c_d, M*N*sizeof(int));

    // transfer H
    cudaMemcpy(a_d, a_h, M*W*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, W*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);
    dim3 dim_grid((N+COL_TILE_WIDTH-1)/COL_TILE_WIDTH, (M+ROW_TILE_WIDTH-1)/ROW_TILE_WIDTH, 1);

    matrixMultiplyDeviceNative<<<dim_grid, dim_block>>>(a_d, b_d, c_d, M, W, N);

    cudaDeviceSynchronize();

    cudaMemcpy(c, c_d, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout<<"-------------C-------------"<<std::endl;
    //printMatrix(c_h, M, N);
    for(int i=0; i<(M*N); i++){
        if(c_h[i]!=c[i]){
            std::cout<<"Failed"<<std::endl;
            exit(1);
        }
    }
    std::cout<<"Successfully Finished"<<std::endl;

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);
    free(c);
}


int main(int argc, char* argv[])
{
    MM();

    return 0;
}

