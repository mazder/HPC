#include<iostream>
#include<omp.h>
#include<cassert>
#include<vector>
#include<functional>
#include<algorithm>


#define TILE_WIDTH 32

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
//template, vector, lamda

template<typename T>
void matrixMultiplyHost(T *A, T *B, T *C, int M, int W, int N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            C[m*N+n]=0;
            for(int w=0; w<W; w++){
                C[m*N+n]+=A[m*W+w]*B[w*N+n];
            }
        }
    }
}

template<typename T>
void transposeMatrix(T *A, T *B, T *C, int M, int W, int N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            C[m*N+n]=A[n*M+m];
        }
    }
}


template<typename T>
__global__ void matrixMultiplyDeviceNative(T *A, T* B, T* C, int M, int W, int N){

    int row = (blockIdx.y*blockDim.y)+threadIdx.y;
    int col = (blockIdx.x*blockDim.x)+threadIdx.x;

    // set boundary constraints
    if(row<M && col<N){
        int c=0;
        for(int w=0; w<W; w++){
            c+=A[row*W+w]*B[w*N+col];
        }
        C[row*N+col]=c;
    }
}


template<typename T>
__global__ void transeposeMatrixNative(T *A, T* B, T* C, int M, int W, int N){

    int row = (blockIdx.y*blockDim.y)+threadIdx.y;
    int col = (blockIdx.x*blockDim.x)+threadIdx.x;

    // set boundary constraints
    if(row<M && col<W){

        C[row*W+col]=A[col*M+row];
    }
}



template<typename T>
__device__ void printHostMatrix(T* A, T M, T N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            printf("%d ", A[m*N+n]);
        }
        printf("\n");
    }
}

template<typename T>
__device__ void printShared(T M[2][2]){

    for(int m=0; m<2; m++){
        for(int n=0; n<2; n++){
            printf("%d ",M[m][n]);
        }
        printf("\n");
    }
}


template<typename T>
__global__ void matrixMultiplyDeviceShared(T *A, T* B, T* C, int M, int W, int N){

    T row = (blockIdx.y*blockDim.y)+threadIdx.y;
    T col = (blockIdx.x*blockDim.x)+threadIdx.x;

    //printf("%d %d %d %d %d %d\n", row, col, blockIdx.y, blockIdx.x, threadIdx.x, threadIdx.y);

    // share memory
    __shared__ T S_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ T S_B[TILE_WIDTH][TILE_WIDTH];

    S_A[threadIdx.y][threadIdx.x] = 0;
    S_B[threadIdx.y][threadIdx.x] = 0;

    T c=0;
    /*
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0){

        for(int m=0; m<TILE_WIDTH; m++){
            for(int n=0; n<TILE_WIDTH; n++){
                printf("%d ",S_A[m][n]);
            }
            printf("\n");
        }

        printShared(S_A);
        printHostMatrix(C, M, N);
    }*/

//    if(blockIdx.x==0 && blockIdx.y==0){
        /**************************************************************/
        for(int i=0; i<(W+TILE_WIDTH-1)/TILE_WIDTH; i++){
            //printf("i=%d\n", i);

            if(row<M && (threadIdx.x + (i*TILE_WIDTH))<W){

                S_A[threadIdx.y][threadIdx.x]=A[row*W+(threadIdx.x+(i*TILE_WIDTH))];

            }
            else{
                S_A[threadIdx.y][threadIdx.x]=0;
            }

            if(col<N && (threadIdx.y+(i*TILE_WIDTH))<W){
                S_B[threadIdx.y][threadIdx.x] = B[(threadIdx.y+(i*TILE_WIDTH)*N+col)];
            }
            else{
                S_B[threadIdx.y][threadIdx.x]=0;
            }

            __syncthreads();

            //printShared(S_A);
            //printShared(S_B);

            for(int j=0; j<TILE_WIDTH; j++){
                c+=S_A[threadIdx.y][j]*S_B[j][threadIdx.x];
            }
        }

        if(row<M && col<N){
            C[row*N+col]=c;
        }
        /**************************************************************/
 //   }
}

template<typename T>
void printMatrix(T* A, T M, T N){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            std::cout<<A[m*N+n]<<' ';
        }
        std::cout<<"\n";
    }
}

// matrix multiplication
template<typename T>
void MM_shared(){

    // Number of elements
    int M = 4;  // 256
    int W = 4; // 1024
    int N = 4; // 4096

    std::cout<<M<<" "<<W<<" "<<N<<std::endl;

    std::vector<T> a_h(M*W);
    std::vector<T> b_h(W*N);
    std::vector<T> c_h(M*N,0);
    std::vector<T> c(M*N,0);

    T *a_d, *b_d, *c_d; // device pointers

    generate(a_h.begin(), a_h.end(), []()->int{ return rand()%10;});


//    matrixMultiplyHost<T>(a_h.data(), b_h.data(), c_h.data(), M, W, N);

    transposeMatrix<T>(a_h.data(), b_h.data(), c_h.data(), M, W, N);


    std::cout<<"--------a_h---------------"<<std::endl;
    printMatrix<T>(a_h.data(), M, W);
    std::cout<<"--------b_h---------------"<<std::endl;
    printMatrix<T>(b_h.data(), W, N);
    std::cout<<"--------c_h---------------"<<std::endl;
    printMatrix<T>(c_h.data(), M, N);

    // allocate into device
    cudaMalloc(&a_d, M*W*sizeof(int));
    cudaMalloc(&b_d, W*N*sizeof(int));
    cudaMalloc(&c_d, M*N*sizeof(int));

    // transfer H
    cudaMemcpy(a_d, a_h.data(), M*W*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h.data(), W*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 dim_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dim_grid((N+TILE_WIDTH-1)/TILE_WIDTH, (M+TILE_WIDTH-1)/TILE_WIDTH, 1);

    //matrixMultiplyDeviceNative<T><<<dim_grid, dim_block>>>(a_d, b_d, c_d, M, W, N);

    //matrixMultiplyDeviceShared<T><<<dim_grid, dim_block>>>(a_d, b_d, c_d, M, W, N);


    transeposeMatrixNative<T><<<dim_grid, dim_block>>>(a_d, b_d, c_d, M, W, N);


    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), c_d, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    std::cout<<"--------c---------------"<<std::endl;
    printMatrix<T>(c.data(), M, N);

    for(int i=0; i<(M*N); i++){
        if(c_h[i]!=c[i]){
            std::cout<<"Failed"<<c_h[i]<<" != "<<c[i]<<std::endl;
            exit(1);
        }
    }
    std::cout<<"Successfully Finished"<<std::endl;

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

}


int main(int argc, char* argv[])
{
    MM_shared<int>();

    return 0;
}

