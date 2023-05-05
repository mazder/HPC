#include<iostream>
#include<cstdlib>   // srand and rand
#include<math.h>
#include<functional>

#include<chrono>

typedef std::chrono::high_resolution_clock Clock;


// A -> M X W
// B -> W X N
// C -> M X N
// W = width

#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32

template <typename T>
void matrix_mul_cpu(T *A, T *B, T *C, int M, int W, int N ){
    for(int m=0; m<M; m++){
        for(int n=0; n<N; n++){
            T c=0;
            for(int w=0; w<W; w++)
                c+=A[m*W+w]*B[w*N+n];
            C[m*N+n] = c;
        }
    }
}
template <typename T>
__global__ void matrix_mul_shared(T *A, T *B, T *C, int M, int W, int N ){

    // shared memory
    __shared__ T sA[TILE_WIDTH][TILE_WIDTH]; // 16 X 16
    __shared__ T sB[TILE_WIDTH][TILE_WIDTH];

    int mRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;

    T c=0.0;

    for (int i=0; i<((W-1)/TILE_WIDTH+1); i++){
        // copy data
        if(mRow<M && (threadIdx.x + (i * TILE_WIDTH)) <W){ // M=rowA W = colA
            sA[threadIdx.x][threadIdx.x] = A[mRow*W + (threadIdx.x + (i * TILE_WIDTH))];
        }
        else{
            sA[threadIdx.x][threadIdx.x] = 0.0;
        }
        if(nCol<N && (threadIdx.y + (i * TILE_WIDTH)) <W){ // W = rowB N=colB
            sB[threadIdx.y][threadIdx.y] = B[(threadIdx.y + (i * TILE_WIDTH))*N + nCol];
        }
        else{
            sB[threadIdx.y][threadIdx.y] = 0.0;
        }
        __syncthreads();

        for(int j=0; j<TILE_WIDTH; j++){
            c+=sA[threadIdx.y][j]*sB[j][threadIdx.x];
        }
    }

    if(mRow<M && nCol<N){
        C[mRow*N+nCol]=c;
    }
}


template <typename T>
__global__ void matrix_mul_native(T *A, T *B, T *C, int M, int W, int N ){
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    // set boundary constraints
    if(m<M && n<N){
        T c=0;
        for(int w=0; w<W; w++)
            c+=A[m*W+w]*B[w*N+n];
        C[m*N+n] = c;
    }
}
template<typename T>
void init_matrix(T* A, int n, int m, std::function<T()> F ){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            A[i*m+j]=F();
        }
    }
}
template<typename T>
void print_matrix(T* A, int n, int m){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            std::cout<<A[i*m+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

template<typename T>
void print_matrix_2(T* A, int n, int m){
    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            std::cout<<A[i*m+j]<<" ";
        }
        std::cout<<std::endl;
    }
}

template<typename T>
bool check(T *A, T *B, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if(abs(A[i*N+j]- B[i*N+j]) > 0.00001){
                return false;
            }
        }
    }
    return true;
}

template<typename T>
void matrix_multiply_unified_memory(T *A, T *B, T *C, int N, int W, int M){

    T *C_h;
    // Unified memory
    cudaMallocManaged(&A, M*W*sizeof(T));
    cudaMallocManaged(&B, W*N*sizeof(T));
    cudaMallocManaged(&C, M*N*sizeof(T));

    C_h = (T*)malloc(sizeof(T)*N*M);

    auto init_1 = []() -> T {
        return 1.2f;
    };

    init_matrix<T>(A, M, W, init_1);
    init_matrix<T>(B, W, N, init_1);

    auto cpu_start = Clock::now();
    matrix_mul_cpu<T>(A,B,C_h, M, W, N);
    auto cpu_end = Clock::now();
    std::cout << "CPU Time: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count() << " nanoseconds.\n";

    dim3 dim_Grid(N/COL_TILE_WIDTH, M/ROW_TILE_WIDTH, 1);
    dim3 dim_Block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

    auto gpu_start = Clock::now();
    matrix_mul_native<T><<<dim_Grid,dim_Block>>>(A, B, C, M, W, N);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();
    std::cout << "GPU Time: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count() << " nanoseconds.\n";

    std::cout << "X: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()/std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count() << std::endl;
    /*
    std::cout<<"A>>"<<std::endl;
    print_matrix_2<T>(A, M, W);
    std::cout<<"B>>"<<std::endl;
    print_matrix_2<T>(B, W, N);
    std::cout<<"C_h>>"<<std::endl;
    print_matrix_2<T>(C_h, M, N);
    std::cout<<"C>>"<<std::endl;
    print_matrix_2<T>(C, M, N);
    */

    if(check<T>(C, C_h, M, N)) std::cout<<"Pass"<<std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(C_h);
}


template<typename T>
void matrix_multiply_non_unified_memory(T *A, T *B, T *C, int N, int W, int M){

    T *C_h;
    // Unified memory
    cudaMallocManaged(&A, M*W*sizeof(T));
    cudaMallocManaged(&B, W*N*sizeof(T));
    cudaMallocManaged(&C, M*N*sizeof(T));

    C_h = (T*)malloc(sizeof(T)*N*M);




    auto init_1 = []() -> T {
        return 1.2f;
    };

    init_matrix<T>(A, M, W, init_1);
    init_matrix<T>(B, W, N, init_1);

    auto cpu_start = Clock::now();
    matrix_mul_cpu<T>(A,B,C_h, M, W, N);
    auto cpu_end = Clock::now();
    std::cout << "CPU Time: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count() << " nanoseconds.\n";

    dim3 dim_Grid(N/COL_TILE_WIDTH, M/ROW_TILE_WIDTH, 1);
    dim3 dim_Block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

    auto gpu_start = Clock::now();
    matrix_mul_native<T><<<dim_Grid,dim_Block>>>(A, B, C, M, W, N);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();
    std::cout << "GPU Time: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count() << " nanoseconds.\n";

    std::cout << "X: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()/std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count() << std::endl;

    if(check<T>(C, C_h, M, N)) std::cout<<"Pass"<<std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(C_h);
}



int  main(int argc, char* argv[]){

    int M = 1<<8;
    int W = 1<<10;
    int N = 1<<12;

    std::cout<<M<<" "<<W<<" "<<N<<std::endl;

    float *A=NULL, *B=NULL, *C=NULL;

    matrix_multiply_unified_memory<float>(A, B, C, M, W, N);

    return 0;
}
