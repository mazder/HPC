__global__ void sum(int * A, int* B)

    __shared int sA[];
    int tid = threadIdx.x
    int i = threadIdx.x + blockIdx.x * blockDim.x
    sA[tid] = A[i]
    __syncthread();
    
    for(int s=0; i<blockDim.x; s*=2){
        if(tid%2==0){
            sA[tid]+=sA[tid+s]
        }
        __syncthreads();
    }

    if(tid==0){
        B[blockIdx.x]=sA[0]
    }





__global__ void mm(T* A, T* B, T* C, int M, int W, int N){
    int mRow=blockIdx.y * blockDim.y + threadIdx.y;
    int nCol=blockIdx.x * blockDim.x + threadIdx.x;
    if(mRow < M && nCol<N){
        T c=0.0;
        for(int w=0; w<W; w++){
            c+=A[mRow*W+w] * B[w*N+nCol];
        }
        C[mRow*N + n] = c;
    } 
}




__global__ void mm(T* A, T* B T* C, int M, int W, int N){

    int mRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T* sA[TW][TW];
    __shared__ T* sB[TW][TW];

    T* c=0.0;

    for(int i=0; i<(W-1)=colA/TW + 1; i++){

        if(mRow<M=rowA && (threadIdx.x + i*TW))<W=colA){ 
            sA[threadIdx.x][threadIdx.y]=A[mRow*W+(threadIdx.x + i*TW)];
        }
        else{
            sA[threadIdx.x][threadIdx.y]=0.0;
        }

        if(nCol<N=colB && (threadIdx.y + i*TW)<W= rowB){ 
            sB[threadIdx.x][threadIdx.y] = B[(threadIdx.x + i*TW)*N+nCol]
        }
        else{
            sB[threadIdx.x][threadIdx.y]=0.0;

        }
        __syncthreads();

        for(int j=0; j<TW; j++){
            c+=sA[threadIdx.y][i]*sB[j][threadIdx.x];
        }

    }
    if(rowM<M && colN<N){
        C[rowM*N+colN]=c;
    }

}

__global__ void MM()

    int rowM = threadIdx.y + blockIdx.y*blockDim.y;
    int colN = threadIdx.x + blockIdx.x*blockDim.x;

    sA[TW][TW]
    sB[TW][TW]
    sA[threadIdx.y][threadIdx.x]=0;
    sB[threadIdx.y][threadIdx.x]=0;
    for(int i =0; i< (colA = W-1)/TW + 1; i++){
        // copy A
        if(rowM<M && (threadIdx.x + i*TW)){
            sA[threadIdx.y][threadIdx.x] = A[rowM*W+((threadIdx.x + i*TW))]
        }
        // copy A
        if(colN<N && (threadIdx.y + i*TW)){
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + i*TW)*N+colN]
        }

        for(j=0; j<TW; j++){
            c+=sA[threadIdx.y][j]*sB[j][threadIdx.x]
        }

    }
