#include<iostream>
#include<immintrin.h>
#include<omp.h>

#define ALIGN 64

void vectorrized(float* A, float* B, int N){
    __m256 sumR = _mm256_set1_ps(0.0);
    __m256 sumI = _mm256_set1_ps(0.0);

    const __m256 conj = _mm256_set_ps(-1,1,-1,1,-1,1,-1,1);
    // Alias of the float32 arrays A and B, as arrays of 256-bit packed single vectors (_ m256)
    __m256 *a = (__m256 *)A; // also can be used load
    __m256 *b = (__m256 *)B;

    const int n = N/8; // 8 floats

    for(int j=0; j<n; j++){
        __m256 cr = _mm256_mul_ps(a[j],b[j]); // |ai*bi | ar*br ...
        __m256 bconj = _mm256_mul_ps(b[j],conj); // conjugate b
        __m256 bflip = _mm256_permute_ps(bconj, 0b10110001); //[2,3,0,1]) // to multiply ar*bi|
        __m256 ci = _mm256_mul_ps(a[j],bflip); // | ai*br| ar*bi ...

        sumR = _mm256_add_ps(sumR,cr);
        sumI = _mm256_add_ps(sumR,ci);

    }
}


void scalar(float* A, float* B, int N){

    float sumR=0.0;
    float sumI=0.0;

    for(int i=0; i<N; i+=2){
        // real and imaginary complonets of A
        float Ar = A[i];
        float Ai = A[i+1];
        // real and imaginary components of B*
        float Br = B[i];
        float Bi = -B[i+1];

        // multiply
        float Cr = Ar*Br - Ai*Bi;
        float Ci = Ar*Bi + Ai*Br;

        sumR+=Cr;
        sumI+=Ci;

    }
    std::cout<<sumR<<" "<<sumI<<std::endl;
}


int main(int argc, char* argv[]){

    // Number of elements in array
    const int N = 1<<10; //67,108,864

    double scalarTime;


    float *A = (float *) aligned_alloc(N*sizeof(float), ALIGN);
    float *B = (float *) aligned_alloc(N*sizeof(float), ALIGN);

    // initialize the array with random value between -1.0 to 1.0
    srand(0);
    std::cout<<N<<std::endl;
    for(int i=0; i<N; i++){
        float a = (2.0f *((float)rand())/RAND_MAX)-1.0f;
        float b = (2.0f *((float)rand())/RAND_MAX)-1.0f;
        A[i]=a;
        B[i]=b;
    }
    scalarTime = omp_get_wtime();
    // scalar sum ab*
    scalar(A,B,N);
    std::cout<<omp_get_wtime()-scalarTime<<"s"<<std::endl;





    return 0;
}
