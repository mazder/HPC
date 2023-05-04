/*
    To compile g++  -mavx simdprogram.cpp -o simdprogram -fopenmp -O3
               g++  -mavx512f simdprogram.cpp -o simdprogram -fopenmp -O3

    To run ./simdprogram N // N is a positive number

*/

#include<iostream>
#include<immintrin.h>
#include<omp.h>

// This is global variable
long N=0; // N = 2^n+k

// AVX2
float simdAddAVX512(float* a){
    // Initialize how many numbers of float in a AVX512 register
    float result=0.0;
    constexpr auto N_FLOAT_IN_AVX512_REG = 16u;
    //number of vectorizable sampels
    const auto sample_size = (N/N_FLOAT_IN_AVX512_REG)*N_FLOAT_IN_AVX512_REG;
    float *sumf16 = (float *)malloc(sizeof(float)*N_FLOAT_IN_AVX512_REG);
    __m512 intermediate_sum = _mm512_setzero_ps();
    auto i=0u;

    for(;i<sample_size; i+=N_FLOAT_IN_AVX512_REG){
        //std::cout<<"AVX512 "<<i<<std::endl;
        auto aRegister = _mm512_loadu_ps(a+i);
        intermediate_sum=_mm512_add_ps(intermediate_sum,aRegister);
    }

    _mm512_storeu_ps(sumf16,intermediate_sum);

    for(auto j=0u;j<N_FLOAT_IN_AVX512_REG; j++){
        //std::cout<<sumf16[j]<<std::endl;
        result+=sumf16[j];
    }
    free(sumf16);
    // do the rest
    //std::cout<<result<<std::endl;
    for(;i<N; i++){
        //std::cout<<i<<std::endl;
        result+=a[i];
    }
    //printf("%f\n",result);
    return result;
}


// AVX2
float simdAddAVX256(float* a){
    // Initialize how many numbers of float in a AVX register
    auto result=0.f;
    constexpr auto N_FLOAT_IN_AVX_REG = 8u;
    //number of vectorizable sampels
    const auto sample_size = (N/N_FLOAT_IN_AVX_REG)*N_FLOAT_IN_AVX_REG;
    float *sumf8 = (float *)malloc(sizeof(float)*N_FLOAT_IN_AVX_REG);
    __m256 intermediate_sum = _mm256_setzero_ps();
    auto i=0u;

    for(;i<sample_size; i+=N_FLOAT_IN_AVX_REG){
        //std::cout<<"AVX256 "<<i<<std::endl;
        auto aRegister = _mm256_loadu_ps(a+i);
        intermediate_sum=_mm256_add_ps(intermediate_sum,aRegister);
    }

    _mm256_storeu_ps(sumf8,intermediate_sum);

    for(auto j=0u;j<N_FLOAT_IN_AVX_REG; j++){
        //std::cout<<sumf8[j]<<std::endl;
        result+=sumf8[j];
    }
    free(sumf8);
    // do the rest
    //std::cout<<result<<std::endl;
    for(;i<N; i++){
        //std::cout<<i<<std::endl;
        result+=a[i];
    }
    //printf("%f\n",result);
    return result;
}

float scalarSum(float* a){
    auto sum=0.f;
    auto i=0u;
    for(; i<N; i++){
        sum+=a[i];
    }
    printf("%f\n",sum);
    return sum;
}

void initArray(float* a){
    auto i=0u;
    for(; i<N; i++){
        a[i]=1.1f;
    }
}

void print(float* a){
    auto i=0u;
    for(; i<N; i++){
        std::cout<<a[i]<<std::endl;
    }
}


void testArraySum(){
    float *a = (float *) malloc(sizeof(float)*N);
    float sum;
    double scalarTime,AVX256Time,AVX512fTime;
    initArray(a);

    scalarTime=omp_get_wtime();
    sum = scalarSum(a);
    scalarTime = omp_get_wtime()-scalarTime;
    std::cout<<"Scalar sum: "<<sum<<" time: "<<scalarTime<<"s"<<std::endl;


    AVX256Time=omp_get_wtime();
    sum = simdAddAVX256(a);
    AVX256Time = omp_get_wtime()-AVX256Time;
    std::cout<<"Vectorized simd AVX256 sum: "<<sum<<" time: "<<AVX256Time<<"s"<<std::endl;

    std::cout<<"Speed up: "<<(double)scalarTime/AVX256Time<<"s"<<std::endl;


    AVX512fTime=omp_get_wtime();
    sum = simdAddAVX512(a);
    AVX512fTime = omp_get_wtime()-AVX512fTime;
    std::cout<<"Vectorized simd AVX512 sum: "<<sum<<" time: "<<AVX512fTime<<"s"<<std::endl;

    std::cout<<"Speed up: "<<(double)scalarTime/AVX512fTime<<"s"<<std::endl;

    free(a);

}

int main(int argc, char* argv[]){

    N=atoi(argv[1])*32+3;
    std::cout<<"N= "<<N<<std::endl;
    testArraySum();


    return 0;
}
