/*
    To compile g++  -mavx navectoradd.cpp -o navectoradd -fopenmp -O3
               g++  -mavx512f navectoradd.cpp -o navectoradd -fopenmp -O3

    To run ./simdprogram N // X = 16^N +1

*/

#include<iostream>
#include<immintrin.h>
#include<omp.h>
#include<vector>
#include<math.h>

using Vector = std::vector<float>;

// This is global variable
unsigned N=0; // X = 16^N +1

void print(Vector& a){
    auto i=0u;
    for(; i<N; i++){
        std::cout<<a[i]<<std::endl;
    }
}

void printv(Vector& a){
    for(const auto e:a){
        std::cout<<e<<std::endl;
    }
}

void printsum(Vector& a){
    auto sum = 0.f;
    for(const auto e:a){
        sum+=e;
    }
    std::cout<<"SUM :"<<sum<<std::endl;
}

Vector scalarAdd(Vector& a, Vector& b){
    Vector vC(a.size());
    auto i=0u;

    for(;i<N; i++){
        vC[i]=a[i]+b[i];
    }
    return vC;
}
Vector AddAVX256(const Vector& a, const Vector& b){
    // allocate result vector of size a vector, same size of b vector
    Vector result(a.size());

    // AVX2 has 256 bits register, a floating point number is 32 bits
    // Therefore 8 floating point numbers can be loaded in a register

    // Initialize how many numbers of float in a AVX register
    constexpr auto N_FLOAT_IN_AVX_REG = 8u;

    // Calculate how many vectorizable samples of a can be process
    const auto nVectorizableSample =(a.size()/N_FLOAT_IN_AVX_REG)*N_FLOAT_IN_AVX_REG;

    auto i=0u;

    for(; i<nVectorizableSample; i+=N_FLOAT_IN_AVX_REG){
        // load unaligned data from vector a and b into registers, ps - single precision 32bits float
        auto aRegister = _mm256_loadu_ps(a.data()+i);
        auto bRegister = _mm256_loadu_ps(b.data()+i);

        // add to a register
        auto intermediateSum = _mm256_add_ps(aRegister, bRegister);

        // copy into result vector

        _mm256_storeu_ps(result.data()+i, intermediateSum);
        // std::cout<<"i="<<i<<std::endl;
    }
    // Handle samples that are not vectorized
    for(; i<result.size(); i++){
        //result[i]=a[i]+b[i];
        result[i]=a[i]; // intentionally not add to verify not processed
    }

    return result;
}

Vector AddAVX512(const Vector& a, const Vector& b){

    // allocate result vector of size a vector, same size of b vector
    Vector result(a.size());

    // AVX2 has 512 bits register, a floating point number is 32 bits
    // Therefore 16 floating point numbers can be loaded in a register

    // Initialize how many numbers of float in a AVX register
    constexpr auto N_FLOAT_IN_AVX_REG = 16u;

    // Calculate how many vectorizable samples of a can be process
    const auto nVectorizableSample =(a.size()/N_FLOAT_IN_AVX_REG)*N_FLOAT_IN_AVX_REG;

    auto i=0u;

    for(; i<nVectorizableSample; i+=N_FLOAT_IN_AVX_REG){
        // load unaligned data from vector a and b into registers, ps - single precision 32bits float
        auto aRegister = _mm512_loadu_ps(a.data()+i);
        auto bRegister = _mm512_loadu_ps(b.data()+i);
        // add to a register
        auto intermediateSum = _mm512_add_ps(aRegister, bRegister);
        // copy into result vector
        _mm512_storeu_ps(result.data()+i, intermediateSum);
        //std::cout<<"i="<<i<<std::endl;
    }

    // Handle samples that are not vectorized
    for(; i<result.size(); i++){
        //result[i]=a[i]+b[i];
        result[i]=a[i]; // intentionally not add to verify not processed
    }

    return result;
}

void testVectorSum(){

    Vector vA(N, 1.f);
    double scalarTime,AVX256Time,AVX512fTime;

    scalarTime=omp_get_wtime();
    auto resultScalar = scalarAdd(vA,vA);
    scalarTime = omp_get_wtime()-scalarTime;
    std::cout<<"Scalar Add: "<<" time: "<<scalarTime<<"s"<<std::endl;
    //printv(resultScalar);
    printsum(resultScalar);
    std::cout<<"======================================"<<std::endl;

    AVX256Time=omp_get_wtime();
    auto resultAVX256 = AddAVX256(vA,vA);
    AVX256Time = omp_get_wtime()-AVX256Time;
    std::cout<<"Vectorized simd AVX256: "<<" time: "<<AVX256Time<<"s"<<std::endl;
    std::cout<<"Speed up: "<<(double)scalarTime/AVX256Time<<"X"<<std::endl;
    //printv(resultAVX256);
    printsum(resultAVX256);
    std::cout<<"======================================"<<std::endl;

    AVX512fTime=omp_get_wtime();
    auto resultAVX512 = AddAVX512(vA,vA);
    AVX512fTime = omp_get_wtime()-AVX512fTime;
    std::cout<<"Vectorized simd AVX512: "<<" time: "<<AVX512fTime<<"s"<<std::endl;
    std::cout<<"Speed up: "<<(double)scalarTime/AVX512fTime<<"X"<<std::endl;
    //printv(resultAVX512);
    printsum(resultAVX512);
    std::cout<<"======================================"<<std::endl;

}

int main(int argc, char* argv[]){

    if(argc<2){
        std::cout<<"n args "<<argc<<std::endl;
        std::cout<<"Run ./simdprogram N // Where X = 16^N +1 "<<std::endl;
        exit(0);
    }
    N=pow(16,atoi(argv[1]))+1;

    std::cout<<"N= "<<N<<std::endl;
    testVectorSum();


    return 0;
}
