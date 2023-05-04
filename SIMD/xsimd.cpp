/*
C++ wrappers for SIMD intrinsics
*/
#include<iostream>
#include<omp.h>
//#include<immintrin.h>
#include "/home/rahman/xsimd/include/xsimd/xsimd.hpp"


namespace xs = xsimd;
using Vector = std::vector<double, xs::aligned_allocator<double>>;

void printv(Vector& a){
    for(const auto e:a){
        std::cout<<e<<std::endl;
    }
}

void print(int* a, int N){
    for(int i=0; i<N; i++){
        std::cout<<a[i]<<std::endl;
    }
}

Vector xsimdAlignedVectorMean(Vector& a, Vector& b){
    Vector result(a.size());

    std::size_t aSize = a.size();
    constexpr std::size_t xsimd_size = xs::simd_type<double>::size;
    //std::cout<<"xsimd_size:"<<xsimd_size<<std::endl;

    std::size_t xsimd_nsample = (aSize/xsimd_size) * xsimd_size;
    //std::size_t xsimd_nsample = aSize -(aSize%xsimd_size);

    auto i = 0u;
    for(; i<xsimd_nsample; i+=xsimd_size){
    //    std::cout<<"Index:: "<<i<<std::endl;
    /*
        auto aRegister = xs::load_aligned(a.data()+i);
        auto bRegister = xs::load_aligned(b.data()+i);
        auto interRegister  = (aRegister + bRegister)/2;
        interRegister.store_aligned(result.data()+i);
     */

        auto aRegister = xs::load_aligned(&a[i]);
        auto bRegister = xs::load_aligned(&b[i]);
        auto interRegister  = (aRegister + bRegister)/2;
        interRegister.store_aligned(&result[i]);
    }

    for(; i<result.size(); i++){
    //    std::cout<<"lastIndex:: "<<i<<std::endl;
        result[i]=(a[i]+b[i])/2;
    }

    return result;
}

void xsimdAlignedVector(){
    Vector a(17, 1.0);
    auto result = xsimdAlignedVectorMean(a,a);
    printv(result);

}

/*

void averageXSIMD256(){

    xs::batch<double, xs::avx2> a = {1.5, 2.5, 3.5, 4.5}; // 4 double
    xs::batch<double, xs::avx2> b = {2.5, 3.5, 4.5, 5.5};
    auto mean = (a + b) / 2;
    std::cout << a<<"\n"<<b<<"\n"<<mean << std::endl;

}

void averageXSIMD512(){

    xs::batch<double, xs::avx512f> a = {1.5, 2.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5}; // 8 double
    xs::batch<double, xs::avx512f> b = {2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 5.5};
    auto mean = (a + b) / 2;
    std::cout << a<<"\n"<<b<<"\n"<<mean << std::endl;
}
*/

void dataTypeAlinedData256(){
    int N = 17;
    int *vA = (int*) aligned_alloc(sizeof(__m256i),sizeof(int)*N);
    int *vB = (int*) aligned_alloc(sizeof(__m256i),sizeof(int)*N);
    int *vC = (int*) aligned_alloc(sizeof(__m256i),sizeof(int)*N);

    // initialize
    for(int i=0; i<N; i++){
        vA[i]=1;
        vB[i]=1;
    }

    int avx2_size = 8;
    int vectorizableSample = (N/avx2_size)*avx2_size;
    int i=0;
    for( ; i<vectorizableSample; i+=avx2_size){
        __m256i aRegister = _mm256_load_si256((__m256i *)&vA[i]);
        __m256i bRegister = _mm256_load_si256((__m256i *)&vB[i]);
        __m256i cRegister = _mm256_add_epi32(aRegister,bRegister);

        _mm256_store_si256((__m256i *)&vC[i],cRegister);
        std::cout<<i<<std::endl;
    }
    // do the rest if any
    for(; i<N; i++){
        vC[i]=vA[i] ; //+vB[i];
    }
    print(vC, N);
}

void dataTypeAlinedData512(){
    int N = 17;
    int *vA = (int*) aligned_alloc(sizeof(__m512i),sizeof(int)*N);
    int *vB = (int*) aligned_alloc(sizeof(__m512i),sizeof(int)*N);
    int *vC = (int*) aligned_alloc(sizeof(__m512i),sizeof(int)*N);

    // initialize
    for(int i=0; i<N; i++){
        vA[i]=1;
        vB[i]=1;
    }

    int avx512_size = 16;
    int vectorizableSample = (N/avx512_size)*avx512_size;
    int i=0;
    for( ; i<vectorizableSample; i+=avx512_size){
        __m512i aRegister = _mm512_load_si512((__m512i *)&vA[i]);
        __m512i bRegister = _mm512_load_si512((__m512i *)&vB[i]);
        __m512i cRegister = _mm512_add_epi32(aRegister,bRegister);

        _mm512_store_si512((__m512i *)&vC[i],cRegister);
        std::cout<<i<<std::endl;
    }
    // do the rest if any
    for(; i<N; i++){
        vC[i]=vA[i] ; //+vB[i];
    }
    print(vC, N);
}



int main(int argc, char* argv[])
{
    //averageXSIMD256();
    //averageXSIMD512();

    //xsimdAlignedVector();

    dataTypeAlinedData256();
    dataTypeAlinedData512();

    return 0;
}
