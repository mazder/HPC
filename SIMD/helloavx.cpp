#include<iostream>
#include<immintrin.h>
#include<omp.h>

// objdump -d helloavx >> helloavx.txt

void subArrayAVX256(){

    float *datau = (float*)malloc(sizeof(float)*8);
    float *dataA,*dataB,*dataC; // for alias pointer

    // unaligned data
    float a[] ={2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    float b[] ={1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};



    __m256 aRegister = _mm256_loadu_ps(a);
    __m256 bRegister = _mm256_loadu_ps(b);
    __m256 cRegister = _mm256_add_ps(aRegister, bRegister);

    // using alias
    dataA = (float*)&aRegister;
    dataB = (float*)&bRegister;
    dataC = (float*)&cRegister;

    for(int i=0; i<8; i++){
        std::cout<<dataA[i]<<" "<<dataB[i]<<" "<<cRegister[i]<<std::endl;
    }
    // using store
    _mm256_storeu_ps(datau, cRegister);
    for(int i=0; i<8; i++){
        std::cout<<datau[i]<<std::endl;
    }


    // using AVX variables
    // index                  7    6    5    4     3    2     1     0
    __m256 A = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
    __m256 B = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);

    // _mm<bit width> <function name> <data type> (arguments)
    __m256 result = _mm256_add_ps(A,B);

    dataA = (float*)&A;
    dataB = (float*)&B;
    dataC = (float*)&result;
    for(int i=0; i<8; i++){
        std::cout<<dataA[i]<<" "<<dataB[i]<<" "<<dataC[i]<<std::endl;
    }

    _mm256_storeu_ps(datau, result);
    for(int i=0; i<8; i++){
        std::cout<<datau[i]<<std::endl;
    }

}

int main(){

    subArrayAVX256();

    return 0;
}
