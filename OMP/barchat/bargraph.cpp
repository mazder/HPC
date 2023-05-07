#include<iostream>
#include<bits/stdc++.h>
#include<omp.h>
#include<pthread.h>
#include<utility>
#include<iomanip>
#include<fstream>

using namespace std;

// initializaiotn
// v : pointer of data array
// N : size of array
// nThreads : number of threads to be used
template<typename T>
void init_vector_auto_parallel(T* v, T N, T nThreads, std::function<int()>F){
    omp_set_num_threads(nThreads);
    int i;
    // automatic loop parallel
    #pragma omp parallel for
        for(i=0; i<N; i++){
            v[i]=F();
        }
}

template<typename T>
void init_vector_explicit_parallel(T* v, T N, T nThreads, std::function<int()>F){
    omp_set_num_threads(nThreads);
    int chunk = N/nThreads;
    int i, id;

    // explicitly parallelized and spcified distribution of data array v to threads
    #pragma omp parallel shared(v) private(i, id)
    {
        id = omp_get_thread_num();
        for(i=id*chunk; i<(id+1)*chunk; i++){
            v[i]=F();
        }
    }
}

template<typename T>
void genrate_bargraph_explicit_parallel(T* v, T* h, T N, T nThreads){
    int i, id;
    int chunk = N/nThreads;
    int vrange = 100/nThreads;
    omp_set_num_threads(nThreads);
    #pragma omp parallel shared(v,h) private(i,id)
    {
        id = omp_get_thread_num();
        for(i=id*chunk; i<(id+1)*chunk; i++){
            if( (id*vrange<v[i]) && (v[i]<(id+1)*vrange+1) ){
                #pragma omp critical
                {
                    h[id]=h[id]+1;
                }
            }
        }
    }
}

// Generate Bargraph
template<typename T>
void genrate_bargraph_auto_parallel(T* v, T* h, T N, T nThreads){
    int i, id;
    int chunk = N/nThreads;
    int vrange = 100/nThreads;
    omp_set_num_threads(nThreads);
    #pragma omp parallel shared(v) private(id)
    {
        id = omp_get_thread_num();
        #pragma omp parallel for
        for(i=0; i<N; i++){
            if( (id*vrange<v[i]) && (v[i]<(id+1)*vrange+1) ){
                #pragma omp critical
                {
                    h[id]=h[id]+1;
                }
            }
        }
    }
}

int main(int argc, char* argv[]){
    /*
        Given a data array of N = 500 containing numbers 1<= n1 <=100
        Create a bargraph of data like the following distribution
        1-20 | 21-40 | 41-60 | 61-80 | 81-100
    */

    int N=500; // data size
    int nT=5;  // number of threads

    int* data = new int[N];
    int* h = new int[nT];

    auto init_v = []()->int{
        return (1 + rand()%100);
    };

    auto init_zero = []()->int{
        return 0;
    };
    // init data array
    //init_vector_explicit_parallel(data, N, nT, init_v);

    // read from file
    fstream dfile("data.txt");
    for(int i=0; i<500; i++){
        dfile>>data[i];
    }
    // init bargraph array
    init_vector_explicit_parallel(h, nT, nT, init_zero);

    genrate_bargraph_explicit_parallel(data, h, N, nT);

    for(int i=0; i<nT; i++){
        cout<<h[i]<<endl;
    }

    delete[] data;
    delete[] h;

    return 0;
}

