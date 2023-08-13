/* How to compile this file
   g++  main.cpp -o main -fopenmp -O3

   How to Run
   ./main
*/

//  main.cpp
//  Create a data array of size N, N is an integer, initialize with interger valuses range from 1 to 100 inclusive.
//        Create a bargraph for the following distribution, size of range = 20;
//        1-20 | 21-40 | 41-60 | 61-80 | 81-100
//
//  Created by Md Mazder Rahman on 2023-07-25.
//  Copyright (c) 2023 Md Mazder Rahman. All rights reserved.

// Header files
#include<iostream>
#include<bits/stdc++.h>
#include<omp.h>

// Header file for time
#include<chrono>

using namespace std;
using namespace std::chrono;

// initializaiotn------------------------------
// data_array : pointer of data array
// N : size of array
template<typename T>
void initialize_array(T* data_array, T N){
        for(int i=1; i<=N; i++){
            data_array[i]=i;
        }
}

// initializaiotn------------------------------
// data_array : pointer of data array
// N : size of array
// n_threads : number of threads to be used
template<typename T>
void init_array_auto_parallel(T* data_array, T N, T n_threads, std::function<int()>F){
    // set number of threads to be run in parallel region
    omp_set_num_threads(n_threads);
    #pragma omp parallel // parallel region start from here
    {   // automatic loop parallel
        #pragma omp for
        for(int i=0; i<N; i++){
            data_array[i]=F();
        }
    }// parallel region end
}


// Generate Bargraph---------using multithreads----------------
// data_array : pointer of data array
// N : size of array
// n_distribution : number of distribution
// dist_array : pointer of distribution array
// size_range : size of data rangne for each distribution
template<typename T>
void genrate_bargraph_auto_parallel(T* data_array, T* dist_array, T N, T size_range, T n_distribution){

    // variable for number of threads
    int n_threads = n_distribution;

    // set number of threads to be run in parallel region
    omp_set_num_threads(n_threads);

    // variable as private to each thread
    int local_array[n_distribution];

    #pragma omp parallel default(shared) private(local_array)
    {
        // initialize local array
        for(int k=0; k<n_distribution; k++){
            local_array[k]=0;
        }

        #pragma omp for
        for (int i=0 ; i<N; i++) {
            int idx=data_array[i]/size_range;
            local_array[idx]=local_array[idx]+1;
        }

        // aggregate local array to distribution array, each thread participate to execute the following code
        for(int k=0; k<n_distribution; k++){
            #pragma omp atomic
            dist_array[k]+=local_array[k];
        }
    }
}

// Generate Bargraph---------sequential that is single threaded ----------------
// data_array : pointer of data array
// N : size of array
// n_distribution : number of distribution
// dist_array : pointer of distribution array
// size_range : size of data rangne for each distribution
template<typename T>
void genrate_bargraph_sequential(T* data_array, T* dist_array, T N, T size_range, T n_distribution){
    for(int i=0; i<N; i++){
        int index=data_array[i]/size_range;
        dist_array[index]=dist_array[index]+1;
    }
}

int main(int argc, char* argv[])
{

    // Data size
    int N=9999999;
    //int N=100;
    // Distribution
    // Allocate data array
    int* data_array = new int[N];

    //number of distributions
    int n_distribution=5;
    int size_of_range=20;

    // Allocate distribution array of size 5 for multithreaded
    int* dist_array_mt = new int[n_distribution];

    // Allocate distribution array of size 5 for single threaded
    int* dist_array_st = new int[n_distribution];

    // Lamda function return a random value in range 1 to 100
    auto init_array = []()->int{
        return (1 + rand()%100);
    };

    auto init_zero = []()->int{
        return 0;
    };

    // call initialize data array
    //initialize_array<int>(data_array, 20);

    // call initialize data array
    init_array_auto_parallel<int>(data_array, N, n_distribution, init_array);

    // call initialize dist_array_mt array
    init_array_auto_parallel<int>(dist_array_mt, n_distribution, n_distribution, init_zero);


    // get time
    auto start_time = high_resolution_clock::now();

    // call calculate dist_array_mt array multithreaded
    genrate_bargraph_auto_parallel<int>(data_array, dist_array_mt, N, size_of_range, n_distribution);

    // get time
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end_time - start_time);
    std::cout<<"Parallel execution Time:  "<<duration.count()<<"(ns)"<<std::endl;


    cout<<"Parallel Results"<<endl;
    for(int i=0; i<n_distribution; i++){
        cout<<dist_array_mt[i]<<endl;
    }

    // call initialize dist_array_mt array
    init_array_auto_parallel<int>(dist_array_st, n_distribution, n_distribution, init_zero);

    // get time
    start_time = high_resolution_clock::now();

    // call calculate dist_array array single threaded
    genrate_bargraph_sequential<int>(data_array, dist_array_st, N, size_of_range, n_distribution);

    // get time
    end_time = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end_time - start_time);
    std::cout<<"Sequential execution Time: "<<duration.count()<<"(ns)"<<std::endl;


    cout<<"Sequential Results"<<endl;
    for(int i=0; i<n_distribution; i++){
        cout<<dist_array_st[i]<<endl;
    }

    for(int i=0; i<n_distribution; i++){
        if(dist_array_st[i]!=dist_array_mt[i]){
            cout<<"Failed"<<endl;
            break;
        }
    }

    cout<<"Success!"<<endl;

    // free memory
    delete[] data_array;
    delete[] dist_array_mt;
    delete[] dist_array_st;

    return 0;
}
