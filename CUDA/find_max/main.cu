#include<iostream>
#include<chrono>
#include<unistd.h>

#include "kernels.cuh"
#include "functions.hpp"


using namespace std::chrono; 

void init_array(int *arr, unsigned int N){
    for(unsigned int i=0; i<N; i++){
        arr[i]=i; //*int(rand());
    }
}

void print_array(int *arr, unsigned int N){
    for(unsigned int i=0; i<10; i++){
        std::cout<<arr[i]<<std::endl;
    }
}

void find_max(){
    // size of data
    unsigned int N = 1024 * 1024 * 20;
    int max = -1;
    int *arr = (int *)malloc(N*sizeof(int));
    
    // device
    int *arr_d;
    int *mx_d;


    // gpu timing variables
    float gpu_elasped_time;
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);



    //initialize arr
    init_array(arr, N);
    //print_array(arr, N);
    //find in host
    auto start_time = high_resolution_clock::now();
    for(int i=0; i<1000; i++){
        max_host(arr, N, &max);
    }
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end_time - start_time);
    std::cout<<"Host: Data size: "<<N<<" MAX value: "<<max<<"    Time: "<<duration.count()<<"(ns)"<<std::endl;
    max=-1;

    int block_size = 256;
    int num_blocks = 256; //(N+block_size-1)/block_size;
    std::cout<<num_blocks<<" "<<block_size<<std::endl;

    // allocate device memory
    //cudaMalloc((void**)arr_d, N*sizeof(int));
    cudaMalloc(&arr_d, N*sizeof(int));
    cudaMalloc(&mx_d, sizeof(int));

    start_time = high_resolution_clock::now();
    cudaEventRecord(gpu_start, 0);
    cudaMemcpy(arr_d, arr, N*sizeof(int), cudaMemcpyHostToDevice);
    for(int i=0; i<1000; i++){
        max_device<<<num_blocks, block_size>>>(arr_d, N, mx_d);
    }
    cudaMemcpy(&max, mx_d, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elasped_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

    end_time = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end_time - start_time);
    std::cout<<"Device: Data size: "<<N<<" MAX value: "<<max<<"  Time: "<<duration.count()<<"(ns), "<<gpu_elasped_time<<"(ms)"<<std::endl;

    // free memory
    free(arr);
    cudaFree(arr_d);
    cudaFree(mx_d);
}

int main(int argc, char* argv[])
{
    srand(0);
    
    find_max();
    return 0;
}