#include<iostream>
#include<omp.h>


void falseSharing(){
// compile with no optimization
// g++  main.cpp -o main -fopenmp

    int *a = new int[100];
    omp_set_num_threads(4);
#pragma omp parallel
{
    int tid = omp_get_thread_num(), strid = 16, i;  // change the value 16 to 1 and compare time

    for(i=0; i<2000000000; i++){
        a[tid*strid]++;
    }
    std::cout<<tid<<" "<<omp_get_num_threads()<<" "<<i<<std::endl;
    
    delete[] a;
    #pragma omp critical
    {
    
    }
    
    
    
    #pragma omp atomic
    
    #pragma omp single // implicit barrier, can be used nowait 
    {
    
    }
    #pragma omp master
    {
    
    }
    #pragma omp barrier
    
    // auto split workload
    #pragma omp for
    for(int i=0; i<16; i++){
    
    }
}
   
    // auto split workload
    #pragma omp parallel for
    for(int i=0; i<16; i++){
    
    }
    #pragma omp parallel for colaspe(2) // 2 loops
    for(int i=0; i<16; i++){
        for(int j=0; j<16; j++){
    
        }
    }
    // reduction
    int sum=0;
    #pragma omp parallel for reduction(+:sum) 
    for(int i=0; i<16; i++){
        sum+=i;
    }
    
    #pragma omp parallel for schedule (dynamic, 1) num_threads (10)
    #pragma omp parallel for schedule (static, 1) num_threads (10)
    
    
    
    
}




void hello(){

// To use global variable export OMP_NUM_THREADS=N
// To overwrite omp_set_num_threads(X);

omp_set_num_threads(4);

#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    int n_threads = omp_get_num_threads();
    std::cout<<"Hello1 "<<thread_id<<" "<<n_threads<<std::endl;

    if(thread_id == 0 ) {
        std::cout<<"Master "<<thread_id<<" "<<n_threads<<std::endl;
    }

}


/*
#pragma omp parallel num_threads(4)
{
    std::cout<<"Hello2"<<std::endl;
}*/

}



int main(int argc, char* argv[]){

    //hello();
    falseSharing();
    return 0;
}
