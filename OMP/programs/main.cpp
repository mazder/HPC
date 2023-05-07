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
}
    delete[] a;

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
