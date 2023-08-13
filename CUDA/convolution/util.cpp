// utilities

#include<iostream>
#include "util.hpp"

void print_matrix(int* matrix, int H, int W){
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            std::cout<<matrix[i*W+j]<<" ";
        }
        std::cout<<std::endl;
    }

}

void init_matrix(int* matrix, int H, int W){
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
           matrix[i*W+j]=rand()%10;
        }
    }
}

void init_matrix_zero(int* matrix, int H, int W){
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
           matrix[i*W+j]=0;
        }
    }
}

bool check_equivalent(int* matrix, int H, int W, int* outmatrix){
    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
           if(matrix[i*W+j]!=outmatrix[i*W+j]){
                return false;
           }
        }
    }

    return true;
}
