// linear seach
#include<iostream>
#include "functions.hpp"

void max_host(int *arr, unsigned int N, int* mx){
    for(unsigned int i=0; i<N; i++){
        if(arr[i]>*mx) *mx=arr[i];
    }

}