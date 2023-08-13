// convolution

#include<iostream>
#include "functions.hpp"

void convolution_2D_host(int* matrix, int* outmatrix, int H, int W, int* mask){

    int temp=0;
    int r_offset;
    int c_offset;
    for(int h=0; h<H; h++){
        for(int w=0; w<W; w++){
            temp=0;
            for(int i=0; i<MASKDIM; i++){
                r_offset=h-MASK_OFFSET+i;
                for(int j=0; j<MASKDIM; j++){
                    c_offset=w-MASK_OFFSET+j;
                    if(r_offset>=0 && r_offset<H){
                        if(c_offset>=0 && c_offset<W){
                            temp+=matrix[r_offset*W+c_offset]*mask[i*MASKDIM+j];
                        }
                    }
                }
            }
            outmatrix[h*W+w]=temp;
        }
    }
}

void convolution(int* matrix, int* outmatrix, int H, int W, int* mask, int MASKD){

    int temp=0;
    int r_offset;
    int c_offset;
    int new_H=(H-MASKD+1);
    int new_W=(W-MASKD+1);

    for(int h=0; h<new_H; h++){
        for(int w=0; w<new_W; w++){
            temp=0;
            for(int i=h; i<MASKD+h; i++){
                r_offset=i-h;
                for(int j=w; j<MASKD+w; j++){
                    c_offset=j-w;
                    temp+=mask[r_offset*MASKD+c_offset]*matrix[i*W+j];
                }
            }
            outmatrix[h*new_W+w]=temp;
        }
    }
}
