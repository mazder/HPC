#include "kernel.h"

kernel::kernel()
{
    //ctor
    this->k_height=0;
    this->k_width=0;
    this->k_matrix=std::vector<float>(0);
}

kernel::~kernel()
{
    //dtor
    k_matrix.clear();
    std::vector<float>().swap(k_matrix);
}


void kernel::buildKernelGeneral(int height, int width){

    this->k_height=height;
    this->k_width=width;
    double sum=0.0;

    std::vector<float> kernel_matrix(this->k_height*this->k_width);

    for(int h=0; h<this->k_height; h++){
        for(int w=0; w<this->k_width; w++){
            if((h+1)==w || (w+1)==h || (h==k_height/2 && w==k_width/2) ){
                kernel_matrix[h*k_width+w]=1;
            }
            else{
                kernel_matrix[h*k_width+w]=0;
            }
        }
    }

    this->k_matrix=kernel_matrix;
}


void kernel::buildKernelGaussian(int height, int width, double sigma){

    this->k_height=height;
    this->k_width=width;
    double sum=0.0;

    std::vector<float> kernel_matrix(this->k_height*this->k_width);

    for(int h=0; h<this->k_height; h++){
        for(int w=0; w<this->k_width; w++){
            kernel_matrix[h*k_width+w]=exp(-(h*h+w*w)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
            sum+=kernel_matrix[h*k_width+w];
        }
    }

    for(int h=0; h<this->k_height; h++){
        for(int w=0; w<this->k_width; w++){
            kernel_matrix[h*k_width+w]/=sum;
        }
    }

    this->k_matrix=kernel_matrix;
}

void kernel::printKernel() const{

    if(this->k_height==0 || this->k_width==0){
        std::cout<<"Kernel is set"<<std::endl;
    }
    for(int h=0; h<this->k_height; h++){
        for(int w=0; w<this->k_width; w++){
            std::cout<<(float)this->k_matrix[h*k_width+w]<<" ";
        }
        std::cout<<std::endl;
    }
}
