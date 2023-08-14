#include <iostream>
#include <chrono>

#include "util.cuh"
#include "image.cuh"
#include "kernel.h"

using namespace std;

int main()
{

    image myImage;
    auto t1=std::chrono::high_resolution_clock::now();
    myImage.loadImage("me.png");
    myImage.saveImage("newme.png");

    kernel myKernel;
    myKernel.buildKernelGaussian(5,5,100.0);
    //myKernel.buildKernelGeneral(3,3);
    myKernel.printKernel();

    myImage.applyFilter(myKernel);

    auto t2=std::chrono::high_resolution_clock::now();

    auto duration=std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();

    //std::cout<<"Execution Time::> "<<duration<<"µs"<<std::endl;
    std::cout<<"Execution Time::> "<<duration<<"\xc2\xb5s"<<std::endl;

    return 0;
}
