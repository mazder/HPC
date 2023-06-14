#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 120

#include<CL/cl.hpp>
#include<cstdio>
#include<iostream>
#include<vector>
#include<cassert>
#include <algorithm>
#include <iterator>
#include "kernel.h" // This kernel.h will be generated at the build time

int main(void)
{
    // get platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(platforms.size()!=0);
    // set platform
    cl::Platform platform=platforms[0];

    // get devices
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(devices.size()!=0);
    //set device
    cl::Device device = devices[0];

    // create context
    cl::Context context({device});


    cl::Program::Sources sources;
    // load kernel codes
    sources.push_back({kernel_ocl, strlen(kernel_ocl)});
    cl::Program program(context, sources);
    if(program.build({device}) != CL_SUCCESS){
        std::cout << "Error in kernel building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    // data size
    int N = 1024 ;

    // create buffer on device
    cl::Buffer buff_A(context, CL_MEM_READ_WRITE, sizeof(int)*N);
    cl::Buffer buff_B(context, CL_MEM_READ_WRITE, sizeof(int)*N);
    cl::Buffer buff_C(context, CL_MEM_READ_WRITE, sizeof(int)*N);

    // create data on host
    int A[N], B[N], C[N];
    for(int i=0; i<N; i++){
        A[i]=i;
        B[i]=N-i;
    }

    // create command queue that device executes
    cl::CommandQueue queue(context, device);

    // write to device
    queue.enqueueWriteBuffer(buff_A, CL_TRUE, 0, sizeof(int)*N, A);
    queue.enqueueWriteBuffer(buff_B, CL_TRUE, 0, sizeof(int)*N, B);

    // create kernel
    cl::Kernel kernel(program, "vector_add");
    //set arguments
    kernel.setArg(0, buff_A);
    kernel.setArg(1, buff_B);
    kernel.setArg(2, buff_C);


    //run kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(32));
    queue.finish();

    // read from device
    queue.enqueueReadBuffer(buff_C, CL_TRUE, 0, sizeof(int)*N, C);


    // check results
    bool success=true;
    for(int i=0; i<N; i++){
        if((A[i]+B[i])!=C[i]){
            success = false;
            break;
        }
        std::cout<<A[i] <<" + "<<B[i]<<" = "<<C[i]<<std::endl;
    }

    if (success)
        std::cout << "Success!" << std::endl;
    else
        std::cout << "Failed!" << std::endl;

    return 0;
}
