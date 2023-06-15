#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 120

#include<CL/cl.hpp>
#include<cstdio>
#include<iostream>
#include<vector>
#include<cassert>
#include <algorithm>
#include <iterator>
#include "util.hpp"

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
    cl::Program program(context, util::loadProgram("kernel.cl"));

    if(program.build({device}) != CL_SUCCESS){
        std::cout << "Error in kernel building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    // data size
    int N = 1024 ;

    // host data
    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::vector<int> h_C(N);
    // initialize
    for(int i=0; i<N; i++){
        h_A[i]=i;
        h_B[i]=N-i;
    }

    // buffer on device
    // create buffer on device
    cl::Buffer d_A(context, CL_MEM_READ_WRITE, sizeof(int)*N);
    cl::Buffer d_B(context, CL_MEM_READ_WRITE, sizeof(int)*N);
    cl::Buffer d_C(context, CL_MEM_READ_WRITE, sizeof(int)*N);

    // create command queue that device executes
    cl::CommandQueue queue(context, device);

    auto vector_add = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(program, "vector_add");

    // write to device
    queue.enqueueWriteBuffer(d_A, CL_TRUE, 0, sizeof(int)*N, h_A.data());
    queue.enqueueWriteBuffer(d_B, CL_TRUE, 0, sizeof(int)*N, h_B.data());

    // launch kernel
    vector_add(cl::EnqueueArgs( queue, cl::NDRange(N), cl::NDRange(32)), d_A, d_B, d_C);
    queue.finish();

    // copy device to host
    queue.enqueueReadBuffer(d_C, CL_TRUE, 0, sizeof(int)*N, h_C.data());

    // check results
    bool success=true;
    for(int i=0; i<N; i++){
        if((h_A[i]+h_B[i])!=h_C[i]){
            success = false;
            break;
        }
        std::cout<<h_A[i] <<" + "<<h_B[i]<<" = "<<h_C[i]<<std::endl;
    }

    if (success)
        std::cout << "Success!" << std::endl;
    else
        std::cout << "Failed!" << std::endl;

    return 0;
}
