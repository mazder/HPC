// gcc -I /usr/local/cuda-12.1/include -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100 main.cpp -o main -L /usr/local/cuda-12.1/lib64 -lOpenCL
// C standard includes
#include <stdio.h>

// OpenCL includes
#include <CL/cl.h>

int main()
{
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    CL_err = clGetPlatformIDs( 0, NULL, &numPlatforms );

    if (CL_err == CL_SUCCESS)
        printf("%u platform(s) found\n", numPlatforms);
    else
        printf("clGetPlatformIDs(%i)\n", CL_err);

    return 0;
}
