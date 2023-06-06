NVCC=/usr/local/cuda-12.1/bin/nvcc
CC=gcc
INC=/usr/local/cuda-12.1/include
LIB=/usr/local/cuda-12.1/lib64

all : main

main : main.cpp
	$(CC) -I$(INC) -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100  $^ -o $@ -L$(LIB) -lOpenCL

clean :
	rm -rf main
