CC=g++
INC=/usr/local/cuda-12.1/include
LIB=/usr/local/cuda-12.1/lib64
COMPUTE=/home/mrahman/HPC/OPENCL/compute/include


all : kernel.h main

kernel.h: kernel.cl
	./stringify_ocl kernel.cl

main : main.cpp
	$(CC) -std=c++0x -I$(INC) $^ -o $@ -L$(LIB) -lOpenCL

clean :
	rm -rf main kernel.h
