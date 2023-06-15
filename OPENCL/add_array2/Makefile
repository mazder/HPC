CC=g++
INC=/usr/local/cuda-12.1/include
LIB=/usr/local/cuda-12.1/lib64

all : main

main : main.cpp
	$(CC) -std=c++0x -I$(INC) $^ -o $@ -L$(LIB) -lOpenCL

clean :
	rm -rf main
