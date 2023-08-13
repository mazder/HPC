NVCC = /usr/local/cuda-12/bin/nvcc
INC = -I/usr/local/cuda-12/include

#Makefile
#define variables
objects = main.o kernels.o functions.o util.o
NVCC = nvcc               #cuda c compiler
opt = -O3 -g -G           #optimization flag
ARCH = -arch=sm_86        #cuda compute capability
LIBS =
execname = main


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) $(INC)

util.o: util.cpp
	$(NVCC) $(opt) $(ARCH) -c util.cpp $(INC)

functions.o: functions.cpp
	$(NVCC) $(opt) $(ARCH) -c functions.cpp $(INC)

kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu $(INC)

main.o: main.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cu $(INC)


#clean Makefile
clean:
	rm $(objects) $(execname)
#end of Makefile
