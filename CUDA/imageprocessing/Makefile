#Makefile

#define variables
objects= main.o convolution.o kernel.o image.o util.o

NVCC= /usr/local/cuda-12.1/bin/nvcc             #cuda c compiler
opt= -O3 -g -G           			#optimization flag
ARCH= -arch=sm_86        			#cuda compute capability
LIBS= -L/usr/local/cuda-12.1/lib64

IMG_LDFLAG	= -lpng
LDFLAGS 	= $(IMG_LDFLAG) -lm

INC= -I/usr/local/cuda-12.1/include
execname= main


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) $(INC) $(LDFLAGS)

convolution.o: convolution.cu
	$(NVCC) $(opt) $(ARCH) -c convolution.cu $(INC)

kernel.o: kernel.cpp
	$(NVCC) $(opt) $(ARCH) -c kernel.cpp $(INC)

image.o: image.cu
	$(NVCC) $(opt) $(ARCH) -c image.cu $(INC)

util.o: util.cu
	$(NVCC) $(opt) $(ARCH) -c util.cu $(INC)

main.o: main.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cu $(INC)


#clean Makefile
clean:
	rm $(objects)
#end of Makefile
