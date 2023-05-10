CC=/usr/local/cuda-11.7/bin/nvcc
INC=/usr/local/cuda-11.7/include

all : addVector addVectorPinned addVectorUM addVectorUMPrefetch reductionVector matrix_multiply matrix_multiply_cpp_shared

addVector : addVector.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

addVectorPinned : addVectorPinned.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

addVectorUM : addVectorUM.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

addVectorUMPrefetch : addVectorUMPrefetch.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

reductionVector : reductionVector.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

matrix_multiply : matrix_multiply.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

matrix_multiply_cpp_shared : matrix_multiply_cpp_shared.cu
	$(CC) -g -G -I$(INC) --std=c++11 $^ -o $@

clean :
	rm -rf addVector addVectorPinned addVectorUM addVectorUMPrefetch reductionVector matrix_multiply matrix_multiply_cpp_shared
