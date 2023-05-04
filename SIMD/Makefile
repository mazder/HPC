

all : simdprogram

simdprogram : simdprogram.cpp
	g++  -mavx512f simdprogram.cpp -o simdprogram -fopenmp -O3

clean :
	rm -rf simdprogram
