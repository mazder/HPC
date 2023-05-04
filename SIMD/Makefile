

all : simdprogram navectoradd

simdprogram : simdprogram.cpp
	g++  -mavx512f simdprogram.cpp -o simdprogram -fopenmp -O3

navectoradd : navectoradd.cpp
	g++  -mavx512f navectoradd.cpp -o navectoradd -fopenmp -O3

clean :
	rm -rf simdprogram navectoradd
