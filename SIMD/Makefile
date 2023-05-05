

all : simdprogram navectoradd xsimd sumdotproduct helloavx

simdprogram : simdprogram.cpp
	g++  -mavx512f simdprogram.cpp -o simdprogram -fopenmp -O3

navectoradd : navectoradd.cpp
	g++  -mavx512f navectoradd.cpp -o navectoradd -fopenmp -O3

xsimd : xsimd.cpp
	g++  -mavx512f xsimd.cpp -o xsimd -fopenmp -O3
	#g++  -mavx xsimd.cpp -o xsimd -fopenmp -O3

sumdotproduct : sumdotproduct.cpp
	g++  -mavx512f sumdotproduct.cpp -o sumdotproduct -fopenmp -O3
	#g++  -mavx sumdotproduct.cpp -o sumdotproduct -fopenmp -O3

helloavx : helloavx.cpp
	g++  -mavx512f helloavx.cpp -o helloavx -fopenmp -O3
	#g++  -mavx helloavx.cpp -o helloavx -fopenmp -O3

clean :
	rm -rf simdprogram navectoradd xsimd
