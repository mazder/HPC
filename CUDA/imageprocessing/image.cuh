#ifndef IMAGE_CUH
#define IMAGE_CUH

#include<vector>
#include<png++/png.hpp>


#include "convolution.cuh"
#include "kernel.h"

class image
{
    public:
        image();
        virtual ~image();
        void loadImage(const char *filename);
        void saveImage(const char *filename) const;
        void saveImage(const char *filename, std::vector<float>& matrix, int height, int width) const;
        void applyFilter(const kernel& k_kernel);
        void applyFilterGPU(const kernel& k_kernel);

    private:

        std::vector<float> im_matrix;
        int im_height;
        int im_width;
};

#endif // IMAGE_CUH
