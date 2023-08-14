#ifndef KERNEL_H
#define KERNEL_H
#include<iostream>
#include<vector>
#include<cmath>


class kernel
{
    public:
        kernel();
        virtual ~kernel();
        void buildKernelGaussian(int height, int width, double sigma);
        void buildKernelGeneral(int height, int width);
        inline std::vector<float> getKernel() const {return k_matrix;};
        void printKernel() const;
        inline int getHeight() const {return k_height;};
        inline int getWidth() const {return k_width;};

    private:
        std::vector<float>k_matrix;
        int k_height;
        int k_width;
};

#endif // KERNEL_H
