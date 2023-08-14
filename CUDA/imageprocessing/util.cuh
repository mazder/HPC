#ifndef UTIL_CUH
#define UTIL_CUH

#include <iostream>
#include<vector>

#include "kernel.h"
#include "image.cuh"

class util
{
    public:
        util();
        virtual ~util();
        void test();
        void functionality(const kernel& k_kernel, const image& m_image);
    private:
};

#endif // UTIL_CUH
