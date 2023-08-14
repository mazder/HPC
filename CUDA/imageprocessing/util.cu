#include "util.cuh"

util::util()
{
    //ctor
}

util::~util()
{
    //dtor
}

void util::test()
{
    std::vector<int> v = { 1, 2, 3, 4, 5 };

    std::cout << "v.size() = " << v.size() << '\n';
    std::cout << "v.capacity() = " << v.capacity() << '\n';

    std::cout << '\n';

    v.clear();

    std::cout << "v.size() = " << v.size() << '\n';
    std::cout << "v.capacity() = " << v.capacity() << '\n';

    std::cout << '\n';

    std::vector<int>().swap( v );

    std::cout << "v.size() = " << v.size() << '\n';
    std::cout << "v.capacity() = " << v.capacity() << '\n';

}

void functionality(const kernel& k_kernel, const image& m_image){

  /*
    for (d=0 ; d<3 ; d++) {
        for (i=0 ; i<new_im_height ; i++) {
            for (j=0 ; j<new_im_width ; j++) {
                for (h=i ; h<i+k_height ; h++) {
                    for (w=j ; w<j+k_width ; w++) {
                        new_im_matrix[d][i][j] += kernel_matrix[h-i][w-j]*im_matrix[d][h][w];
                    }
                }
            }
        }
    }
    */

/*
    std::vector<float> new_org_im_matrix(3*im_height*im_width);
    for(int i=0; i<3; i++){
        for(int h=0; h<im_height; h++){
            for(int w=0; w<im_width; w++){
                new_org_im_matrix[i*(im_height*im_width)+h*im_width+w]=im_matrix[i*(im_height*im_width)+h*im_width+w];
            }
        }
    }
    saveImage("new_filtered_image.png", new_org_im_matrix, im_height, im_width);
*/

}
