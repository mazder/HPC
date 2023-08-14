#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <chrono>


#include "image.cuh"

image::image()
{
    //ctor
    im_height=0;
    im_width=0;
}

image::~image()
{
    //dtor
    im_matrix.clear();
    std::vector<float>().swap(im_matrix);
}

void image::loadImage(const char *filename){

    png::image<png::rgb_pixel> image(filename);

    this->im_height=image.get_height();
    this->im_width=image.get_width();

    std::vector<float> imageMatrix(3*im_height*im_width); // imageMatrix[i][h][w]

    for(int h=0; h<im_height; h++){
        for(int w=0; w<im_width; w++){
            imageMatrix[0*(im_height*im_width)+h*im_width+w]=image[h][w].red;
            imageMatrix[1*(im_height*im_width)+h*im_width+w]=image[h][w].green;
            imageMatrix[2*(im_height*im_width)+h*im_width+w]=image[h][w].blue;
        }
    }

    this->im_matrix=imageMatrix;
}


void image::saveImage(const char *filename) const {

    int height=this->im_height;
    int width=this->im_width;
    png::image<png::rgb_pixel> image(width, height);

    for(int h=0; h<im_height; h++){
        for(int w=0; w<im_width; w++){
            image[h][w].red=this->im_matrix[0*(im_height*im_width)+h*im_width+w];
            image[h][w].green=this->im_matrix[1*(im_height*im_width)+h*im_width+w];
            image[h][w].blue=this->im_matrix[2*(im_height*im_width)+h*im_width+w];
        }
    }
    image.write(filename);
}


void image::saveImage(const char *filename, std::vector<float>& matrix, int height, int width) const {

    png::image<png::rgb_pixel> image(width, height);

    for(int h=0; h<height; h++){
        for(int w=0; w<width; w++){
            image[h][w].red=matrix[0*(height*width)+h*width+w];
            image[h][w].green=matrix[1*(height*width)+h*width+w];
            image[h][w].blue=matrix[2*(height*width)+h*width+w];
        }
    }
    image.write(filename);
}

void image::applyFilter(const kernel& k_kernel){

    std::vector<float> kernel_matrix=k_kernel.getKernel();

    int k_height=k_kernel.getHeight();
    int k_width=k_kernel.getWidth();

    int new_im_height=im_height-k_height+1;
    int new_im_width=im_width-k_width+1;

    std::vector<float> new_im_matrix(3*new_im_height*new_im_width);
    int temp=0;

    for(int i=0; i<3; i++){
        for(int h=0; h<new_im_height; h++){
            for(int w=0; w<new_im_width; w++){
                temp=0;
                for(int kh=h; kh<k_height+h; kh++){
                    for(int kw=w; kw<k_width+w; kw++){
                        temp+=kernel_matrix[(kh-h)*k_width+(kw-w)]*im_matrix[(i*(im_height*im_width))+kh*(im_width)+kw];
                    }
                }
                new_im_matrix[i*(new_im_height*new_im_width)+(h*new_im_width)+w]=temp;
            }
        }
    }
    saveImage("new_filtered_image.png", new_im_matrix, new_im_height, new_im_width);
}

void image::applyFilterGPU(const kernel& k_kernel){

    std::vector<float> kernel_matrix=k_kernel.getKernel();
    int k_height=k_kernel.getHeight();
    int k_width=k_kernel.getWidth();
    int new_im_height=im_height-k_height+1;
    int new_im_width=im_width-k_width+1;
    std::vector<float> new_im_matrix(3*new_im_height*new_im_width);

    // TODO write

    //saveImage("new_filtered_image.png", new_im_matrix, new_im_height, new_im_width);
}

