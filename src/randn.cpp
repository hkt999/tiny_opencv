#include <math.h>
#include "tiny_opencv.hpp"

namespace KCV {

double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

double normal_random() {
   double v1 = rand_gen();
   double v2 = rand_gen();
   return cos(2.0 * M_PI * v2) * sqrt(-2. * log(v1));
}

void randn(Mat &dst, float mean, float sigma)
{
    float *data = dst.getData<float>();
    int dtype = dst.type&0x07;
    int ch = 1 + (dst.type>>3);
    int counter = dst.cols * dst.rows * ch;

    switch (dtype) {
        case CV_32F: {
            float *data = dst.getData<float>();
            while (counter-->0) {
                *data++ = (float) normal_random();
            }
        }
        break;

        case CV_64F: {
            double *data = dst.getData<double>();
            while (counter-->0) {
                *data++ = normal_random();
            }
        }
        break;
    }
}

}