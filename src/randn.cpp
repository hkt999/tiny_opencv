#include <math.h>
#include "tiny_opencv.hpp"

namespace KCV {

inline double rand_gen() {
    // return a uniformly distributed random value
    return rand() / (double)RAND_MAX;
}

/* 
    u = rand() / (double)RAND_MAX;
    v = rand() / (double)RAND_MAX;
    x = sqrt(-2 * log(u)) * cos(2 * M_PI * v) * std + mean;
*/
double normal_random(double mean, double sigma) {
    double u = rand_gen();
    double v = rand_gen();
    return sqrt(-2 * log(u)) * cos(2 * M_PI * v) * sigma + mean;
}

void randn(Mat &dst, float mean, float sigma)
{
    int dtype = dst.type&0x07;
    int ch = 1 + (dst.type>>3);
    int counter = dst.cols * dst.rows * ch;

    switch (dtype) {
        case CV_32F: {
            float *data = dst.getData<float>();
            while (counter-->0) {
                *data++ = (float) normal_random(mean, sigma);
            }
        }
        break;

        case CV_64F: {
            double *data = dst.getData<double>();
            while (counter-->0) {
                *data++ = (double) normal_random(mean, sigma);
            }
        }
        break;
    }
}

}
