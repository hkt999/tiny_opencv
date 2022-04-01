#include "tiny_opencv.hpp"
#include <stdlib.h>
#include <cmath>
#include <iomanip>

namespace KCV {
 
static void create_gaussian_matrix(float *kernel, int cols, int rows, double sigma = 1.0)
{
    // set standard deviation to 1.0
    double r, s = 2.0 * sigma * sigma;
    int pad = cols / 2;
 
    // sum is for normalization
    double sum = 0.0;
 
    // generate gaussian kernel
    float *k = (float *)kernel;
    for(int y = -pad; y <= pad; y++) {
        for (int x = -pad; x <= 2; x++) {
            r = sqrt(x*x + y*y);
            *k = (exp(-(r*r)/s))/(M_PI * s);
            sum += *k;
            k++;
        }
    }
 
    // normalize the Kernel
    int count = cols * rows;
    k = kernel;
    while (count-->0) {
        *k = *k / sum;
        k++;
    }
}

void gaussianBlur(const Mat src, Mat &dst, Size ksize, double sigmaX, double sigmaY)
{
    if (sigmaY == 0.0)
        sigmaY = sigmaX;

    if (sigmaX == 0) 
        sigmaX = 1.0;

    /* build kernel */
    Mat kernel(ksize.width, ksize.height, CV_32FC1);
    create_gaussian_matrix((float *)kernel.data, ksize.width, ksize.height, sigmaX);
    filter2D(src, dst, -1, kernel);
}

}