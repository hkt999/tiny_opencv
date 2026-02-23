#include "tiny_opencv.hpp"
#include <stdlib.h>
#include <cmath>
#include <iomanip>

namespace KCV {
 
static void create_gaussian_matrix(float *kernel, int cols, int rows, double sigma_x = 1.0, double sigma_y = 1.0)
{
    if (sigma_x <= 0.0)
        sigma_x = 1.0;
    if (sigma_y <= 0.0)
        sigma_y = sigma_x;

    const double cx = (cols - 1) * 0.5;
    const double cy = (rows - 1) * 0.5;
    const double sxx = 2.0 * sigma_x * sigma_x;
    const double syy = 2.0 * sigma_y * sigma_y;
    const double norm = 1.0 / (2.0 * M_PI * sigma_x * sigma_y);
    double sum = 0.0;

    float *k = kernel;
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            const double dx = x - cx;
            const double dy = y - cy;
            const double v = norm * exp(-(dx * dx / sxx + dy * dy / syy));
            *k = (float)v;
            sum += v;
            k++;
        }
    }

    const int count = cols * rows;
    for (int i = 0; i < count; i++) {
        kernel[i] = (float)(kernel[i] / sum);
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
    create_gaussian_matrix((float *)kernel.ref->data, ksize.width, ksize.height, sigmaX, sigmaY);
    filter2D(src, dst, -1, kernel);
}

}
