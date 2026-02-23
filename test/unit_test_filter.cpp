#include "tiny_opencv.hpp"
#include "unit_test.hpp"

using namespace KCV;
extern void *dup_mat_data( Mat &mat );

void *unit_test_equalize_hist(void *data, int width, int height)
{  
    Mat src(height, width, CV_8UC1, data);
    Mat dst;

    equalizeHist(src, dst);

    return dup_mat_data(dst);
}

void *unit_test_blur_c1(void *data, int width, int height, int kernel_w, int kernel_h)
{
    Mat src(height, width, CV_8UC1, data);
    Mat dst;

    blur(src, dst, Size(kernel_w, kernel_h));

    return dup_mat_data(dst);
}

void *unit_test_blur_c3(void *data, int width, int height, int kernel_w, int kernel_h)
{
    Mat src(height, width, CV_8UC3, data);
    Mat dst;

    blur(src, dst, Size(kernel_w, kernel_h));

    return dup_mat_data(dst);
}

void *unit_test_filter2d_c1(void *data, int width, int height, int kernel_w, int kernel_h, float *kernel_v)
{
    Mat src(height, width, CV_8UC1, data);
    Mat dst;
    Mat kernel(kernel_h, kernel_w, CV_32FC1, kernel_v);

    filter2D(src, dst, -1, kernel);

    return dup_mat_data(dst);
}

void *unit_test_filter2d_c3(void *data, int width, int height, int kernel_w, int kernel_h, float *kernel_v)
{
    Mat src(height, width, CV_8UC3, data);
    Mat dst;
    Mat kernel(kernel_h, kernel_w, CV_32FC1, kernel_v);

    filter2D(src, dst, -1, kernel);

    return dup_mat_data(dst);
}

void *unit_test_gaussian_c1(void *data, int width, int height, int ksize, double sigma)
{
    (void)ksize;
    (void)sigma;
    Mat src(height, width, CV_8UC1, data);
    Mat dst;

    gaussianBlur(src, dst, Size(5,5), 0);

    return dup_mat_data(dst);
}

void *unit_test_gaussian_c3(void *data, int width, int height, int ksize, double sigma)
{
    (void)ksize;
    (void)sigma;
    Mat src(height, width, CV_8UC3, data);
    Mat dst;

    gaussianBlur(src, dst, Size(5,5), 0);

    return dup_mat_data(dst);
}

void *unit_test_bilateral_c1(void *data, int width, int height, int d, double sigmaColor, double sigmaSpace )
{
    Mat src(height, width, CV_8UC1, data);
    Mat dst;

    bilateralFilter( src, dst, d, sigmaColor, sigmaSpace );

    return dup_mat_data(dst);
}

void *unit_test_bilateral_c3(void *data, int width, int height, int d, double sigmaColor, double sigmaSpace )
{
    Mat src(height, width, CV_8UC3, data);
    Mat dst;

    bilateralFilter( src, dst, d, sigmaColor, sigmaSpace );

    return dup_mat_data(dst);
}

void *unit_test_median_blue_c1(void *data, int width, int height, int ksize)
{
    Mat src(height, width, CV_8UC1, data);
    Mat dst;

    medianBlur(src, dst, ksize);

    return dup_mat_data(dst);
}

void *unit_test_median_blue_c3(void *data, int width, int height, int ksize)
{
    Mat src(height, width, CV_8UC3, data);
    Mat dst;
    medianBlur(src, dst, ksize);
    return dup_mat_data(dst);
}

void *unit_test_resize_c1(void *data, int width, int height, int dst_width, int dst_height)
{
    Mat src(height, width, CV_8UC1, data);
    Mat dst;
    resize(src, dst, Size(dst_width, dst_height));
    return dup_mat_data(dst);
}

void *unit_test_resize_c3(void *data, int width, int height, int dst_width, int dst_height)
{
    Mat src(height, width, CV_8UC3, data);
    Mat dst;

    resize(src, dst, Size(dst_width, dst_height));
    return dup_mat_data(dst);
}

void *unit_test_crop_c1(void *data, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h)
{
    Mat src(height, width, CV_8UC1, data);
    Rect roi(roi_x, roi_y, roi_w, roi_h);

    Mat dst = src(roi);
    return dup_mat_data(dst);
}

void *unit_test_crop_c3(void *data, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h)
{
    float m1d[] = {1,3,2,4};
    float m2d[] = {5,7,6,8};
    Mat m1(2, 2, CV_32F, m1d);
    Mat m2(2, 2, CV_32F, m2d);
    Mat inv = m1.inverse();
    Mat test = m1 * inv;

    Mat m3 = m1 * m2;

    Mat src(height, width, CV_8UC3, data);
    Rect roi(roi_x, roi_y, roi_w, roi_h);

    Mat dst = src(roi);
    return dup_mat_data(dst);
}
