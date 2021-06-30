#include <string.h>
#include <stdlib.h>
#include "opencv.hpp"
#include "unit_test.hpp"

using namespace KCV;
extern void *dup_mat_data( Mat &mat );

// utility
void *dup_mat_data( Mat &mat );

void *unit_test_bgr2gray(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC3, src);
    Mat mdst(height, width, CV_8UC1);

    cvtColor(msrc, mdst, CV_BGR2GRAY);
    return dup_mat_data(mdst);
}

void *unit_test_rgb2gray(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC3, src);
    Mat mdst(height, width, CV_8UC1);

    cvtColor(msrc, mdst, CV_RGB2GRAY);
    return dup_mat_data(mdst);
}

void *unit_test_gray2bgr(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC1, src);
    Mat mdst(height, width, CV_8UC3);

    cvtColor(msrc, mdst, CV_GRAY2BGR);
    return dup_mat_data(mdst);
}

void *unit_test_gray2rgb(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC1, src);
    Mat mdst(height, width, CV_8UC3);

    cvtColor(msrc, mdst, CV_GRAY2RGB);
    return dup_mat_data(mdst);
}

void *uint_test_bgr2yuv_i420(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC3, src);
    Mat mdst(height*3/2, width, CV_8UC1);

    cvtColor(msrc, mdst, CV_BGR2YUV_I420);
    return dup_mat_data(mdst);
}

void *uint_test_rgb2yuv_i420(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC3, src);
    Mat mdst(height*3/2, width, CV_8UC1);

    cvtColor(msrc, mdst, CV_RGB2YUV_I420);
    return dup_mat_data(mdst);
}

void *unit_test_yuv2bgr_i420(void *src, int width, int height)
{
    Mat msrc(height*3/2, width, CV_8UC1, src);
    Mat mdst;

    cvtColor(msrc, mdst, CV_YUV2BGR_I420);
    return dup_mat_data(mdst);
}

void *unit_test_yuv2rgb_i420(void *src, int width, int height)
{
    Mat msrc(height*3/2, width, CV_8UC1, src);
    Mat mdst;

    cvtColor(msrc, mdst, CV_YUV2RGB_I420);
    return dup_mat_data(mdst);
}

void *unit_test_bgr2rgb(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC3, src);
    Mat mdst;

    cvtColor(msrc, mdst, CV_BGR2RGB);
    return dup_mat_data(mdst);
}

void *unit_test_rgb2bgr(void *src, int width, int height)
{
    Mat msrc(height, width, CV_8UC3, src);
    Mat mdst;

    cvtColor(msrc, mdst, CV_RGB2BGR);
    return dup_mat_data(mdst);
}
