#include "tiny_opencv.hpp"
#include <cstring>
#include <cstdlib>

namespace KCV {

#define MAX_CHAN    4
#define MUL         8192

extern void clear_boundary(Mat &dst, Size ksize);
static void make_fix_point_kernel( const Mat kernel, Mat &fix_point_kernel)
{
    fix_point_kernel.cols = kernel.cols;
    fix_point_kernel.rows = kernel.rows;
    fix_point_kernel.type = CV_32SC1;
    fix_point_kernel.createBuffer();

    int kernel_size = kernel.cols * kernel.rows;
    int32_t *fixp_kernel = (int32_t *)fix_point_kernel.data;
    int i;

    switch (kernel.type) {
        case CV_8UC1:
            {
                uchar *uc_kernel = (uchar *) kernel.data;
                for (int i=0; i<kernel_size; i++) {
                    fixp_kernel[i] = ((int32_t)uc_kernel[i]) * MUL;
                }
            }
            break;

        case CV_32SC1:
            {
                int32_t *i32_kernel = (int32_t *) kernel.data;
                for (int i=0; i<kernel_size; i++) {
                    fixp_kernel[i] = (int32_t)i32_kernel[i];
                }
            }
            break;

        case CV_32FC1:
            {
                float *fp_kernel = (float *) kernel.data;
                for (i=0; i<kernel_size; i++) {
                    fixp_kernel[i] = (int32_t)(fp_kernel[i] * (float) MUL);
                }
            }
            break;

        case CV_64FC1:
            {
                double *dp_kernel = (double *)kernel.data;
                for (i=0; i<kernel_size; i++) {
                    fixp_kernel[i] = (int32_t)(dp_kernel[i] * (double) MUL);
                }
            }
            break;
    }
}

// matrix convolution
static void do_convolution(const Mat src, Mat &dst, Mat &kernel, int chans)
{
    Mat fix_point_kernel;

    make_fix_point_kernel(kernel, fix_point_kernel);
    Size ksize(kernel.cols, kernel.rows);
    int h_pad = (ksize.width-1)/2;
    int v_pad = (ksize.height-1)/2;
    int32_t value[MAX_CHAN];

    dst.cols = src.cols;
    dst.rows = src.rows;
    dst.type = src.type;
    dst.createBuffer();

    int32_t *k_val = (int32_t *)fix_point_kernel.data;
    uchar *wline = (uchar *)dst.data + (v_pad * src.cols + h_pad) * chans;
    int line_bytes = dst.cols * chans;
    for (int y=v_pad; y<src.rows-v_pad; y++) {
        uchar *w = wline;
        for (int x=h_pad; x<src.cols-h_pad; x++) {
            for (int c=0; c<chans; c++)
                value[c] = 0;

            uchar *rline = (uchar *)src.data + ((y-v_pad) * src.cols + x - h_pad) * chans;
            // covolution kernel
            int32_t *k_idx = k_val;
            for (int by=-v_pad; by<=v_pad; by++) {
                uchar *r = rline;
                for (int bx=-h_pad; bx<=h_pad; bx++) {
                    for (int c=0; c<chans; c++) {
                        value[c] += ((int32_t)(*r++) * (*k_idx));
                    }
                    k_idx++;
                }
                rline += line_bytes;
            }
            for (int c=0; c<chans; c++) {
                *w++ = (uchar)(value[c]/MUL);
            }
        }
        wline += line_bytes;
    }

    clear_boundary(dst, ksize);
}

void filter2D(const Mat src, Mat &dst, int ddepth, Mat &kernel)
{
    int chans;
    switch (src.type) {
        case CV_8UC1:
            chans = 1;
            break;

        case CV_8UC3:
            chans = 3;
            break;
    }
    do_convolution(src, dst, kernel, chans);
}

} // end of namespace