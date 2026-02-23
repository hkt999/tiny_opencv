#include "tiny_opencv.hpp"

using namespace KCV;

extern int rgb24_to_yuv420 (int x_dim, int y_dim, unsigned char *bmp, unsigned char *yuv, int flip);
extern void yuv420p_to_rgb24(unsigned char* yuvbuffer,unsigned char* rgbbuffer, int width,int height);
extern void yuv420p_to_bgr24(unsigned char *yuvbuffer, unsigned char *rgbbuffer, int width, int height);
extern void rgb_swap_order(uchar *src, uchar *dst, int width, int height);

extern void rgb2hsv(uchar *src, uchar *dst, int width, int height);
extern void bgr2hsv(uchar *src, uchar *dst, int width, int height);
extern void hsv2rgb(uchar *src, uchar *dst, int width, int height);
extern void hsv2bgr(uchar *src, uchar *dst, int width, int height);

namespace KCV {
// 0.299R + 0.587G + 0.114B
// GRAY = 76R + 150G + 29B
static void do_bgr2gray(int cols, int rows, uchar *src, uchar *dst)
{
    int r, g, b;
    int count = cols * rows;
    while (count-->0) {
        b = *src++;
        g = *src++;
        r = *src++;
        *dst++ = (uchar)((76 * r + 150 * g + 29 * b) >> 8);
    }
}

static void do_rgb2gray(int cols, int rows, uchar *src, uchar *dst)
{
    int r, g, b;
    int count = cols * rows;
    while (count-->0) {
        r = *src++;
        g = *src++;
        b = *src++;
        *dst++ = (uchar)((76 * r + 150 * g + 29 * b) >> 8);
    }
}

static void do_gray2bgr_rgb(int cols, int rows, uchar *src, uchar *dst)
{
    int r, g, b;
    int count = cols * rows;
    while (count-->0) {
        *dst++ = *src;
        *dst++ = *src;
        *dst++ = *src++;
    }
}

#define COLOR_RGB   0
#define COLOR_BGR   1

static void do_bgr2yuv_i420(uchar *src, uchar *dst, int cols, int rows)
{
    rgb24_to_yuv420 (cols, rows, src, dst, COLOR_BGR);
}

static void do_rgb2yuv_i420(uchar *src, uchar *dst, int cols, int rows)
{
    rgb24_to_yuv420 (cols, rows, src, dst, COLOR_RGB);
}

static void do_yuv2bgr_i420(uchar *src, uchar *dst, int cols, int rows)
{
    yuv420p_to_bgr24(src, dst, cols, rows);
}

static void do_yuv2rgb_i420(uchar *src, uchar *dst, int cols, int rows)
{
    yuv420p_to_rgb24(src, dst, cols, rows);
}

// Color conversion
void cvtColor(const Mat src, Mat &dst , int code)
{
    dst.cols = src.cols;
    dst.rows = src.rows;

    switch (code) {
    	case CV_BGR2GRAY:
            dst.type = CV_8UC1;
            dst.createBuffer();
            do_bgr2gray(src.cols, src.rows, (uchar *)src.ref->data, (uchar *)dst.ref->data);
            break;

        case CV_RGB2GRAY:
            dst.type = CV_8UC1;
            dst.createBuffer();
            do_rgb2gray(src.cols, src.rows, (uchar *)src.ref->data, (uchar *)dst.ref->data);
            break;

	    case CV_GRAY2BGR:
        case CV_GRAY2RGB:
            dst.type = CV_8UC3;
            dst.createBuffer();
            do_gray2bgr_rgb(src.cols, src.rows, (uchar *)src.ref->data, (uchar *)dst.ref->data);
            break;

	    case CV_BGR2YUV_I420:
            dst.type = CV_8UC1;
            dst.rows = (src.rows * 3) / 2;
            dst.createBuffer();
            do_bgr2yuv_i420((uchar *)src.ref->data, (uchar *)dst.ref->data, src.cols, src.rows);
            break;

        case CV_RGB2YUV_I420:
            dst.type = CV_8UC1;
            dst.rows = (src.rows * 3) / 2;
            dst.createBuffer();
            do_rgb2yuv_i420((uchar *)src.ref->data, (uchar *)dst.ref->data, src.cols, src.rows);
            break;

	    case CV_YUV2BGR_I420:
            dst.type = CV_8UC3;
            dst.rows = (src.rows * 2) / 3; 
            dst.createBuffer();
            do_yuv2bgr_i420((uchar *)src.ref->data, (uchar *)dst.ref->data, src.cols, dst.rows);
            break;

        case CV_YUV2RGB_I420:
            dst.type = CV_8UC3;
            dst.rows = (src.rows * 2) / 3; 
            dst.createBuffer();
            do_yuv2rgb_i420((uchar *)src.ref->data, (uchar *)dst.ref->data, src.cols, dst.rows);
            break;

        case CV_RGB2BGR:
        case CV_BGR2RGB:
            dst.type = CV_8UC3;
            dst.createBuffer();
            rgb_swap_order((uchar *)src.ref->data, (uchar *)dst.ref->data, dst.cols, dst.rows);
            break;

	    case CV_BGR2HSV:
            dst.type = CV_8UC3;
            dst.createBuffer();
            bgr2hsv((uchar *)src.ref->data, (uchar *)dst.ref->data, dst.cols, dst.rows);
            break;

    	case CV_RGB2HSV:
            dst.type = CV_8UC3;
            dst.createBuffer();
            rgb2hsv((uchar *)src.ref->data, (uchar *)dst.ref->data, dst.cols, dst.rows);
            break;

	    case CV_HSV2BGR:
            dst.type = CV_8UC3;
            dst.createBuffer();
            hsv2bgr((uchar *)src.ref->data, (uchar *)dst.ref->data, dst.cols, dst.rows);
            break;

	    case CV_HSV2RGB:
            dst.type = CV_8UC3;
            dst.createBuffer();
            hsv2rgb((uchar *)src.ref->data, (uchar *)dst.ref->data, dst.cols, dst.rows);
            break;

        default:
            break;
    }
}

} // end of namespace
