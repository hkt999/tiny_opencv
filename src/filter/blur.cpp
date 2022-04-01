#include "tiny_opencv.hpp"
#include <string.h>

namespace KCV {

enum {
    TOP_BOUNDARY = 0,
    BOTTOM_BOUNDARY,
    LEFT_BOUNDARY,
    RIGHT_BOUNDARY
};

#define MAX_CHAN    4
static void boundary(const Mat &src, Mat &dst, int h_pad, int v_pad, int chans, int mode)
{
    uchar *wline, *rline;
    int start_y, end_y;
    int start_x, end_x;
    int value[MAX_CHAN];

    switch (mode) {
        case TOP_BOUNDARY:
            start_y = 0;
            end_y = v_pad;
            start_x = 0;
            end_x = dst.cols;
            break;

        case BOTTOM_BOUNDARY:
            start_y = dst.rows - v_pad;
            end_y = dst.rows;
            start_x = 0;
            end_x = dst.cols;
            break;

        case LEFT_BOUNDARY:
            start_y = v_pad;
            end_y = dst.rows - v_pad;
            start_x = 0;
            end_x = h_pad;
            break;

        case RIGHT_BOUNDARY:
            start_y = v_pad;
            end_y = dst.rows - v_pad;
            start_x = dst.cols - h_pad;
            end_x = dst.cols;
            break;
    }

    int line_bytes = dst.cols * chans;
    rline = (uchar *)src.data + (start_y * src.cols) * chans;
    wline = (uchar *)dst.data + (start_y * dst.cols) * chans;
    for (int y=start_y; y<end_y; y++) {
        uchar *w_local = wline + start_x * chans;
        for (int x=start_x; x<end_x; x++) {
            int count = 0;
            for (int c=0; c<chans; c++)
                value[c] = 0;

            uchar *r_start = rline + (x - v_pad * src.cols - h_pad) * chans;
            for (int ty=y-v_pad; ty<=y+v_pad; ty++) {
                if (ty < 0) {
                    r_start += line_bytes;
                    continue;
                }

                if (ty >= src.rows) {
                    r_start += line_bytes;
                    continue;
                }

                uchar *r_local = r_start;
                for (int tx=x-h_pad; tx<=x+h_pad; tx++) {
                    if (tx < 0) {
                        r_local += chans; 
                        continue;
                    }
                    if (tx >= src.cols) {
                        r_local += chans;
                        continue;
                    }
                    count++;
                    for (int c=0; c<chans; c++) {
                        value[c] += *r_local++;
                    }
                }
                r_start += line_bytes;
            }
            for (int c=0; c<chans; c++) {
                *w_local++ = value[c]/count;
            }
        }
        wline += line_bytes; 
        rline += line_bytes;
    }
}

static void do_blur(const Mat src, Mat &dst, Size ksize, int chans)
{
    int h_pad = (ksize.width-1)/2;
    int v_pad = (ksize.height-1)/2;
    int value[MAX_CHAN];

    dst.cols = src.cols;
    dst.rows = src.rows;
    dst.type = src.type;
    dst.createBuffer();

    int total = ksize.width * ksize.height;

    // for normal case
    uchar *wline = (uchar *)dst.data + (v_pad * src.cols + h_pad) * chans;
    int line_bytes = dst.cols * chans;
    for (int y=v_pad; y<src.rows-v_pad; y++) {
        uchar *w = wline;
        for (int x=h_pad; x<src.cols-h_pad; x++) {
            for (int c=0; c<chans; c++)
                value[c] = 0;
            uchar *rline = (uchar *)src.data + ((y-v_pad) * src.cols + x - h_pad) * chans;
            for (int by=-v_pad; by<=v_pad; by++) {
                uchar *r = rline;
                for (int bx=-h_pad; bx<=h_pad; bx++) {
                    for (int c=0; c<chans; c++)
                        value[c] += (int)(*r++);
                }
                rline += line_bytes;
            }
            for (int c=0; c<chans; c++) {
                *w++ = (uchar)(value[c]/total);
            }
        }
        wline += line_bytes;
    }

    boundary(src, dst, h_pad, v_pad, chans, TOP_BOUNDARY);
    boundary(src, dst, h_pad, v_pad, chans, BOTTOM_BOUNDARY);
    boundary(src, dst, h_pad, v_pad, chans, LEFT_BOUNDARY);
    boundary(src, dst, h_pad, v_pad, chans, RIGHT_BOUNDARY);
}

void blur(const Mat src, Mat &dst, Size ksize)
{
    switch (src.type) {
        case CV_8UC1:
            do_blur(src, dst, ksize, 1);
            break;

        case CV_8UC3:
            do_blur(src, dst, ksize, 3);
            break;
    }
}

} // end of namespace