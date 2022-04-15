#include "tiny_opencv.hpp"
#include "string.h"
#include <cstdlib>

namespace KCV {

extern void clear_boundary(Mat &dst, Size ksize);

static int uchar_compare(const void *a, const void *b)
{
    return *(uchar *)a < *(uchar *)b;
}

void medianBlur(const Mat src, Mat &dst, int ksize)
{
    int pad = (ksize-1)/2;

    dst.cols = src.cols;
    dst.rows = src.rows;
    dst.type = src.type;
    dst.createBuffer();

    int total = ksize * ksize;
    int chans = dst.channels();
    uchar *m = (uchar *)malloc( total * chans );

    // for normal case
    uchar *wline = (uchar *)dst.ref->data + (pad * src.cols + pad) * chans;
    int line_bytes = dst.cols * chans;
    for (int y=pad; y<src.rows-pad; y++) {
        uchar *w = wline;
        int count = 0;
        uchar **pb = (uchar **)malloc(chans*sizeof(uchar **));
        // run the kernel
        for (int x=pad; x<src.cols-pad; x++) {
            uchar *rline = (uchar *)src.ref->data + ((y-pad) * src.cols + x-pad) * chans;

            // reset color space pointer
            for (int i=0; i<chans; i++) {
                pb[i] = m + i*total;
            }
            for (int by=-pad; by<=pad; by++) {
                uchar *r = rline;
                for (int bx=-pad; bx<=pad; bx++) {
                    for (int c=0; c<chans; c++) {
                        *pb[c]++ = *r++;
                    }
                }
                rline += line_bytes;
            }
            uchar *p = m;
            for (int c=0; c<chans; c++) {
                qsort(p, sizeof(uchar), total, uchar_compare);
                *w++ = p[total/2];
                p += total;
            }
        }
        free(pb);
        wline += line_bytes;
    }
    free(m);
    clear_boundary(dst, Size(ksize, ksize));
}

}