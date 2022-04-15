#include "tiny_opencv.hpp"
#include <string.h>

namespace KCV {
void equalizeHist(const Mat src, Mat &dst)
{
    if (src.type != CV_8UC1)
        return; // only support one channel

    unsigned int hist[256];
    memset(hist, 0, sizeof(hist));

    int count = src.cols * src.rows;
    uchar *p = (uchar *)src.ref->data;
    while (count-->0) {
        hist[*p++]++;
    }

    // accumerate the histogram
    unsigned int accu = 0;
    for (int i=0; i<256; i++) {
        hist[i] += accu;
        accu = hist[i];
    }

    // normalization
    for (int i=0; i<256; i++) {
        hist[i] = (255*hist[i])/accu;
    }

    dst.cols = src.cols;
    dst.rows = src.rows;
    dst.type = src.type;
    dst.createBuffer();

    count = src.cols * src.rows;
    p = (uchar *)src.ref->data;
    uchar *w = (uchar *)dst.ref->data;
    while (count-->0) {
        *w++ = hist[*p++];
    }
}

} // end of namespace