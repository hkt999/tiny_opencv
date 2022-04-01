#include "tiny_opencv.hpp"

namespace KCV {

void threshold(const Mat in, Mat &out, double thresh, double maxval, int type)
{
    if (in.type != CV_8UC1) // only support gray level
        return;

    out.rows = in.rows;
    out.cols = in.cols;
    out.type = in.type;
    out.createBuffer();
    int counter = out.rows * out.cols;
    uchar *src = (uchar *)in.data;
    uchar *dst = (uchar *)out.data;
    uchar th8 = (uchar) thresh;
    uchar max8 = (uchar) maxval;
    switch (type) {
        case THRESH_BINARY:
            while (counter-->0) {
                *dst++ = (*src++ > th8) ? max8 : 0;
            }
            break;

        case THRESH_BINARY_INV:
            while (counter-->0) {
                *dst++ = (*src++ > th8) ? 0 : max8;
            }
            break;

        case THRESH_TRUNC:
            while (counter-->0) {
                *dst++ = (*src > th8) ? th8 : *src;
                src++;
            }
            break;

        case THRESH_TOZERO:
            while (counter-->0) {
                *dst++ = (*src > th8) ? *src : 0;
                src++;
            }
            break;

        case THRESH_TOZERO_INV:
            while (counter-->0) {
                *dst++ = (*src > th8) ? 0 : *src;
                src++;
            }
            break;

        default:
            out.cols = 0;
            out.rows = 0;
            break;
    }
}

} // end of name space