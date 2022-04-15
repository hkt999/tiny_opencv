#include "tiny_opencv.hpp"
#include <cstring>

namespace KCV {

void clear_boundary(Mat &dst, Size ksize)
{
    uchar *wline;
    int h_pad = (ksize.width - 1) / 2;
    int v_pad = (ksize.height - 1) / 2;
    int chans = dst.channels();
    int line_bytes = dst.cols * chans;
    int side_bytes = h_pad * chans;

    // top
    memset(dst.ref->data, 0, line_bytes * v_pad);

    // bottom
    memset((uchar *)dst.ref->data + line_bytes * (dst.rows-v_pad), 0, line_bytes * v_pad);

    // left & right
    uchar *wline_left = (uchar *)dst.ref->data + line_bytes * v_pad;
    uchar *wline_right = wline_left + line_bytes - side_bytes;
    for (int y=v_pad; y<dst.rows-v_pad; y++) {
        memset(wline_left, 0, side_bytes);
        memset(wline_right, 0, side_bytes);
        wline_left += line_bytes;
        wline_right += line_bytes;
    }
}

}