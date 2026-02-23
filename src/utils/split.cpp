#include "tiny_opencv.hpp"
#include <cstring>

namespace KCV {

static int depth_size_bytes(int depth)
{
    switch (depth) {
        case CV_8U:
        case CV_8S:
            return 1;
        case CV_16U:
        case CV_16S:
            return 2;
        case CV_32S:
        case CV_32F:
            return 4;
        case CV_64F:
            return 8;
        default:
            return 0;
    }
}

void split(const Mat src, Mat *mv, int count)
{
    if (src.empty()) {
        throw Exception("split: input image is empty");
    }
    if (mv == 0) {
        throw Exception("split: destination channel array is null");
    }
    if (count <= 0 || count != src.channels()) {
        throw Exception("split: channel count mismatch");
    }

    int depth = src.type & 0x7;
    int elem_bytes = depth_size_bytes(depth);
    if (elem_bytes <= 0) {
        throw Exception("split: unsupported depth");
    }
    int pix_count = src.rows * src.cols;
    for (int c = 0; c < count; c++) {
        mv[c].create(src.rows, src.cols, depth);
    }

    const uchar *src_ptr = (const uchar *)src.ref->data;
    for (int i = 0; i < pix_count; i++) {
        for (int c = 0; c < count; c++) {
            uchar *dst_ptr = (uchar *)mv[c].ref->data + i * elem_bytes;
            memcpy(dst_ptr, src_ptr + c * elem_bytes, elem_bytes);
        }
        src_ptr += elem_bytes * src.channels();
    }
}

} // end of namespace
