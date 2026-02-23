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

void merge(const Mat *mv, int count, Mat &dst)
{
    if (mv == 0) {
        throw Exception("merge: source channel array is null");
    }
    if (count <= 0 || count > 4) {
        throw Exception("merge: invalid channel count");
    }

    const Mat &first = mv[0];
    if (first.empty()) {
        throw Exception("merge: channel 0 is empty");
    }
    if (first.channels() != 1) {
        throw Exception("merge: only single-channel inputs are supported");
    }

    int depth = first.type & 0x7;
    for (int c = 1; c < count; c++) {
        if (mv[c].empty()) {
            throw Exception("merge: input channel is empty");
        }
        if (mv[c].rows != first.rows || mv[c].cols != first.cols) {
            throw Exception("merge: channel shape mismatch");
        }
        if ((mv[c].type & 0x7) != depth || mv[c].channels() != 1) {
            throw Exception("merge: channel type mismatch");
        }
    }

    dst.create(first.rows, first.cols, depth + (count - 1) * 8);
    int elem_bytes = depth_size_bytes(depth);
    if (elem_bytes <= 0) {
        throw Exception("merge: unsupported depth");
    }
    int pix_count = first.rows * first.cols;
    uchar *dst_ptr = (uchar *)dst.ref->data;
    for (int i = 0; i < pix_count; i++) {
        for (int c = 0; c < count; c++) {
            const uchar *src_ptr = (const uchar *)mv[c].ref->data + i * elem_bytes;
            memcpy(dst_ptr, src_ptr, elem_bytes);
            dst_ptr += elem_bytes;
        }
    }
}

} // end of namespace
