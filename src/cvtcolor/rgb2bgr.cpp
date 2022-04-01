#include "tiny_opencv.hpp"

void rgb_swap_order(uchar *src, uchar *dst, int width, int height)
{
    int counter = width * height;
    while (counter-->0) {
        *dst++ = src[2];
        *dst++ = src[1];
        *dst++ = src[0];
        src += 3;
    }
}