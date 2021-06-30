#include "opencv.hpp"

namespace KCV {

template <typename T>
static void nearest_resize(T *src, int w1, int h1, T *dst, int w2, int h2)
{
    int x_step = w1 / w2;
    int x_err = (w1 * 65536) / w2 - x_step * 65536;
    int y_step = h1 / h2;
    int y_err = (h1 * 65536) / h2 - y_step * 65536;

    int accu_y = 0;
    int accu_y_err = 0;
    T *line_start = src;
    for (int y=0; y<h2; y++) {
        int accu_x_err = 0;
        T *r = line_start;
        for (int x=0; x<w2; x++) {
            *dst++ = *r;
            r += x_step;
            accu_x_err += x_err;
            if (accu_x_err > 65536) {
                r++;
                accu_x_err -= 65536;
            }
        }
        line_start += y_step * w1;
        accu_y_err += y_err;
        if (accu_y_err > 65536) {
            line_start += w1;
            accu_y_err -= 65536;
        }
    }
}


void resize(const Mat src, Mat &dst, Size size, float h_ratio, float v_ratio, int mode)
{
    dst.type = src.type;
    if (size.empty()) {
        dst.cols = src.cols * h_ratio;
        dst.rows = src.rows * v_ratio;
    } else {
        dst.cols = size.width;
        dst.rows = size.height;
    }

    switch (mode) {
        case INTER_NEAREST:
            switch (src.type) {
                case CV_8UC1:
                    dst.createBuffer();
                    nearest_resize( (cv8uc1_t *) src.data, src.cols, src.rows, (cv8uc1_t *)dst.data, size.width, size.height);
                    break;

                case CV_8UC3:
                    dst.createBuffer();
                    nearest_resize( (cv8uc3_t *) src.data, src.cols, src.rows, (cv8uc3_t *)dst.data, size.width, size.height);
                    break;
            }
            break;

        case INTER_LINEAR:
            break;

        case INTER_CUBIC:
            break;
    }
    if (dst.data == 0) {
        dst.cols = dst.rows = 0;
    }
}

} // end of namespace