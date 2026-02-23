#include "tiny_opencv.hpp"
#include <cmath>

namespace KCV {

template <typename T>
static void nearest_resize(T *src, int w1, int h1, T *dst, int w2, int h2)
{
    int x_step = w1 / w2;
    int x_err = (w1 * 65536) / w2 - x_step * 65536;
    int y_step = h1 / h2;
    int y_err = (h1 * 65536) / h2 - y_step * 65536;

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

static void linear_resize_u8c1(uchar *src, int w1, int h1, uchar *dst, int w2, int h2)
{
    if (w2 == 1 && h2 == 1) {
        dst[0] = src[0];
        return;
    }

    float x_scale = (w2 > 1) ? (float)(w1 - 1) / (float)(w2 - 1) : 0.0f;
    float y_scale = (h2 > 1) ? (float)(h1 - 1) / (float)(h2 - 1) : 0.0f;
    for (int y = 0; y < h2; y++) {
        float fy = y * y_scale;
        int y0 = (int)fy;
        int y1 = MIN(y0 + 1, h1 - 1);
        float wy = fy - y0;
        for (int x = 0; x < w2; x++) {
            float fx = x * x_scale;
            int x0 = (int)fx;
            int x1 = MIN(x0 + 1, w1 - 1);
            float wx = fx - x0;

            int p00 = src[y0 * w1 + x0];
            int p01 = src[y0 * w1 + x1];
            int p10 = src[y1 * w1 + x0];
            int p11 = src[y1 * w1 + x1];

            float top = p00 + (p01 - p00) * wx;
            float bottom = p10 + (p11 - p10) * wx;
            int val = (int)roundf(top + (bottom - top) * wy);
            dst[y * w2 + x] = (uchar)MIN(MAX(val, 0), 255);
        }
    }
}

static void linear_resize_u8c3(cv8uc3_t *src, int w1, int h1, cv8uc3_t *dst, int w2, int h2)
{
    float x_scale = (w2 > 1) ? (float)(w1 - 1) / (float)(w2 - 1) : 0.0f;
    float y_scale = (h2 > 1) ? (float)(h1 - 1) / (float)(h2 - 1) : 0.0f;
    for (int y = 0; y < h2; y++) {
        float fy = y * y_scale;
        int y0 = (int)fy;
        int y1 = MIN(y0 + 1, h1 - 1);
        float wy = fy - y0;
        for (int x = 0; x < w2; x++) {
            float fx = x * x_scale;
            int x0 = (int)fx;
            int x1 = MIN(x0 + 1, w1 - 1);
            float wx = fx - x0;

            cv8uc3_t p00 = src[y0 * w1 + x0];
            cv8uc3_t p01 = src[y0 * w1 + x1];
            cv8uc3_t p10 = src[y1 * w1 + x0];
            cv8uc3_t p11 = src[y1 * w1 + x1];

            int vals[3] = {0, 0, 0};
            float c00[3] = {(float)p00.c1, (float)p00.c2, (float)p00.c3};
            float c01[3] = {(float)p01.c1, (float)p01.c2, (float)p01.c3};
            float c10[3] = {(float)p10.c1, (float)p10.c2, (float)p10.c3};
            float c11[3] = {(float)p11.c1, (float)p11.c2, (float)p11.c3};
            for (int c = 0; c < 3; c++) {
                float top = c00[c] + (c01[c] - c00[c]) * wx;
                float bottom = c10[c] + (c11[c] - c10[c]) * wx;
                vals[c] = (int)roundf(top + (bottom - top) * wy);
                vals[c] = MIN(MAX(vals[c], 0), 255);
            }
            cv8uc3_t &d = dst[y * w2 + x];
            d.c1 = (uchar)vals[0];
            d.c2 = (uchar)vals[1];
            d.c3 = (uchar)vals[2];
        }
    }
}


void resize(const Mat src, Mat &dst, Size size, float h_ratio, float v_ratio, int mode)
{
    if (src.empty()) {
        throw Exception("resize: input image is empty");
    }
    dst.type = src.type;
    if (size.empty()) {
        if (h_ratio <= 0.0f || v_ratio <= 0.0f) {
            throw Exception("resize: invalid scale ratio");
        }
        dst.cols = src.cols * h_ratio;
        dst.rows = src.rows * v_ratio;
    } else {
        dst.cols = size.width;
        dst.rows = size.height;
    }
    if (dst.cols <= 0 || dst.rows <= 0) {
        throw Exception("resize: output size must be positive");
    }

    switch (mode) {
        case INTER_NEAREST:
            switch (src.type) {
                case CV_8UC1:
                    dst.createBuffer();
                    nearest_resize((cv8uc1_t *) src.ref->data, src.cols, src.rows, (cv8uc1_t *)dst.ref->data, dst.cols, dst.rows);
                    break;

                case CV_8UC3:
                    dst.createBuffer();
                    nearest_resize((cv8uc3_t *) src.ref->data, src.cols, src.rows, (cv8uc3_t *)dst.ref->data, dst.cols, dst.rows);
                    break;

                default:
                    throw Exception("resize: only CV_8UC1 and CV_8UC3 are supported");
            }
            break;

        case INTER_LINEAR:
            switch (src.type) {
                case CV_8UC1:
                    dst.createBuffer();
                    linear_resize_u8c1((uchar *)src.ref->data, src.cols, src.rows, (uchar *)dst.ref->data, dst.cols, dst.rows);
                    break;

                case CV_8UC3:
                    dst.createBuffer();
                    linear_resize_u8c3((cv8uc3_t *)src.ref->data, src.cols, src.rows, (cv8uc3_t *)dst.ref->data, dst.cols, dst.rows);
                    break;

                default:
                    throw Exception("resize: only CV_8UC1 and CV_8UC3 are supported");
            }
            break;

        case INTER_CUBIC:
            throw Exception("resize: INTER_CUBIC is not supported");
            break;

        default:
            throw Exception("resize: unknown interpolation mode");
    }
    if (dst.ref == 0 || dst.ref->data == 0) {
        dst.cols = dst.rows = 0;
    }
}

} // end of namespace
