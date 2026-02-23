#include "tiny_opencv.hpp"

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef MAX3
#define MAX3(a,b,c) MAX(MAX(a,b),c)
#endif

#ifndef MIN3
#define MIN3(a,b,c) MIN(MIN(a,b),c)
#endif

// TODO: make the calculation fixed point
static void convertHSVtoRGB(const float h, const float s, const float v, unsigned char * r, unsigned char * g, unsigned char * b)
{
    /* Normalize hue so 360 maps to 0, then convert to 0-6 space */
    float h_norm = h;
    while (h_norm >= 360.0f) {
        h_norm -= 360.0f;
    }
    while (h_norm < 0.0f) {
        h_norm += 360.0f;
    }
    const float hex = h_norm / 60.0f;
    const unsigned char primary = (int) hex;
    const float secondary = hex - primary;

    /* a, b, c */
    const float x = (1.0 - s) * v;
    const float y = (1.0 - (s * secondary)) * v;
    const float z = (1.0 - (s * (1.0 - secondary))) * v;

    if(primary == 0) {
        /* 0: R = v, G = c, B = a */
        *r = (v * 255.0) + 0.5;
        *g = (z * 255.0) + 0.5;
        *b = (x * 255.0) + 0.5;
    } else if(primary == 1) {
        /* 1: R = b, G = v, B = a */
        *r = (y * 255.0) + 0.5;
        *g = (v * 255.0) + 0.5;
        *b = (x * 255.0) + 0.5;
    } else if(primary == 2) {
        /* 2: R = a, G = v, B = c */
        *r = (x * 255.0) + 0.5;
        *g = (v * 255.0) + 0.5;
        *b = (z * 255.0) + 0.5;
    } else if(primary == 3) {
        /* 3: R = a, G = b, B = v */
        *r = (x * 255.0) + 0.5;
        *g = (y * 255.0) + 0.5;
        *b = (v * 255.0) + 0.5;
    } else if(primary == 4) {
        /* 4: R = c, G = a, B = v */
        *r = (z * 255.0) + 0.5;
        *g = (x * 255.0) + 0.5;
        *b = (v * 255.0) + 0.5;
    } else if(primary == 5) {
        /* 5: R = v, G = a, B = b */
        *r = (v * 255.0) + 0.5;
        *g = (x * 255.0) + 0.5;
        *b = (y * 255.0) + 0.5;
    } else {
        /* Should not happen after hue normalization */
        *r = (v * 255.0) + 0.5;
        *g = (v * 255.0) + 0.5;
        *b = (v * 255.0) + 0.5;
    }
}

static void convertRGBtoHSV(const unsigned char r, const unsigned char g, const unsigned char b, float * h, float * s, float * v)
{
    const unsigned char max = MAX3(r, g, b);
    const float max2 = max / 255.0;
    const unsigned char min = MIN3(r, g, b);
    const float min2 = min / 255.0;

    *s = (max2 < 0.0001) ? 0 : (max2 - min2) / max2;
    *v = max2;

    /* Saturation is 0 */
    if((*s * 100.0) < 0.1) {
        /* Hue is undefined, monochrome */
        *h = 0;
        return;
    } else if(r == max) {
        if(g == min) {
            /* H = 5 + B' */
            *h = 5 + ((max2 - (b / 255.0)) / (max2 - min2));
        } else {
            /* H = 1 - G' */
            *h = 1 - ((max2 - (g / 255.0)) / (max2 - min2));
        }
    }
    else if(g == max) {
        if(b == min) {
            /* H = R' + 1 */
            *h = ((max2 - (r / 255.0)) / (max2 - min2)) + 1;
        } else {
            /* H = 3 - B' */
            *h = 3 - ((max2 - (b / 255.0)) / (max2 - min2));
        }
    } else if(b == max && r == min) {
        /* H = 3 + G' */
        *h = 3 + ((max2 - (g / 255.0)) / (max2 - min2));
    } else {
        /* H = 5 - R' */
        *h = 5 - ((max2 - (r / 255.0)) / (max2 - min2));
    }

    /* Hue is then converted to degrees by multiplying by 60 */
    *h = MIN(*h * 60, 360);
}

void rgb2hsv(uchar *src, uchar *dst, int width, int height)
{
    float h, s, v;
    int count = width * height;
    while (count-->0) {
        convertRGBtoHSV(src[0], src[1], src[2], &h, &s, &v);
        dst[0] = (uchar)(255.0f * h / 360.0f);
        dst[1] = (uchar)(255.0f * s);
        dst[2] = (uchar)(255.0f * v);
        src += 3;
        dst += 3;
    }
}

void bgr2hsv(uchar *src, uchar *dst, int width, int height)
{
    float h, s, v;
    int count = width * height;
    while (count-->0) {
        convertRGBtoHSV(src[2], src[1], src[0], &h, &s, &v);
        dst[0] = (uchar)(255.0f * h / 360.0f);
        dst[1] = (uchar)(255.0f * s);
        dst[2] = (uchar)(255.0f * v);
        src += 3;
        dst += 3;
    }
}

void hsv2rgb(uchar *src, uchar *dst, int width, int height)
{
    int count = width * height;
    while (count-->0) {
        float h = (src[0] * 360.0)/255.0;
        float s = src[1] / 255.0f;
        float v = src[2] / 255.0f;
        convertHSVtoRGB(h, s, v, &dst[0], &dst[1], &dst[2]);
        src += 3;
        dst += 3;
    }
}

void hsv2bgr(uchar *src, uchar *dst, int width, int height)
{
    int count = width * height;
    while (count-->0) {
        float h = (src[0] * 360.0)/255.0;
        float s = src[1] / 255.0f;
        float v = src[2] / 255.0f;
        convertHSVtoRGB(h, s, v, &dst[2], &dst[1], &dst[0]);
        src += 3;
        dst += 3;
    }
}
