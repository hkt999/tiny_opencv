#include "tiny_opencv.hpp"
#include <cmath>

// kevin: very slow, non-optimized implementation
// single channel
namespace KCV {

static float distance(int x, int y, int i, int j) {
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

static double gaussian(float x, double sigma) {
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * M_PI * pow(sigma, 2));

}

static void apply_bilateral_filter_c1(Mat source, Mat &filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
    double iFiltered = 0;
    double wP = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;

    for(int i = 0; i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            neighbor_x = x - (half - i);
            neighbor_y = y - (half - j);
            double gi = gaussian(source.at<uchar>(neighbor_y, neighbor_x) - source.at<uchar>(x, y), sigmaI);
            double gs = gaussian(distance(x, y, neighbor_y, neighbor_x), sigmaS);
            double w = gi * gs;
            iFiltered = iFiltered + source.at<uchar>(neighbor_y, neighbor_x) * w;
            wP = wP + w;
        }
    }
    iFiltered = iFiltered / wP;
    uchar *p = (uchar *)filteredImage.ref->data;
    p += y * filteredImage.cols + x;
    *p = (uchar)iFiltered;
}

static void apply_bilateral_filter_c3(Mat source, Mat &filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
    double iFiltered_c1 = 0;
    double iFiltered_c2 = 0;
    double iFiltered_c3 = 0;
    double wP1 = 0;
    double wP2 = 0;
    double wP3 = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;

    for(int i = 0; i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            neighbor_x = x - (half - i);
            neighbor_y = y - (half - j);
            uchar *addr_neighbor = (uchar *)source.ref->data + (neighbor_y * source.cols + neighbor_x) * 3;
            uchar *addr_src = (uchar *)source.ref->data + (y * source.cols + x) * 3;
            double gi_c1 = gaussian(*addr_neighbor++ - *addr_src++, sigmaI);
            double gi_c2 = gaussian(*addr_neighbor++ - *addr_src++, sigmaI);
            double gi_c3 = gaussian(*addr_neighbor++ - *addr_src++, sigmaI);
            double gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
            double w1 = gi_c1 * gs;
            double w2 = gi_c2 * gs;
            double w3 = gi_c3 * gs;
            addr_src = (uchar *)source.ref->data + (y * source.cols + x) * 3;
            iFiltered_c1 += *addr_src++ * w1;
            iFiltered_c2 += *addr_src++ * w2;
            iFiltered_c3 += *addr_src++ * w3;
            wP1 = wP1 + w1;
            wP2 = wP2 + w2;
            wP3 = wP3 + w3;
        }
    }
    iFiltered_c1 = iFiltered_c1 / wP1;
    iFiltered_c2 = iFiltered_c2 / wP2;
    iFiltered_c3 = iFiltered_c3 / wP3;

    uchar *p = (uchar *)filteredImage.ref->data;
    p += (y * filteredImage.cols + x) * 3;
    *p++ = (uchar)iFiltered_c1;
    *p++ = (uchar)iFiltered_c2;
    *p++ = (uchar)iFiltered_c3;
}

static void do_bilateralFilter_c1( const Mat src, Mat &dst, int d, double sigmaColor, double sigmaSpace)
{
    int width = src.cols;
    int height = src.rows;

    for(int i = 2; i < height - 2; i++) {
        for(int j = 2; j < width - 2; j++) {
            apply_bilateral_filter_c1(src, dst, i, j, d, sigmaColor, sigmaSpace);
        }
    }
}

static void do_bilateralFilter_c3( const Mat src, Mat &dst, int d, double sigmaColor, double sigmaSpace)
{
    int width = src.cols;
    int height = src.rows;

    for(int i = 2; i < height - 2; i++) {
        for(int j = 2; j < width - 2; j++) {
            apply_bilateral_filter_c3(src, dst, i, j, d, sigmaColor, sigmaSpace);
        }
    }
}

void bilateralFilter(const Mat src, Mat &dst, int d, double sigmaColor, double sigmaSpace)
{
    dst.rows = src.rows;
    dst.cols = src.cols;
    dst.type = src.type;
    dst.createBuffer();

    switch (src.type) {
        case CV_8UC1:
            do_bilateralFilter_c1(src, dst, d, sigmaColor, sigmaSpace);
            break;

        case CV_8UC3:
            do_bilateralFilter_c3(src, dst, d, sigmaColor, sigmaSpace);
            break;

        default:
            dst.release();
            dst.rows = dst.cols = 0;
    }
}

} // end of namespace
