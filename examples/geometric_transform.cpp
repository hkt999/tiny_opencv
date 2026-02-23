#include "tiny_opencv.hpp"

using namespace KCV;

int main() {
    Point2f center(100.0f, 80.0f);
    Mat rot = getRotationMatrix2D(center, 45.0, 1.0); // 2x3 rotation matrix
    (void)rot;

    Point2f src[3] = { Point2f(0, 0), Point2f(100, 0), Point2f(0, 50) };
    Point2f dst[3] = { Point2f(10, 20), Point2f(120, 10), Point2f(15, 80) };
    Mat affine = getAffineTransform(src, dst); // 2x3 affine matrix
    (void)affine;

    return 0;
}
