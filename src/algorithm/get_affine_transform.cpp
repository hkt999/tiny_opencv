#include "tiny_opencv.hpp"

namespace KCV {

// 
// A = a00, a01
//     a10, a11
// B = b00
//     b10
// M = [A B]

Mat getAffineTransform(const Point2f src[], const Point2f dst[])
{
    Mat M(2, 3, CV_32FC1), X(6, 1, CV_32FC1, M.ptr());
    float a[6*6], b[6];
    Mat A(6, 6, CV_32FC1, a), B(6, 1, CV_32FC1, b);

    for (int i = 0; i < 3; i++) {
        int j = i*12;
        int k = i*12+6;
        a[j] = a[k+3] = src[i].x;
        a[j+1] = a[k+4] = src[i].y;
        a[j+2] = a[k+5] = 1;
        a[j+3] = a[j+4] = a[j+5] = 0;
        a[k] = a[k+1] = a[k+2] = 0;
        b[i*2] = dst[i].x;
        b[i*2+1] = dst[i].y;
    }

    X = A.inverse() * B;

    return M;
}

} // namespace KCV
