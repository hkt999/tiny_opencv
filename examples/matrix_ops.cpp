#include "tiny_opencv.hpp"

using namespace KCV;

int main() {
    // Create identity matrix
    Mat identity = Mat::eye(3, 3, CV_32FC1);
    (void)identity;

    Mat a = Mat::eye(3, 3, CV_32FC1);
    Mat b = Mat::eye(3, 3, CV_32FC1);

    // Matrix multiplication and inversion
    Mat result = a * b;
    Mat inverted = a.inverse();
    (void)result;
    (void)inverted;

    // Element access
    float value = a.at<float>(0, 0);
    (void)value;

    return 0;
}
