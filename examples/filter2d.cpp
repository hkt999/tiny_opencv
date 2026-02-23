#include "tiny_opencv.hpp"

using namespace KCV;

int main() {
    // Input should be CV_8UC1 or CV_8UC3
    Mat src(64, 64, CV_8UC1);

    // 3x3 sharpening kernel
    Mat kernel(3, 3, CV_32FC1);
    kernel.at<float>(0, 0) = 0;  kernel.at<float>(0, 1) = -1; kernel.at<float>(0, 2) = 0;
    kernel.at<float>(1, 0) = -1; kernel.at<float>(1, 1) = 5;  kernel.at<float>(1, 2) = -1;
    kernel.at<float>(2, 0) = 0;  kernel.at<float>(2, 1) = -1; kernel.at<float>(2, 2) = 0;

    Mat filtered;
    filter2D(src, filtered, CV_8UC1, kernel);

    return 0;
}
