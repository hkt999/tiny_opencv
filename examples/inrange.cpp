#include "tiny_opencv.hpp"

using namespace KCV;

int main() {
    Mat colorImage(200, 200, CV_8UC3);

    // Detect red blobs (BGR format: B=0-10, G=0-10, R=240-255)
    Mat redMask;
    inRange(colorImage, Scalar(0, 0, 240), Scalar(10, 10, 255), redMask);

    // Detect specific grayscale range (100-150)
    Mat grayImage, grayMask;
    cvtColor(colorImage, grayImage, CV_BGR2GRAY);
    inRange(grayImage, Scalar(100), Scalar(150), grayMask);

    return 0;
}
