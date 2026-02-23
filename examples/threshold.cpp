#include "tiny_opencv.hpp"

using namespace KCV;

int main() {
    // Input must be CV_8UC1 (grayscale)
    Mat gray(128, 128, CV_8UC1);

    Mat binary;
    threshold(gray, binary, 128, 255, THRESH_BINARY);

    return 0;
}
