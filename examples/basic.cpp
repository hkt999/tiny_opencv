#include "tiny_opencv.hpp"

using namespace KCV;

int main() {
    Mat img(480, 640, CV_8UC3);

    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);

    Mat blurred;
    gaussianBlur(img, blurred, Size(5, 5), 1.0);

    Mat resized;
    resize(img, resized, Size(320, 240));

    return 0;
}
