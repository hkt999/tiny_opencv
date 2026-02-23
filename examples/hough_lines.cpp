#include "tiny_opencv.hpp"
#include <vector>

using namespace KCV;

int main() {
    // edges should be a binary image (0 or 255)
    Mat edges(100, 100, CV_8UC1);

    std::vector<HoughLine> lines;
    double rho = 1.0;
    double theta = 3.1415926 / 180.0; // 1 degree
    HoughLines(edges, lines, rho, theta, 50, 30, 10);

    return 0;
}
