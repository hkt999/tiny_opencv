#include "tiny_opencv.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace KCV {

void HoughLines(const Mat src, std::vector<HoughLine> &lines, double rho, double theta, int threshold, double minLineLength, double maxLineGap) {
    lines.clear();
    
    if (src.empty() || src.channels() != 1) {
        throw Exception("HoughLines: Input image must be single channel (grayscale)");
    }
    
    const int width = src.cols;
    const int height = src.rows;
    const uchar* data = src.getData<uchar>();
    
    // Calculate accumulator dimensions
    const double maxRho = sqrt(width * width + height * height);
    const int rhoSteps = (int)(2 * maxRho / rho) + 1;
    const int thetaSteps = (int)(M_PI / theta);
    
    // Create accumulator array
    std::vector<std::vector<int>> accumulator(rhoSteps, std::vector<int>(thetaSteps, 0));
    
    // Precompute sin and cos values
    std::vector<double> cosTable(thetaSteps);
    std::vector<double> sinTable(thetaSteps);
    for (int t = 0; t < thetaSteps; t++) {
        double angle = t * theta;
        cosTable[t] = cos(angle);
        sinTable[t] = sin(angle);
    }
    
    // Vote in Hough space for each edge pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (data[y * width + x] > 0) { // Edge pixel
                for (int t = 0; t < thetaSteps; t++) {
                    double r = x * cosTable[t] + y * sinTable[t];
                    int rhoIdx = (int)((r + maxRho) / rho + 0.5);
                    
                    if (rhoIdx >= 0 && rhoIdx < rhoSteps) {
                        accumulator[rhoIdx][t]++;
                    }
                }
            }
        }
    }
    
    // Find peaks in accumulator
    for (int r = 0; r < rhoSteps; r++) {
        for (int t = 0; t < thetaSteps; t++) {
            if (accumulator[r][t] >= threshold) {
                double rhoVal = (r * rho) - maxRho;
                double thetaVal = t * theta;
                
                HoughLine line(rhoVal, thetaVal, accumulator[r][t]);
                
                // Calculate line endpoints
                double cosTheta = cos(thetaVal);
                double sinTheta = sin(thetaVal);
                
                if (fabs(sinTheta) > fabs(cosTheta)) {
                    // More vertical line
                    double x0 = rhoVal / cosTheta;
                    line.start = Point2f(x0, 0);
                    line.end = Point2f(x0, height - 1);
                    
                    if (cosTheta != 0) {
                        line.start.x = (rhoVal - 0 * sinTheta) / cosTheta;
                        line.end.x = (rhoVal - (height - 1) * sinTheta) / cosTheta;
                    }
                } else {
                    // More horizontal line
                    double y0 = rhoVal / sinTheta;
                    line.start = Point2f(0, y0);
                    line.end = Point2f(width - 1, y0);
                    
                    if (sinTheta != 0) {
                        line.start.y = (rhoVal - 0 * cosTheta) / sinTheta;
                        line.end.y = (rhoVal - (width - 1) * cosTheta) / sinTheta;
                    }
                }
                
                // Clip line to image bounds
                if (line.start.x < 0) {
                    if (cosTheta != 0) {
                        line.start.y = (rhoVal - 0 * cosTheta) / sinTheta;
                        line.start.x = 0;
                    }
                }
                if (line.start.x >= width) {
                    if (cosTheta != 0) {
                        line.start.y = (rhoVal - (width - 1) * cosTheta) / sinTheta;
                        line.start.x = width - 1;
                    }
                }
                if (line.start.y < 0) {
                    if (sinTheta != 0) {
                        line.start.x = (rhoVal - 0 * sinTheta) / cosTheta;
                        line.start.y = 0;
                    }
                }
                if (line.start.y >= height) {
                    if (sinTheta != 0) {
                        line.start.x = (rhoVal - (height - 1) * sinTheta) / cosTheta;
                        line.start.y = height - 1;
                    }
                }
                
                if (line.end.x < 0) {
                    if (cosTheta != 0) {
                        line.end.y = (rhoVal - 0 * cosTheta) / sinTheta;
                        line.end.x = 0;
                    }
                }
                if (line.end.x >= width) {
                    if (cosTheta != 0) {
                        line.end.y = (rhoVal - (width - 1) * cosTheta) / sinTheta;
                        line.end.x = width - 1;
                    }
                }
                if (line.end.y < 0) {
                    if (sinTheta != 0) {
                        line.end.x = (rhoVal - 0 * sinTheta) / cosTheta;
                        line.end.y = 0;
                    }
                }
                if (line.end.y >= height) {
                    if (sinTheta != 0) {
                        line.end.x = (rhoVal - (height - 1) * sinTheta) / cosTheta;
                        line.end.y = height - 1;
                    }
                }
                
                // Check line length if specified
                if (minLineLength > 0) {
                    double dx = line.end.x - line.start.x;
                    double dy = line.end.y - line.start.y;
                    double length = sqrt(dx * dx + dy * dy);
                    if (length < minLineLength) {
                        continue;
                    }
                }
                
                lines.push_back(line);
            }
        }
    }
    
    // Sort lines by vote count (descending)
    std::sort(lines.begin(), lines.end(), [](const HoughLine& a, const HoughLine& b) {
        return a.votes > b.votes;
    });
}

} // namespace KCV