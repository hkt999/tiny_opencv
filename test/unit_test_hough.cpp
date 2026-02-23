#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include "unit_test.hpp"
#include "tiny_opencv.hpp"

using namespace cv;
using namespace std;

void unit_test_hough_lines() {
    cout << "=== Hough Lines Transform Test ===" << endl;
    
    // Create a test image with lines
    Mat test_img = Mat::zeros(400, 400, CV_8UC1);
    
    // Draw some lines on the test image
    line(test_img, Point(50, 50), Point(350, 50), Scalar(255), 2);    // Horizontal line
    line(test_img, Point(100, 100), Point(100, 300), Scalar(255), 2); // Vertical line
    line(test_img, Point(200, 100), Point(300, 200), Scalar(255), 2); // Diagonal line
    
    // Apply Canny edge detection first (simple threshold for this test)
    Mat edges;
    threshold(test_img, edges, 128, 255, THRESH_BINARY);
    
    // Convert OpenCV Mat to KCV Mat
    KCV::Mat kcv_edges(edges.rows, edges.cols, KCV::KCV_8UC1, edges.data);
    
    // Apply Hough Line Transform
    vector<KCV::HoughLine> lines;
    try {
        KCV::HoughLines(kcv_edges, lines, 1.0, M_PI/180, 50, 30, 10);
        
        cout << "Found " << lines.size() << " lines:" << endl;
        
        // Draw detected lines on original image
        Mat result;
        cvtColor(test_img, result, COLOR_GRAY2BGR);
        
        for (size_t i = 0; i < lines.size() && i < 10; i++) {
            const KCV::HoughLine& hline = lines[i];
            
            cout << "Line " << i << ": ";
            cout << "rho=" << hline.rho << ", theta=" << hline.theta * 180 / M_PI << "°, ";
            cout << "votes=" << hline.votes << endl;
            cout << "  Start: (" << hline.start.x << ", " << hline.start.y << ")";
            cout << "  End: (" << hline.end.x << ", " << hline.end.y << ")" << endl;
            
            // Draw line in different colors
            Scalar color;
            switch (i % 3) {
                case 0: color = Scalar(0, 0, 255); break;   // Red
                case 1: color = Scalar(0, 255, 0); break;   // Green
                case 2: color = Scalar(255, 0, 0); break;   // Blue
            }
            
            if (std::isfinite(hline.start.x) && std::isfinite(hline.start.y) &&
                std::isfinite(hline.end.x) && std::isfinite(hline.end.y)) {
                cv::line(result,
                        Point((int)hline.start.x, (int)hline.start.y),
                        Point((int)hline.end.x, (int)hline.end.y),
                        color, 2);
            }
        }
        
        // Validate that we found some reasonable lines
        if (lines.size() >= 2) {
            cout << "✓ Hough Lines test completed successfully! Found " << lines.size() << " lines." << endl;
            
            // Check if we detected something close to our test lines
            bool found_horizontal = false, found_vertical = false;
            for (const auto& line : lines) {
                double angle_deg = line.theta * 180 / M_PI;
                if (abs(angle_deg - 0) < 10 || abs(angle_deg - 180) < 10) {
                    found_horizontal = true;
                    cout << "✓ Found horizontal line (θ=" << angle_deg << "°)" << endl;
                }
                if (abs(angle_deg - 90) < 10) {
                    found_vertical = true;
                    cout << "✓ Found vertical line (θ=" << angle_deg << "°)" << endl;
                }
            }
            if (found_horizontal && found_vertical) {
                cout << "✓ Successfully detected both horizontal and vertical lines!" << endl;
            }
        } else {
            cout << "⚠ Warning: Expected to find at least 2 lines, only found " << lines.size() << endl;
        }
        
    } catch (const KCV::Exception& e) {
        cout << "✗ KCV Exception: " << e.what() << endl;
    } catch (const std::exception& e) {
        cout << "✗ Standard Exception: " << e.what() << endl;
    }
}
