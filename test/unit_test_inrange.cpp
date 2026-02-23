#include <opencv2/opencv.hpp>
#include <iostream>
#include "unit_test.hpp"
#include "tiny_opencv.hpp"

using namespace cv;
using namespace std;

void unit_test_in_range() {
    cout << "=== inRange Color Blob Detection Test ===" << endl;
    
    // Test 1: Grayscale image
    cout << "Test 1: Grayscale inRange" << endl;
    Mat gray_test = Mat::zeros(100, 100, CV_8UC1);
    
    // Create some regions with different intensities
    rectangle(gray_test, Rect(10, 10, 30, 30), Scalar(100), -1);  // Medium gray
    rectangle(gray_test, Rect(50, 50, 30, 30), Scalar(200), -1);  // Light gray
    circle(gray_test, Point(25, 75), 15, Scalar(50), -1);         // Dark gray
    
    // Convert to KCV format
    KCV::Mat kcv_gray(gray_test.rows, gray_test.cols, KCV::KCV_8UC1, gray_test.data);
    KCV::Mat gray_result;
    
    try {
        // Test range detection for medium gray (90-110)
        KCV::inRange(kcv_gray, KCV::Scalar(90), KCV::Scalar(110), gray_result);
        
        // Count white pixels (should find the medium gray rectangle)
        uchar* result_data = gray_result.getData<uchar>();
        int white_pixels = 0;
        for (int i = 0; i < gray_result.rows * gray_result.cols; i++) {
            if (result_data[i] == 255) white_pixels++;
        }
        
        cout << "✓ Found " << white_pixels << " pixels in range [90, 110]" << endl;
        if (white_pixels >= 800 && white_pixels <= 1000) {  // ~30x30 = 900 pixels
            cout << "✓ Grayscale inRange test passed!" << endl;
        } else {
            cout << "⚠ Expected ~900 pixels, found " << white_pixels << endl;
        }
        
    } catch (const KCV::Exception& e) {
        cout << "✗ KCV Exception in grayscale test: " << e.what() << endl;
    }
    
    // Test 2: Color image (BGR)
    cout << "\nTest 2: Color inRange" << endl;
    Mat color_test = Mat::zeros(100, 100, CV_8UC3);
    
    // Create colored regions
    rectangle(color_test, Rect(10, 10, 30, 30), Scalar(0, 0, 255), -1);     // Red
    rectangle(color_test, Rect(50, 10, 30, 30), Scalar(0, 255, 0), -1);     // Green  
    rectangle(color_test, Rect(10, 50, 30, 30), Scalar(255, 0, 0), -1);     // Blue
    rectangle(color_test, Rect(50, 50, 30, 30), Scalar(0, 255, 255), -1);   // Yellow
    
    // Convert to KCV format
    KCV::Mat kcv_color(color_test.rows, color_test.cols, KCV::KCV_8UC3, color_test.data);
    KCV::Mat color_result;
    
    try {
        // Test range detection for red color (B=0, G=0, R=240-255)
        KCV::inRange(kcv_color, KCV::Scalar(0, 0, 240), KCV::Scalar(10, 10, 255), color_result);
        
        // Count white pixels (should find the red rectangle)
        uchar* result_data = color_result.getData<uchar>();
        int white_pixels = 0;
        for (int i = 0; i < color_result.rows * color_result.cols; i++) {
            if (result_data[i] == 255) white_pixels++;
        }
        
        cout << "✓ Found " << white_pixels << " red pixels in range" << endl;
        if (white_pixels >= 800 && white_pixels <= 1000) {  // ~30x30 = 900 pixels
            cout << "✓ Color inRange test passed!" << endl;
        } else {
            cout << "⚠ Expected ~900 red pixels, found " << white_pixels << endl;
        }
        
        // Test range detection for green color (B=0, G=240-255, R=0)
        KCV::inRange(kcv_color, KCV::Scalar(0, 240, 0), KCV::Scalar(10, 255, 10), color_result);
        
        result_data = color_result.getData<uchar>();
        white_pixels = 0;
        for (int i = 0; i < color_result.rows * color_result.cols; i++) {
            if (result_data[i] == 255) white_pixels++;
        }
        
        cout << "✓ Found " << white_pixels << " green pixels in range" << endl;
        if (white_pixels >= 800 && white_pixels <= 1000) {
            cout << "✓ Green detection test passed!" << endl;
        } else {
            cout << "⚠ Expected ~900 green pixels, found " << white_pixels << endl;
        }
        
    } catch (const KCV::Exception& e) {
        cout << "✗ KCV Exception in color test: " << e.what() << endl;
    }
    
    // Test 3: Edge cases
    cout << "\nTest 3: Edge cases" << endl;
    
    try {
        // Test with empty image
        KCV::Mat empty_mat;
        KCV::Mat empty_result;
        KCV::inRange(empty_mat, KCV::Scalar(0), KCV::Scalar(255), empty_result);
        cout << "✗ Should have thrown exception for empty image" << endl;
    } catch (const KCV::Exception& e) {
        cout << "✓ Correctly handled empty image: " << e.what() << endl;
    }
    
    cout << "\n✓ inRange function tests completed!" << endl;
}
