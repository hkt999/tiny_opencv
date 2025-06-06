# Tiny OpenCV

A lightweight C++ computer vision library that provides essential image processing and computer vision functionalities with minimal dependencies.

## Features

### Core Data Structures
- **Mat**: Memory-efficient matrix class with reference counting
- **Point, Size, Rect**: Basic geometric data structures
- **Scalar**: Multi-channel scalar values

### Image Processing
- **Color Space Conversion**: BGR/RGB ↔ Gray, BGR/RGB ↔ YUV I420, BGR ↔ RGB, BGR/RGB ↔ HSV
- **Image Filtering**: 
  - Blur, Gaussian blur, bilateral filter
  - Box filter, median filter, custom 2D convolution
  - Histogram equalization
- **Image Transformation**: Resize with multiple interpolation methods

### Computer Vision Algorithms
- **Kalman Filter**: 2D state estimation and tracking
- **Hungarian Algorithm**: Assignment problem solver
- **Hough Line Transform**: Line detection in binary images
- **Color Blob Detection**: inRange function for color-based segmentation
- **Geometric Transformations**: Rotation matrix, affine transformation

### Additional Features
- **QR Code Detection**: Integrated quirc library for QR code decoding
- **Matrix Operations**: Transpose, inverse, determinant calculations
- **Random Number Generation**: Gaussian noise generation

## Building

### Requirements
- CMake 3.12+
- C++11 compatible compiler
- OpenCV (for testing only)

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

This will create:
- `libtiny_opencv.a`: The main library
- `tiny_opencv_test`: Test executable

## Usage

### Basic Example

```cpp
#include "tiny_opencv.hpp"
using namespace KCV;

// Create a matrix
Mat img(480, 640, CV_8UC3);

// Color conversion
Mat gray;
cvtColor(img, gray, CV_BGR2GRAY);

// Apply Gaussian blur
Mat blurred;
gaussianBlur(img, blurred, Size(5, 5), 1.0);

// Resize image
Mat resized;
resize(img, resized, Size(320, 240));
```

### Kalman Filter Example

```cpp
#include "kalman_filter.hpp"

// Initialize 2D Kalman filter
kalman_filter_t *kf = kalman_filter_alloc(4, 2);
kalman_filter_init(kf);

// Update with measurement
point_t measurement = {x, y};
point_t prediction = kalman_filter_update(kf, measurement);

kalman_filter_free(kf);
```

### Matrix Operations

```cpp
// Create identity matrix
Mat identity = Mat::eye(3, 3, CV_32FC1);

// Matrix multiplication and inversion
Mat result = matrix1 * matrix2;
Mat inverted = matrix.inverse();

// Element access
float value = matrix.at<float>(row, col);
```

### Color Blob Detection Example

```cpp
#include "tiny_opencv.hpp"
using namespace KCV;

// Load color image
Mat colorImage;

// Detect red blobs (BGR format: B=0-10, G=0-10, R=240-255)
Mat redMask;
inRange(colorImage, Scalar(0, 0, 240), Scalar(10, 10, 255), redMask);

// Detect specific grayscale range (100-150)
Mat grayImage, grayMask;
cvtColor(colorImage, grayImage, CV_BGR2GRAY);
inRange(grayImage, Scalar(100), Scalar(150), grayMask);

// Result is binary image: 255 for pixels in range, 0 otherwise
```

## API Reference

### Core Classes

#### Mat
- **Constructors**: `Mat()`, `Mat(rows, cols, type)`, `Mat(size, type)`
- **Operations**: `clone()`, `copyTo()`, `transpose()`, `inverse()`
- **Element Access**: `at<T>(row, col)`, `ptr()`, `getData<T>()`

#### Geometric Types
- **Point_<T>**: 2D point with x, y coordinates
- **Size_<T>**: Width and height dimensions  
- **Rect_<T>**: Rectangle with position and size
- **Scalar**: Multi-channel scalar values

### Image Processing Functions

```cpp
// Color conversion
void cvtColor(const Mat src, Mat &dst, int code);

// Filtering
void blur(const Mat src, Mat &dst, Size ksize);
void gaussianBlur(const Mat src, Mat &dst, Size ksize, double sigmaX, double sigmaY=0.0);
void bilateralFilter(const Mat src, Mat &dst, int d, double sigmaColor, double sigmaSpace);
void medianBlur(const Mat src, Mat &dst, int ksize);

// Transformations
void resize(const Mat src, Mat &dst, Size size, float h_ratio=0.0, float v_ratio=0.0, int mode=INTER_NEAREST);
void threshold(const Mat in, Mat &out, double thresh, double maxval, int type);

// Geometric transformations
Mat getRotationMatrix2D(Point2f center, double angle, double scale);
Mat getAffineTransform(const Point2f src[], const Point2f dst[]);

// Color blob detection
void inRange(const Mat src, const Scalar lowerb, const Scalar upperb, Mat &dst);
```

## Supported Data Types

| Type | Value | Description |
|------|--------|-------------|
| CV_8UC1 | 0 | 8-bit unsigned, 1 channel |
| CV_8UC3 | 16 | 8-bit unsigned, 3 channels |
| CV_32FC1 | 5 | 32-bit float, 1 channel |
| CV_32FC3 | 21 | 32-bit float, 3 channels |

## Testing

Run the test suite:

```bash
./tiny_opencv_test
```

The test program demonstrates various functionalities including:
- Color space conversions
- Image filtering operations
- Matrix operations
- Kalman filter tracking
- Resize and crop operations

## License

This project is released under the MIT License. See the LICENSE file for details.

## Dependencies

- **Core Library**: No external dependencies
- **QR Code Detection**: Built-in quirc library
- **Testing**: OpenCV (for comparison and visualization only)