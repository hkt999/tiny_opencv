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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

Release build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This will create:
- `libtiny_opencv.a`: The main library
- `tiny_opencv_test`: Test executable

## Usage

### CMake Integration (Library Only)

```cmake
add_subdirectory(path/to/tiny_opencv)
target_link_libraries(your_app PRIVATE tiny_opencv)
target_include_directories(your_app PRIVATE path/to/tiny_opencv/include)
```

### Examples

See `examples/` for standalone snippets:
- `examples/basic.cpp`: Basic image ops (cvtColor, blur, resize)
- `examples/kalman_filter.cpp`: Kalman filter usage
- `examples/hungarian.cpp`: Hungarian assignment solver
- `examples/hough_lines.cpp`: Hough line transform
- `examples/geometric_transform.cpp`: Rotation and affine matrices
- `examples/matrix_ops.cpp`: Matrix ops (eye, multiply, inverse, access)
- `examples/inrange.cpp`: Color blob detection
- `examples/filter2d.cpp`: Custom kernel filter2D
- `examples/threshold.cpp`: Thresholding

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

// Channel ops
void split(const Mat src, Mat *mv, int count);
void merge(const Mat *mv, int count, Mat &dst);
```

## Supported Data Types

| Type | Value | Description |
|------|--------|-------------|
| CV_8UC1 | 0 | 8-bit unsigned, 1 channel |
| CV_8UC3 | 16 | 8-bit unsigned, 3 channels |
| CV_32FC1 | 5 | 32-bit float, 1 channel |
| CV_32FC3 | 21 | 32-bit float, 3 channels |

Notes:
- Types outside the table are not implemented and may produce undefined behavior.
- Many image processing functions are optimized for `CV_8UC1` and `CV_8UC3` inputs.

## Testing

Run the test suite:

```bash
./tiny_opencv_test
```

Optional test flags:

```bash
# Enable window display during tests
./tiny_opencv_test --interactive

# Limit kalman interactive-style test steps
./tiny_opencv_test --kalman-steps=100
```

Coverage gate (line/branch thresholds):

```bash
./scripts/coverage_gate.sh
```

Customize thresholds with env vars:

```bash
MIN_LINE_COVERAGE=75 MIN_BRANCH_COVERAGE=55 ./scripts/coverage_gate.sh
```

Also supports per-file gates:

```bash
KEY_FILE_GATES="src/mat.cpp:80:75;src/cvtcolor/rgb2hsv.cpp:75:70" ./scripts/coverage_gate.sh
```

Note: The CMake configuration currently always builds the test target and requires OpenCV to be installed and discoverable. If you only need the library and do not have OpenCV, remove/disable the `tiny_opencv_test` target in `CMakeLists.txt` or add your own CMake option to guard it.

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

## Third-Party

- **quirc**: Embedded QR code decoder used by the QR detection feature. Please ensure its upstream license is compatible with your use.
