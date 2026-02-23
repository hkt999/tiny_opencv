#ifndef _KCV_UNITTEST_HPP_
#define _KCV_UNITTEST_HPP_

// color space conversion test
void *unit_test_bgr2gray(void *src, int width, int height);
void *unit_test_gray2bgr(void *src, int width, int height);
void *uint_test_bgr2yuv_i420(void *src, int width, int height);
void *unit_test_yuv2bgr_i420(void *src, int width, int height);

void *unit_test_rgb2gray(void *src, int width, int height);
void *unit_test_gray2rgb(void *src, int width, int height);
void *uint_test_rgb2yuv_i420(void *src, int width, int height);
void *unit_test_yuv2rgb_i420(void *src, int width, int height);

void *unit_test_bgr2rgb(void *src, int width, int height);
void *unit_test_rgb2bgr(void *src, int width, int height);

// equalize histogray test
void *unit_test_equalize_hist(void *src, int width, int height);
void *unit_test_blur_c1(void *src, int width, int height, int kernel_w, int kernel_h);
void *unit_test_blur_c3(void *src, int width, int height, int kernel_w, int kernel_h);

// filter 2d test functions
void *unit_test_filter2d_c1(void *data, int width, int height, int kernel_w, int kernel_h, float *kernel_v);
void *unit_test_filter2d_c3(void *data, int width, int height, int kernel_w, int kernel_h, float *kernel_v);

// gaussian blur test functions
void *unit_test_gaussian_c1(void *data, int width, int height, int ksize, double sigma);
void *unit_test_gaussian_c3(void *data, int width, int height, int ksize, double sigma);

// bilateral filter test functions
void *unit_test_bilateral_c1(void *data, int width, int height, int d, double sigmaColor, double sigmaSpace );
void *unit_test_bilateral_c3(void *data, int width, int height, int d, double sigmaColor, double sigmaSpace );

// media filter test functions
void *unit_test_median_blue_c1(void *data, int width, int height, int ksize);
void *unit_test_median_blue_c3(void *data, int width, int height, int ksize);

// resiz test functions
void *unit_test_resize_c1(void *data, int width, int height, int dst_width, int dst_height);
void *unit_test_resize_c3(void *data, int width, int height, int dst_width, int dst_height);

// crop test functions
void *unit_test_crop_c1(void *data, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h);
void *unit_test_crop_c3(void *data, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h);

// matrix operation test functions
int unit_test_inverse_matrix();
int unit_test_transpose_matrix();

// hough lines test functions
void unit_test_hough_lines();

// inRange test functions
void unit_test_in_range();
void unit_test_coverage();

typedef struct _point_t {
    int x;
    int y;
} point_t;

typedef point_t (*get_observation_t)(void *data);
typedef void (*draw_frame_t)(void *data, point_t observed, point_t predicted, point_t actual_to);
void unit_test_kalman_filter_angle(void *data, draw_frame_t func);
void unit_test_kalman_filter_mouse(void *data, draw_frame_t func, get_observation_t obs);

#endif
