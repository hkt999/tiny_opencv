
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "unit_test.hpp"

using namespace cv;
using namespace std;

#if 0
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+
#endif

const char *color2str(int type)
{
	static const char *name[] = { "CV_8UC", "CV_8SC", "CV_16UC", "CV_16SCl", "CV_32SC", "CV_32FC", "CV_64FC", "PADDING"};
	static char ret[32];
	int type1 = type%8;
	int type2 = 1 + type/8 - type1*8;
	snprintf(ret, sizeof(ret), "(%d:%s%d)", type, name[type1], type2);

	return ret;
}

// test cvtColor
void mat_test_bgr_gray(Mat &img)
{
	void *gray = unit_test_bgr2gray(img.data, img.cols, img.rows);
	void *bgr = unit_test_gray2bgr(gray, img.cols, img.rows);

	Mat gray_img(img.rows, img.cols, CV_8UC1, gray);
	imshow("BGR24 to Gray", gray_img);

	Mat bgr_img(img.rows, img.cols, CV_8UC3, bgr);
	imshow("Gray to BGR24", bgr_img);
	free(gray);
	free(bgr);
}

void mat_test_bgr_yuv(Mat &img)
{
	void *yuv = uint_test_bgr2yuv_i420(img.data, img.cols, img.rows);
	void *bgr = unit_test_yuv2bgr_i420(yuv, img.cols, img.rows);

	Mat yuv_img(img.rows*3/2, img.cols, CV_8UC1, yuv);
	imshow("BGR24 to I420", yuv_img);

	Mat rgb_img(img.rows, img.cols, CV_8UC3, bgr);
	imshow("I420 to BGR24", rgb_img);

	free(yuv);
	free(bgr);
}

void mat_test_rgb_gray(Mat &img)
{
	void *gray = unit_test_bgr2gray( img.data, img.cols, img.rows);
	void *rgb = unit_test_gray2bgr(gray, img.cols, img.rows);

	Mat gray_img(img.rows, img.cols, CV_8UC1, gray);
	imshow("RGB24 to Gray", gray_img);

	Mat rgb_img(img.rows, img.cols, CV_8UC3, rgb);
	imshow("Gray to RGB24", rgb_img);
	free(gray);
	free(rgb);
}

void mat_test_rgb_yuv(Mat &img)
{
	void *yuv = uint_test_rgb2yuv_i420(img.data, img.cols, img.rows);
	void *rgb = unit_test_yuv2rgb_i420(yuv, img.cols, img.rows);

	Mat yuv_img(img.rows*3/2, img.cols, CV_8UC1, yuv);
	imshow("RGB24 to I420", yuv_img);

	Mat rgb_img(img.rows, img.cols, CV_8UC3, rgb);
	imshow("I420 to RGB24", rgb_img);

	free(yuv);
	free(rgb);
}

void mat_test_rgb_bgr(Mat &img)
{
	void *rgb = unit_test_bgr2rgb(img.data, img.cols, img.rows);
	void *bgr = unit_test_rgb2bgr(rgb, img.cols, img.rows);

	Mat rgb_img(img.rows, img.cols, CV_8UC3, rgb);
	imshow("BGR to RGB", rgb_img);

	Mat bgr_img(img.rows, img.cols, CV_8UC3, bgr);
	imshow("RGB to BGR", bgr_img);

	free(rgb);
	free(bgr);
}

// test equalizeHist
void mat_test_equalize_hist(Mat &img)
{
	Mat g;
	cvtColor(img, g, CV_BGR2GRAY);
	imshow("Original Gray", g);

	void *gray = unit_test_equalize_hist(g.data, g.cols, g.rows);
	Mat gray_img(g.rows, g.cols, CV_8UC1, gray);
	imshow("Histogram Equalized", gray_img);

	free(gray);
}

// test blur
void mat_test_blur_c1(Mat &img)
{
	Mat g;
	cvtColor(img, g, CV_BGR2GRAY);
	imshow("Original Gray", g);

	void *gray = unit_test_blur_c1(g.data, g.cols, g.rows, 5, 5);
	Mat gray_img(g.rows, g.cols, CV_8UC1, gray);
	imshow("Blur 5x5 C1", gray_img);

	free(gray);
}

void mat_test_blur_c3(Mat &img)
{
	void *c3 = unit_test_blur_c3(img.data, img.cols, img.rows, 5, 5);
	Mat g( img.rows, img.cols, CV_8UC3, c3);
	imshow("Blur 5x5 C3", g);
	free(c3);
}

// Gaussion matrix 5x5 for test
static float kernel_5x5[5*5] = {
	0.00296902, 0.0133062, 0.0219382, 0.0133062, 0.00296902,
	0.0133062,  0.0596343, 0.0983203, 0.0596343, 0.0133062,
	0.0219382,  0.0983203, 0.162103,  0.0983203, 0.0219382,
	0.0133062,  0.0596343, 0.0983203, 0.0596343, 0.0133062,
	0.00296902, 0.0133062, 0.0219382, 0.0133062, 0.00296902
};

void mat_test_filter2d_c1(Mat &img)
{
	Mat g;
	void *c1 = unit_test_filter2d_c1(img.data, img.cols, img.rows, 5, 5, kernel_5x5);
	cvtColor( img, g, CV_BGR2GRAY);
	imshow("filter2D 5x5 C1", g);
	free(c1);
}

void mat_test_filter2d_c3(Mat &img)
{
	void *c3 = unit_test_filter2d_c3(img.data, img.cols, img.rows, 5, 5, kernel_5x5);
	Mat g( img.rows, img.cols, CV_8UC3, c3);
	imshow("filter2D 5x5 C3", g);
	free(c3);
}

#if 0
void *unit_test_gaussian_c1(void *data, int width, int height, int ksize, double sigma);
void *unit_test_gaussian_c3(void *data, int width, int height, int ksize, double sigma);
#endif

void mat_test_gaussian_c1(Mat &img)
{
	void *c1 = unit_test_gaussian_c1(img.data, img.cols, img.rows, 5, 0);
	Mat g( img.rows, img.cols, CV_8UC1, c1);
	imshow("Gaussian 5x5 C1", g);
	free(c1);
}

void mat_test_gaussian_c3(Mat &img)
{
	void *c3 = unit_test_gaussian_c3(img.data, img.cols, img.rows, 5, 0);
	Mat g( img.rows, img.cols, CV_8UC3, c3);
	imshow("Gaussian 5x5 C1", g);
	free(c3);
}

void mat_test_bilateral_c1(Mat &img)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	void *c1 = unit_test_bilateral_c1(gray.data, gray.cols, gray.rows, 2, 1.0, 1.0);
	Mat g(img.rows, img.cols, CV_8UC1, c1);
	imshow("Bilateral Filter (C1)", g);
	free(c1);
}

void mat_test_bilateral_c3(Mat &img)
{
	void *c3 = unit_test_bilateral_c3(img.data, img.cols, img.rows, 2, 1.0, 1.0);
	Mat g(img.rows, img.cols, CV_8UC3, c3);
	imshow("Bilateral Filter (C3)", g);
	free(c3);
}

void mat_test_median_filter_c1(Mat &img)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	void *c1 = unit_test_median_blue_c1(gray.data, gray.cols, gray.rows, 5);
	Mat g(img.rows, img.cols, CV_8UC1, c1);
	imshow("Median Filter 5x5 (C1)", g);
	free(c1);
}

void mat_test_median_filter_c3(Mat &img)
{
	void *c3 = unit_test_median_blue_c3(img.data, img.cols, img.rows, 5);
	Mat g(img.rows, img.cols, CV_8UC3, c3);
	imshow("Median Filter 5x5 (C3)", g);
	free(c3);
}

#define RESIZE_WIDTH	1024
#define RESIZE_HEIGHT	1024

void mat_test_resize_c1(Mat &img)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	void *c1 = unit_test_resize_c1(gray.data, gray.cols, gray.rows, RESIZE_WIDTH, RESIZE_HEIGHT);
	Mat g(RESIZE_HEIGHT, RESIZE_WIDTH, CV_8UC1, c1);
	imshow("Resize filter (C1)", g);
	free(c1);
}

void mat_test_resize_c3(Mat &img)
{
	void *c3 = unit_test_resize_c3(img.data, img.cols, img.rows, RESIZE_WIDTH, RESIZE_HEIGHT);
	Mat g(RESIZE_HEIGHT, RESIZE_WIDTH, CV_8UC3, c3);
	imshow("Resize (C3)", g);
	free(c3);
}

void mat_test_crop_c1(Mat &img)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	void *c1 = unit_test_crop_c1(gray.data, gray.cols, gray.rows, 100, 100, 400, 400);
	Mat g(400, 400, CV_8UC1, c1);
	imshow("Crop (100,100,400,400) C1", g);
	free(c1);
}

void mat_test_crop_c3(Mat &img)
{
	void *c3 = unit_test_crop_c3(img.data, img.cols, img.rows, 100, 100, 400, 400);
	Mat g(400, 400, CV_8UC3, c3);
	imshow("Crop (100,100,400,400) C3", g);
	free(c3);
}

typedef struct kalman_test_t_ {
    Mat *img;
	int inited;
	point_t pos;
} kalman_test_t;

point_t cv_get_mouse(void *data)
{
	kalman_test_t *obj = (kalman_test_t *)data;
	return obj->pos;
}

void mouseHandler( int e, int x, int y, int d, void *ptr)
{
	kalman_test_t *p = (kalman_test_t *)ptr;
	p->pos.x = x;
	p->pos.y = y;
}

// Kalman tester callback for drawing points
void cv_draw_frame(void *data, point_t observed, point_t predicted, point_t actual_to)
{
	kalman_test_t *obj = (kalman_test_t *)data;
	printf("cb observed(%d, %d), predicted(%d, %d), actual_to(%d, %d)\n",
		observed.x, observed.y,
		predicted.x, predicted.y,
		actual_to.x, actual_to.y);

	if (obj->inited == 0) {
		obj->img = new Mat(512, 512, CV_8UC3);
		obj->inited = 1;
		namedWindow("Kalman");
	    setMouseCallback("Kalman", mouseHandler, obj); 
	}

	memset(obj->img->data, 0,  obj->img->rows * obj->img->cols * 3);
    circle(*obj->img, Point(observed.x, observed.y), 4, cv::Scalar(128, 255, 255));  // observed
    circle(*obj->img, Point(predicted.x, predicted.y), 4, cv::Scalar(255, 255, 255), 2);  // predicted
    circle(*obj->img, Point(actual_to.x, actual_to.y), 4, cv::Scalar(0, 0, 255));  // actual to
    imshow("Kalman", *obj->img);

	if ((cv::waitKey(100) & 255) == 27) {
		exit(0);
    }
}

void static_test()
{
	Mat img = imread("test/lena.jpg", cv::IMREAD_COLOR);
	if (img.empty()) {
		printf("image cannot be loaded...\n");
		exit(1);
	}
	cout << "img:type=" << color2str(img.type()) << endl;
	imshow("Original Image", img);

	mat_test_bgr_gray(img);
	mat_test_bgr_yuv(img);
	mat_test_rgb_gray(img);
	mat_test_rgb_yuv(img);
	mat_test_rgb_bgr(img);
	mat_test_equalize_hist(img);
	mat_test_blur_c1(img);
	mat_test_blur_c3(img);
	mat_test_filter2d_c3(img);
	mat_test_filter2d_c1(img);
	mat_test_filter2d_c3(img);
	mat_test_gaussian_c1(img);
	mat_test_gaussian_c3(img);
	mat_test_bilateral_c1(img);
	mat_test_bilateral_c3(img);
	mat_test_median_filter_c1(img);
	mat_test_median_filter_c3(img);
	mat_test_resize_c1(img);
	mat_test_resize_c3(img);
	mat_test_crop_c1(img);
	mat_test_crop_c3(img);
	unit_test_transpose_matrix();
	unit_test_inverse_matrix();
	unit_test_hough_lines();
	unit_test_in_range();
	waitKey(0);
	destroyAllWindows();
}
int main(int argc, const char **argv)
{
	kalman_test_t obj;

	memset(&obj, 0, sizeof(kalman_test_t));
	static_test();
	//unit_test_kalman_filter_angle(&obj, cv_draw_frame);
	//unit_test_kalman_filter_mouse(&obj, cv_draw_frame, cv_get_mouse);
}
