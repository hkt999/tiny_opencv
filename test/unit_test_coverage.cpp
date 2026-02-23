#include <cmath>
#include <iostream>
#include "unit_test.hpp"
#include "tiny_opencv.hpp"
#include "hungarian.hpp"
#include "kalman_filter.hpp"

using namespace std;
using namespace KCV;

static void fail_test(const char *msg)
{
    cout << "Coverage test failed: " << msg << endl;
    exit(1);
}

static void expect_true(bool cond, const char *msg)
{
    if (!cond) {
        fail_test(msg);
    }
}

static void expect_uchar_near(uchar got, int expect, int tol, const char *msg)
{
    int diff = abs((int)got - expect);
    if (diff > tol) {
        fail_test(msg);
    }
}

static void expect_hue_red(uchar got, const char *msg)
{
    int v = (int)got;
    if (v <= 3 || v >= 252) {
        return;
    }
    fail_test(msg);
}

#define EXPECT_THROW(stmt, msg)                    \
    do {                                           \
        bool thrown = false;                       \
        try {                                      \
            stmt;                                  \
        } catch (const KCV::Exception &) {         \
            thrown = true;                         \
        }                                          \
        if (!thrown) {                             \
            fail_test(msg);                        \
        }                                          \
    } while (0)

static void test_threshold_modes()
{
    Mat src(1, 5, CV_8UC1);
    src.at<uchar>(0, 0) = 10;
    src.at<uchar>(0, 1) = 50;
    src.at<uchar>(0, 2) = 100;
    src.at<uchar>(0, 3) = 150;
    src.at<uchar>(0, 4) = 200;

    Mat out;

    threshold(src, out, 100, 255, THRESH_BINARY);
    uchar exp_bin[5] = {0, 0, 0, 255, 255};
    for (int i = 0; i < 5; i++) expect_true(out.at<uchar>(0, i) == exp_bin[i], "THRESH_BINARY mismatch");

    threshold(src, out, 100, 255, THRESH_BINARY_INV);
    uchar exp_bin_inv[5] = {255, 255, 255, 0, 0};
    for (int i = 0; i < 5; i++) expect_true(out.at<uchar>(0, i) == exp_bin_inv[i], "THRESH_BINARY_INV mismatch");

    threshold(src, out, 100, 255, THRESH_TRUNC);
    uchar exp_trunc[5] = {10, 50, 100, 100, 100};
    for (int i = 0; i < 5; i++) expect_true(out.at<uchar>(0, i) == exp_trunc[i], "THRESH_TRUNC mismatch");

    threshold(src, out, 100, 255, THRESH_TOZERO);
    uchar exp_tozero[5] = {0, 0, 0, 150, 200};
    for (int i = 0; i < 5; i++) expect_true(out.at<uchar>(0, i) == exp_tozero[i], "THRESH_TOZERO mismatch");

    threshold(src, out, 100, 255, THRESH_TOZERO_INV);
    uchar exp_tozero_inv[5] = {10, 50, 100, 0, 0};
    for (int i = 0; i < 5; i++) expect_true(out.at<uchar>(0, i) == exp_tozero_inv[i], "THRESH_TOZERO_INV mismatch");

    Mat invalid_type(1, 3, CV_8UC3);
    Mat invalid_out;
    threshold(invalid_type, invalid_out, 10, 255, THRESH_BINARY);
    expect_true(invalid_out.empty(), "threshold should reject non-gray input");

    threshold(src, out, 100, 255, 99);
    expect_true(out.empty(), "threshold invalid mode should produce empty output");
}

static void test_box_filter_and_geometry()
{
    Mat src(5, 5, CV_8UC1);
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            src.at<uchar>(r, c) = (uchar)(r * 10 + c);
        }
    }

    Mat dst;
    boxFilter(src, dst, -1, Size(3, 3));
    expect_true(dst.rows == src.rows && dst.cols == src.cols, "boxFilter shape mismatch");
    expect_true(dst.type == src.type, "boxFilter type mismatch");

    Mat rot = getRotationMatrix2D(Point2f(0.0f, 0.0f), 0.0, 1.0);
    expect_true(rot.rows == 2 && rot.cols == 3, "rotation matrix shape mismatch");
    expect_true(fabs(rot.at<float>(0, 0) - 1.0f) < 1e-4f, "rotation matrix value mismatch");
    expect_true(fabs(rot.at<float>(1, 1) - 1.0f) < 1e-4f, "rotation matrix value mismatch");

    Point2f src_tri[3] = {Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)};
    Point2f dst_tri[3] = {Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)};
    Mat aff = getAffineTransform(src_tri, dst_tri);
    expect_true(aff.rows == 2 && aff.cols == 3, "affine matrix shape mismatch");
    for (int i = 0; i < aff.rows; i++) {
        for (int j = 0; j < aff.cols; j++) {
            expect_true(isfinite(aff.at<float>(i, j)), "affine matrix contains non-finite value");
        }
    }
}

static void test_hungarian_and_random()
{
    Mat cost(2, 2, CV_32FC1);
    cost.at<float>(0, 0) = 4;
    cost.at<float>(0, 1) = 1;
    cost.at<float>(1, 0) = 2;
    cost.at<float>(1, 1) = 3;
    Mat assignment(2, 1, CV_32FC1);
    HungarianAlgorithm solver;
    double total_cost = solver.Solve(cost, assignment);
    expect_true(fabs(total_cost - 3.0) < 1e-5, "hungarian cost mismatch");
    expect_true((int)assignment.at<float>(0, 0) == 1, "hungarian assignment row0 mismatch");
    expect_true((int)assignment.at<float>(1, 0) == 0, "hungarian assignment row1 mismatch");

    Mat rect_cost(3, 2, CV_32FC1);
    rect_cost.at<float>(0, 0) = 1; rect_cost.at<float>(0, 1) = 9;
    rect_cost.at<float>(1, 0) = 9; rect_cost.at<float>(1, 1) = 1;
    rect_cost.at<float>(2, 0) = 5; rect_cost.at<float>(2, 1) = 5;
    Mat rect_assign(3, 1, CV_32FC1);
    double rect_total = solver.Solve(rect_cost, rect_assign);
    expect_true(isfinite(rect_total), "hungarian rectangular cost is non-finite");
    for (int i = 0; i < 3; i++) {
        int a = (int)rect_assign.at<float>(i, 0);
        expect_true(a >= -1 && a < 2, "hungarian rectangular assignment out of range");
    }

    Mat noise(64, 1, CV_32FC1);
    randn(noise, 0.0f, 1.0f);
    float first = noise.at<float>(0, 0);
    bool has_diff = false;
    for (int i = 1; i < noise.rows; i++) {
        if (noise.at<float>(i, 0) != first) {
            has_diff = true;
            break;
        }
    }
    expect_true(has_diff, "randn produced constant output");
}

static void test_kalman_noninteractive()
{
    KalmanFilter kf(4, 2, 0, CV_32F);
    setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(0, 2) = 1.0f;
    kf.transitionMatrix.at<float>(1, 3) = 1.0f;
    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, Scalar::all(1e-3));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    Mat meas(2, 1, CV_32FC1);
    Mat pred;
    Mat corr;
    for (int i = 0; i < 20; i++) {
        meas.at<float>(0, 0) = 10.0f + i * 0.5f;
        meas.at<float>(1, 0) = 20.0f + i * 0.25f;
        pred = kf.predict();
        corr = kf.correct(meas);
        expect_true(pred.rows == 4 && pred.cols == 1, "kalman predict shape mismatch");
        expect_true(corr.rows == 4 && corr.cols == 1, "kalman correct shape mismatch");
        expect_true(isfinite(corr.at<float>(0, 0)) && isfinite(corr.at<float>(1, 0)), "kalman output non-finite");
    }
    expect_true(fabs(corr.at<float>(0, 0) - meas.at<float>(0, 0)) < 5.0f, "kalman x estimate diverged");
    expect_true(fabs(corr.at<float>(1, 0) - meas.at<float>(1, 0)) < 5.0f, "kalman y estimate diverged");
}

static void fill_rgb_2x2(Mat &img, uchar r, uchar g, uchar b)
{
    for (int i = 0; i < 4; i++) {
        img.at<cv8uc3_t>(i).c1 = r;
        img.at<cv8uc3_t>(i).c2 = g;
        img.at<cv8uc3_t>(i).c3 = b;
    }
}

static void fill_bgr_2x2(Mat &img, uchar b, uchar g, uchar r)
{
    for (int i = 0; i < 4; i++) {
        img.at<cv8uc3_t>(i).c1 = b;
        img.at<cv8uc3_t>(i).c2 = g;
        img.at<cv8uc3_t>(i).c3 = r;
    }
}

static void test_yuv_i420_numeric()
{
    Mat rgb(2, 2, CV_8UC3);
    fill_rgb_2x2(rgb, 255, 0, 0);

    Mat yuv;
    cvtColor(rgb, yuv, CV_RGB2YUV_I420);
    expect_true(yuv.rows == 3 && yuv.cols == 2, "RGB2YUV_I420 shape mismatch");
    expect_true(yuv.type == CV_8UC1, "RGB2YUV_I420 type mismatch");
    expect_true(yuv.at<uchar>(0, 0) == 81, "RGB2YUV_I420 Y mismatch for red");
    expect_true(yuv.at<uchar>(2, 0) == 90, "RGB2YUV_I420 U mismatch for red");
    expect_true(yuv.at<uchar>(2, 1) == 239, "RGB2YUV_I420 V mismatch for red");

    Mat rgb_back;
    cvtColor(yuv, rgb_back, CV_YUV2RGB_I420);
    cv8uc3_t rgb_back_px = rgb_back.at<cv8uc3_t>(0, 0);
    expect_uchar_near(rgb_back_px.c1, 255, 14, "YUV2RGB_I420 red channel mismatch");
    expect_uchar_near(rgb_back_px.c2, 0, 14, "YUV2RGB_I420 green channel mismatch");
    expect_uchar_near(rgb_back_px.c3, 0, 14, "YUV2RGB_I420 blue channel mismatch");

    Mat bgr(2, 2, CV_8UC3);
    fill_bgr_2x2(bgr, 255, 0, 0);
    Mat yuv_from_bgr;
    cvtColor(bgr, yuv_from_bgr, CV_BGR2YUV_I420);
    expect_true(yuv_from_bgr.at<uchar>(0, 0) == 40, "BGR2YUV_I420 Y mismatch for blue");
    expect_true(yuv_from_bgr.at<uchar>(2, 0) == 239, "BGR2YUV_I420 U mismatch for blue");
    expect_true(yuv_from_bgr.at<uchar>(2, 1) == 119, "BGR2YUV_I420 V mismatch for blue");

    Mat bgr_back;
    cvtColor(yuv_from_bgr, bgr_back, CV_YUV2BGR_I420);
    cv8uc3_t bgr_back_px = bgr_back.at<cv8uc3_t>(0, 0);
    expect_uchar_near(bgr_back_px.c1, 255, 14, "YUV2BGR_I420 blue channel mismatch");
    expect_uchar_near(bgr_back_px.c2, 0, 14, "YUV2BGR_I420 green channel mismatch");
    expect_uchar_near(bgr_back_px.c3, 0, 14, "YUV2BGR_I420 red channel mismatch");
}

static void test_hsv_numeric()
{
    Mat bgr_red(1, 1, CV_8UC3);
    bgr_red.at<cv8uc3_t>(0, 0).c1 = 0;
    bgr_red.at<cv8uc3_t>(0, 0).c2 = 0;
    bgr_red.at<cv8uc3_t>(0, 0).c3 = 255;

    Mat hsv;
    cvtColor(bgr_red, hsv, CV_BGR2HSV);
    cv8uc3_t hsv_px = hsv.at<cv8uc3_t>(0, 0);
    expect_hue_red(hsv_px.c1, "BGR2HSV hue mismatch for red");
    expect_uchar_near(hsv_px.c2, 255, 2, "BGR2HSV saturation mismatch for red");
    expect_uchar_near(hsv_px.c3, 255, 2, "BGR2HSV value mismatch for red");

    Mat bgr_back;
    cvtColor(hsv, bgr_back, CV_HSV2BGR);
    cv8uc3_t bgr_back_px = bgr_back.at<cv8uc3_t>(0, 0);
    expect_uchar_near(bgr_back_px.c1, 0, 3, "HSV2BGR blue mismatch for red");
    expect_uchar_near(bgr_back_px.c2, 0, 3, "HSV2BGR green mismatch for red");
    expect_uchar_near(bgr_back_px.c3, 255, 3, "HSV2BGR red mismatch for red");

    Mat rgb_red(1, 1, CV_8UC3);
    rgb_red.at<cv8uc3_t>(0, 0).c1 = 255;
    rgb_red.at<cv8uc3_t>(0, 0).c2 = 0;
    rgb_red.at<cv8uc3_t>(0, 0).c3 = 0;
    Mat hsv_from_rgb;
    cvtColor(rgb_red, hsv_from_rgb, CV_RGB2HSV);
    cv8uc3_t hsv_rgb_px = hsv_from_rgb.at<cv8uc3_t>(0, 0);
    expect_hue_red(hsv_rgb_px.c1, "RGB2HSV hue mismatch for red");

    Mat rgb_back;
    cvtColor(hsv_from_rgb, rgb_back, CV_HSV2RGB);
    cv8uc3_t rgb_back_px = rgb_back.at<cv8uc3_t>(0, 0);
    expect_uchar_near(rgb_back_px.c1, 255, 3, "HSV2RGB red mismatch");
    expect_uchar_near(rgb_back_px.c2, 0, 3, "HSV2RGB green mismatch");
    expect_uchar_near(rgb_back_px.c3, 0, 3, "HSV2RGB blue mismatch");

    Mat bgr_src(1, 1, CV_8UC3);
    bgr_src.at<cv8uc3_t>(0, 0).c1 = 90;
    bgr_src.at<cv8uc3_t>(0, 0).c2 = 140;
    bgr_src.at<cv8uc3_t>(0, 0).c3 = 210;
    Mat hsv_mid, bgr_roundtrip;
    cvtColor(bgr_src, hsv_mid, CV_BGR2HSV);
    cvtColor(hsv_mid, bgr_roundtrip, CV_HSV2BGR);
    cv8uc3_t src_px = bgr_src.at<cv8uc3_t>(0, 0);
    cv8uc3_t rt_px = bgr_roundtrip.at<cv8uc3_t>(0, 0);
    expect_uchar_near(rt_px.c1, src_px.c1, 4, "HSV roundtrip blue mismatch");
    expect_uchar_near(rt_px.c2, src_px.c2, 4, "HSV roundtrip green mismatch");
    expect_uchar_near(rt_px.c3, src_px.c3, 4, "HSV roundtrip red mismatch");
}

static void test_split_merge()
{
    Mat src(2, 2, CV_8UC3);
    src.at<cv8uc3_t>(0, 0).c1 = 1;  src.at<cv8uc3_t>(0, 0).c2 = 2;  src.at<cv8uc3_t>(0, 0).c3 = 3;
    src.at<cv8uc3_t>(0, 1).c1 = 4;  src.at<cv8uc3_t>(0, 1).c2 = 5;  src.at<cv8uc3_t>(0, 1).c3 = 6;
    src.at<cv8uc3_t>(1, 0).c1 = 7;  src.at<cv8uc3_t>(1, 0).c2 = 8;  src.at<cv8uc3_t>(1, 0).c3 = 9;
    src.at<cv8uc3_t>(1, 1).c1 = 10; src.at<cv8uc3_t>(1, 1).c2 = 11; src.at<cv8uc3_t>(1, 1).c3 = 12;

    Mat ch[3];
    split(src, ch, 3);
    expect_true(ch[0].type == CV_8UC1 && ch[1].type == CV_8UC1 && ch[2].type == CV_8UC1, "split output type mismatch");
    expect_true(ch[0].at<uchar>(1, 1) == 10, "split channel0 mismatch");
    expect_true(ch[1].at<uchar>(1, 1) == 11, "split channel1 mismatch");
    expect_true(ch[2].at<uchar>(1, 1) == 12, "split channel2 mismatch");

    Mat merged;
    merge(ch, 3, merged);
    expect_true(merged.type == CV_8UC3 && merged.rows == 2 && merged.cols == 2, "merge output shape/type mismatch");
    cv8uc3_t p = merged.at<cv8uc3_t>(1, 0);
    expect_true(p.c1 == 7 && p.c2 == 8 && p.c3 == 9, "merge values mismatch");

    Mat gray(2, 2, CV_8UC1);
    gray.at<uchar>(0, 0) = 9; gray.at<uchar>(0, 1) = 8;
    gray.at<uchar>(1, 0) = 7; gray.at<uchar>(1, 1) = 6;
    Mat gch[1];
    split(gray, gch, 1);
    expect_true(gch[0].at<uchar>(1, 0) == 7, "split gray mismatch");
    Mat gray_merged;
    merge(gch, 1, gray_merged);
    expect_true(gray_merged.type == CV_8UC1, "merge single channel type mismatch");
    expect_true(gray_merged.at<uchar>(0, 1) == 8, "merge single channel value mismatch");

    EXPECT_THROW(split(src, ch, 2), "split should fail on count mismatch");
    Mat bad_merge[2];
    bad_merge[0] = Mat(2, 2, CV_8UC1);
    bad_merge[1] = Mat(1, 2, CV_8UC1);
    EXPECT_THROW(merge(bad_merge, 2, merged), "merge should fail on shape mismatch");
}

static void test_resize_modes_and_errors()
{
    Mat src(2, 2, CV_8UC1);
    src.at<uchar>(0, 0) = 0;
    src.at<uchar>(0, 1) = 64;
    src.at<uchar>(1, 0) = 128;
    src.at<uchar>(1, 1) = 255;

    Mat nearest;
    resize(src, nearest, Size(4, 4), 0.0f, 0.0f, INTER_NEAREST);
    expect_true(nearest.rows == 4 && nearest.cols == 4, "nearest resize shape mismatch");
    expect_true(nearest.at<uchar>(0, 0) == 0, "nearest resize top-left mismatch");
    expect_true(nearest.at<uchar>(3, 3) == 255, "nearest resize bottom-right mismatch");

    Mat linear;
    resize(src, linear, Size(3, 3), 0.0f, 0.0f, INTER_LINEAR);
    expect_true(linear.rows == 3 && linear.cols == 3, "linear resize shape mismatch");
    expect_uchar_near(linear.at<uchar>(1, 1), 112, 3, "linear resize center mismatch");

    Mat by_ratio;
    resize(src, by_ratio, Size(), 2.0f, 2.0f, INTER_NEAREST);
    expect_true(by_ratio.rows == 4 && by_ratio.cols == 4, "ratio resize shape mismatch");

    EXPECT_THROW(resize(src, nearest, Size(), 0.0f, 2.0f, INTER_NEAREST), "resize should reject invalid ratio");
    EXPECT_THROW(resize(src, nearest, Size(4, 4), 0.0f, 0.0f, INTER_CUBIC), "resize should reject unsupported cubic mode");
    EXPECT_THROW(resize(src, nearest, Size(4, 4), 0.0f, 0.0f, 99), "resize should reject unknown mode");
}

static void test_error_paths()
{
    Mat src8(3, 3, CV_8UC1);
    Mat dst;

    Mat kernel_even(2, 2, CV_32FC1);
    for (int i = 0; i < 4; i++) kernel_even.at<float>(i) = 0.25f;
    EXPECT_THROW(filter2D(src8, dst, -1, kernel_even), "filter2D should reject even kernel");

    Mat kernel_bad_type(3, 3, CV_8UC3);
    EXPECT_THROW(filter2D(src8, dst, -1, kernel_bad_type), "filter2D should reject multi-channel kernel");

    Mat kernel_ok(3, 3, CV_32FC1);
    for (int i = 0; i < 9; i++) kernel_ok.at<float>(i) = (i == 4) ? 1.0f : 0.0f;
    EXPECT_THROW(filter2D(src8, dst, CV_32FC1, kernel_ok), "filter2D should reject unsupported ddepth");

    Mat bad_src(3, 3, CV_32FC1);
    EXPECT_THROW(filter2D(bad_src, dst, -1, kernel_ok), "filter2D should reject unsupported source type");

    Mat src_bgr(2, 2, CV_8UC3);
    EXPECT_THROW(cvtColor(src8, dst, CV_BGR2GRAY), "cvtColor should reject wrong src type");
    EXPECT_THROW(cvtColor(src_bgr, dst, 999), "cvtColor should reject unknown code");

    Mat odd(3, 3, CV_8UC3);
    EXPECT_THROW(cvtColor(odd, dst, CV_BGR2YUV_I420), "cvtColor should reject odd-size I420 input");

    Mat invalid_i420(5, 2, CV_8UC1);
    EXPECT_THROW(cvtColor(invalid_i420, dst, CV_YUV2RGB_I420), "cvtColor should reject invalid I420 layout");
}

static void test_mat_semantics()
{
    Mat a(2, 2, CV_8UC1);
    a.at<uchar>(0, 0) = 10;
    a.at<uchar>(0, 1) = 20;
    a.at<uchar>(1, 0) = 30;
    a.at<uchar>(1, 1) = 40;

    Mat b = a;
    expect_true(a.ref == b.ref, "copy constructor should share ref");
    b.at<uchar>(0, 0) = 77;
    expect_true(a.at<uchar>(0, 0) == 77, "shared ref write should be visible");

    Mat *clone_p = &a.clone();
    clone_p->at<uchar>(0, 0) = 5;
    expect_true(a.at<uchar>(0, 0) == 77, "clone should be deep copy");
    delete clone_p;

    Mat copy_v;
    a.copyTo(copy_v);
    copy_v.at<uchar>(1, 1) = 99;
    expect_true(a.at<uchar>(1, 1) == 40, "copyTo should be deep copy");

    Mat self = a;
    self = self;
    expect_true(self.at<uchar>(1, 0) == 30, "self assignment should keep data");

    Mat roi_src(3, 3, CV_8UC1);
    int val = 0;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            roi_src.at<uchar>(r, c) = (uchar)(val++);
        }
    }
    Mat roi_copy(roi_src, Rect(1, 1, 2, 2));
    expect_true(roi_copy.rows == 2 && roi_copy.cols == 2, "roi constructor shape mismatch");
    expect_true(roi_copy.at<uchar>(0, 0) == roi_src.at<uchar>(1, 1), "roi constructor value mismatch");

    Mat m1(2, 3, CV_32FC1), m2(4, 2, CV_32FC1);
    EXPECT_THROW(m1 * m2, "operator* should reject dimension mismatch");
}

void unit_test_coverage()
{
    cout << "=== Coverage Supplement Test ===" << endl;
    test_threshold_modes();
    test_box_filter_and_geometry();
    test_hungarian_and_random();
    test_kalman_noninteractive();
    test_split_merge();
    test_resize_modes_and_errors();
    test_error_paths();
    test_mat_semantics();
    test_yuv_i420_numeric();
    test_hsv_numeric();
    cout << "✓ Coverage supplement test passed!" << endl;
}
