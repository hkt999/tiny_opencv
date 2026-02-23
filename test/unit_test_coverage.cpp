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

static void test_kalman_and_hsv_paths()
{
    KalmanFilter kf(4, 2, 0, CV_32F);
    setIdentity(kf.transitionMatrix);
    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, Scalar::all(1e-3));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-2));
    setIdentity(kf.errorCovPost, Scalar::all(1));

    Mat m(2, 1, CV_32FC1);
    m.at<float>(0, 0) = 10.0f;
    m.at<float>(1, 0) = 20.0f;
    Mat pred = kf.predict();
    Mat corr = kf.correct(m);
    expect_true(pred.rows == 4 && pred.cols == 1, "kalman predict shape mismatch");
    expect_true(corr.rows == 4 && corr.cols == 1, "kalman correct shape mismatch");
    expect_true(isfinite(corr.at<float>(0, 0)) && isfinite(corr.at<float>(1, 0)), "kalman output non-finite");

    Mat bgr(1, 1, CV_8UC3);
    bgr.at<cv8uc3_t>(0, 0).c1 = 10;
    bgr.at<cv8uc3_t>(0, 0).c2 = 20;
    bgr.at<cv8uc3_t>(0, 0).c3 = 30;

    Mat hsv_from_bgr;
    cvtColor(bgr, hsv_from_bgr, CV_BGR2HSV);
    expect_true(hsv_from_bgr.rows == 1 && hsv_from_bgr.cols == 1, "BGR2HSV shape mismatch");

    Mat hsv_from_rgb;
    cvtColor(bgr, hsv_from_rgb, CV_RGB2HSV);
    expect_true(hsv_from_rgb.rows == 1 && hsv_from_rgb.cols == 1, "RGB2HSV shape mismatch");

    Mat bgr_out;
    cvtColor(hsv_from_bgr, bgr_out, CV_HSV2BGR);
    expect_true(bgr_out.rows == 1 && bgr_out.cols == 1, "HSV2BGR shape mismatch");

    Mat rgb_out;
    cvtColor(hsv_from_bgr, rgb_out, CV_HSV2RGB);
    expect_true(rgb_out.rows == 1 && rgb_out.cols == 1, "HSV2RGB shape mismatch");
}

void unit_test_coverage()
{
    cout << "=== Coverage Supplement Test ===" << endl;
    test_threshold_modes();
    test_box_filter_and_geometry();
    test_hungarian_and_random();
    test_kalman_and_hsv_paths();
    cout << "✓ Coverage supplement test passed!" << endl;
}
