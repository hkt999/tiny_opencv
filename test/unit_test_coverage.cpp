#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
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

static void expect_hue_near(uchar got, int expect, int tol, const char *msg)
{
    int g = (int)got;
    int d = abs(g - expect);
    if (d > 128) {
        d = 256 - d;
    }
    if (d > tol) {
        fail_test(msg);
    }
}

static void expect_float_near(float got, float expect, float tol, const char *msg)
{
    if (fabs(got - expect) > tol) {
        fail_test(msg);
    }
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

extern int rgb24_to_yuv420(int x_dim, int y_dim, unsigned char *bmp, unsigned char *yuv, int flip);
extern void rgb24_to_yuv420p(uint8_t *lum, uint8_t *cb, uint8_t *cr, uint8_t *src, int width, int height);
extern void yuv422p_to_rgb24(unsigned char *yuv422p, unsigned char *rgb, int width, int height);

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

static void test_bilateral_stability()
{
    Mat src(7, 7, CV_8UC1);
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            src.at<uchar>(r, c) = (uchar)(r * 10 + c);
        }
    }
    Mat dst;
    bilateralFilter(src, dst, 5, 0.0, 0.0);
    expect_true(dst.rows == src.rows && dst.cols == src.cols, "bilateral c1 shape mismatch");
    expect_true(dst.at<uchar>(3, 3) == src.at<uchar>(3, 3), "bilateral c1 sigma-zero center mismatch");

    Mat src3(7, 7, CV_8UC3);
    for (int r = 0; r < src3.rows; r++) {
        for (int c = 0; c < src3.cols; c++) {
            src3.at<cv8uc3_t>(r, c).c1 = (uchar)(r + c);
            src3.at<cv8uc3_t>(r, c).c2 = (uchar)(r * 3 + c);
            src3.at<cv8uc3_t>(r, c).c3 = (uchar)(200 - r - c);
        }
    }
    Mat dst3;
    bilateralFilter(src3, dst3, 5, 0.0, 0.0);
    cv8uc3_t center = dst3.at<cv8uc3_t>(3, 3);
    cv8uc3_t center_src = src3.at<cv8uc3_t>(3, 3);
    expect_true(center.c1 == center_src.c1, "bilateral c3 sigma-zero center c1 mismatch");
    expect_true(center.c2 == center_src.c2, "bilateral c3 sigma-zero center c2 mismatch");
    expect_true(center.c3 == center_src.c3, "bilateral c3 sigma-zero center c3 mismatch");
}

static void test_filter2d_kernel_types()
{
    Mat src(5, 5, CV_8UC1);
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            src.at<uchar>(r, c) = (uchar)(r * 10 + c);
        }
    }

    Mat dst;
    Mat k8(3, 3, CV_8UC1);
    memset(k8.ref->data, 0, 9);
    k8.at<uchar>(1, 1) = 1;
    filter2D(src, dst, -1, k8);
    expect_true(dst.at<uchar>(2, 2) == src.at<uchar>(2, 2), "filter2D CV_8UC1 kernel center mismatch");

    Mat k32s(3, 3, CV_32SC1);
    for (int i = 0; i < 9; i++) {
        k32s.at<int32_t>(i) = 0;
    }
    k32s.at<int32_t>(1, 1) = 8192; // internal fixed-point scale (MUL)
    filter2D(src, dst, -1, k32s);
    expect_true(dst.at<uchar>(2, 2) == src.at<uchar>(2, 2), "filter2D CV_32SC1 kernel center mismatch");

    Mat k64(3, 3, CV_64FC1);
    for (int i = 0; i < 9; i++) {
        k64.at<double>(i) = 0.0;
    }
    k64.at<double>(1, 1) = 1.0;
    filter2D(src, dst, -1, k64);
    expect_true(dst.at<uchar>(2, 2) == src.at<uchar>(2, 2), "filter2D CV_64FC1 kernel center mismatch");
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

static void permute_min_cost_square(
    const Mat &cost, int row, vector<int> &used, float running,
    float &best_cost, vector<int> &assign, vector<int> &best_assign)
{
    int n = cost.rows;
    if (row == n) {
        if (running < best_cost) {
            best_cost = running;
            best_assign = assign;
        }
        return;
    }

    for (int c = 0; c < n; c++) {
        if (used[c]) {
            continue;
        }
        used[c] = 1;
        assign[row] = c;
        permute_min_cost_square(
            cost, row + 1, used, running + cost.at<float>(row, c),
            best_cost, assign, best_assign);
        used[c] = 0;
    }
}

static float brute_force_square_min_cost(const Mat &cost, vector<int> &best_assign)
{
    int n = cost.rows;
    vector<int> used(n, 0);
    vector<int> assign(n, -1);
    float best_cost = numeric_limits<float>::max();
    permute_min_cost_square(cost, 0, used, 0.0f, best_cost, assign, best_assign);
    return best_cost;
}

static void test_hungarian_exhaustive()
{
    HungarianAlgorithm solver;

    Mat cost_2x3(2, 3, CV_32FC1);
    cost_2x3.at<float>(0, 0) = 8; cost_2x3.at<float>(0, 1) = 3; cost_2x3.at<float>(0, 2) = 4;
    cost_2x3.at<float>(1, 0) = 6; cost_2x3.at<float>(1, 1) = 5; cost_2x3.at<float>(1, 2) = 2;
    Mat assign_2x3(2, 1, CV_32FC1);
    double cost_a = solver.Solve(cost_2x3, assign_2x3);
    expect_true(fabs(cost_a - 5.0) < 1e-5, "hungarian 2x3 cost mismatch");
    expect_true((int)assign_2x3.at<float>(0, 0) == 1, "hungarian 2x3 row0 mismatch");
    expect_true((int)assign_2x3.at<float>(1, 0) == 2, "hungarian 2x3 row1 mismatch");

    Mat tie(3, 3, CV_32FC1);
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            tie.at<float>(r, c) = 1.0f;
        }
    }
    Mat tie_assign(3, 1, CV_32FC1);
    double tie_cost = solver.Solve(tie, tie_assign);
    expect_true(fabs(tie_cost - 3.0) < 1e-5, "hungarian tie cost mismatch");

    Mat neg(2, 2, CV_32FC1);
    neg.at<float>(0, 0) = -1.0f; neg.at<float>(0, 1) = 2.0f;
    neg.at<float>(1, 0) = 3.0f;  neg.at<float>(1, 1) = 4.0f;
    Mat neg_assign(2, 1, CV_32FC1);
    std::ostringstream sink;
    std::streambuf *old_cerr = std::cerr.rdbuf(sink.rdbuf());
    double neg_cost = solver.Solve(neg, neg_assign);
    std::cerr.rdbuf(old_cerr);
    expect_true(isfinite(neg_cost), "hungarian negative-cost path produced non-finite");

    Mat square(4, 4, CV_32FC1);
    float vals[16] = {
        9, 2, 7, 8,
        6, 4, 3, 7,
        5, 8, 1, 8,
        7, 6, 9, 4
    };
    for (int i = 0; i < 16; i++) {
        square.at<float>(i) = vals[i];
    }
    Mat square_assign(4, 1, CV_32FC1);
    double square_cost = solver.Solve(square, square_assign);
    vector<int> brute_assign;
    float brute_cost = brute_force_square_min_cost(square, brute_assign);
    expect_true(fabs(square_cost - brute_cost) < 1e-5, "hungarian brute-force square cost mismatch");
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

static void test_kalman_control_and_init_paths()
{
    KalmanFilter kf_default;
    kf_default.init(2, 1, -1, CV_32F);
    expect_true(kf_default.controlMatrix.empty(), "kalman init CP<0 should release control matrix");

    Mat z(1, 1, CV_32FC1);
    z.at<float>(0, 0) = 1.0f;
    Mat p0 = kf_default.predict();
    Mat c0 = kf_default.correct(z);
    expect_true(p0.rows == 2 && p0.cols == 1, "kalman default init predict shape mismatch");
    expect_true(c0.rows == 2 && c0.cols == 1, "kalman default init correct shape mismatch");

    KalmanFilter kf_ctrl(2, 1, 1, CV_32F);
    setIdentity(kf_ctrl.transitionMatrix);
    setIdentity(kf_ctrl.measurementMatrix);
    setIdentity(kf_ctrl.processNoiseCov, Scalar::all(1e-3));
    setIdentity(kf_ctrl.measurementNoiseCov, Scalar::all(1e-2));
    setIdentity(kf_ctrl.errorCovPost, Scalar::all(1));
    kf_ctrl.controlMatrix.at<float>(0, 0) = 1.0f;
    kf_ctrl.controlMatrix.at<float>(1, 0) = 0.5f;

    Mat control(1, 1, CV_32FC1);
    control.at<float>(0, 0) = 2.0f;
    Mat pc = kf_ctrl.predict(control);
    expect_float_near(pc.at<float>(0, 0), 2.0f, 1e-4f, "kalman control x update mismatch");
    expect_float_near(pc.at<float>(1, 0), 1.0f, 1e-4f, "kalman control y update mismatch");
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

static void test_yuv_low_level_paths()
{
    const int w = 2, h = 2;
    unsigned char rgb[12] = {
        255, 0, 0, 255, 0, 0,
        255, 0, 0, 255, 0, 0
    };
    unsigned char bgr[12] = {
        0, 0, 255, 0, 0, 255,
        0, 0, 255, 0, 0, 255
    };

    unsigned char yuv_from_rgb[6] = {0};
    unsigned char yuv_from_bgr[6] = {0};
    rgb24_to_yuv420(w, h, rgb, yuv_from_rgb, 0);
    rgb24_to_yuv420(w, h, bgr, yuv_from_bgr, 1);
    for (int i = 0; i < 6; i++) {
        expect_true(yuv_from_rgb[i] == yuv_from_bgr[i], "rgb24_to_yuv420 flip path mismatch");
    }

    uint8_t lum[4] = {0};
    uint8_t cb[1] = {0};
    uint8_t cr[1] = {0};
    rgb24_to_yuv420p(lum, cb, cr, rgb, w, h);
    for (int i = 0; i < 4; i++) expect_uchar_near(lum[i], 76, 2, "rgb24_to_yuv420p Y mismatch");
    expect_uchar_near(cb[0], 84, 4, "rgb24_to_yuv420p Cb mismatch");
    expect_uchar_near(cr[0], 255, 2, "rgb24_to_yuv420p Cr mismatch");

    unsigned char yuv422[8] = {
        16, 235, 16, 235, // Y
        128, 128,         // U
        128, 128          // V
    };
    unsigned char rgb_out[12] = {0};
    yuv422p_to_rgb24(yuv422, rgb_out, 2, 2);
    expect_true(rgb_out[0] <= rgb_out[3], "yuv422 gray ramp mismatch");
    expect_true(rgb_out[1] <= rgb_out[4], "yuv422 gray ramp mismatch");
    expect_true(rgb_out[2] <= rgb_out[5], "yuv422 gray ramp mismatch");
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

    Mat gray(1, 1, CV_8UC3);
    gray.at<cv8uc3_t>(0, 0).c1 = 120;
    gray.at<cv8uc3_t>(0, 0).c2 = 120;
    gray.at<cv8uc3_t>(0, 0).c3 = 120;
    Mat hsv_gray;
    cvtColor(gray, hsv_gray, CV_BGR2HSV);
    cv8uc3_t gray_hsv = hsv_gray.at<cv8uc3_t>(0, 0);
    expect_true(gray_hsv.c1 == 0, "BGR2HSV gray hue should be 0");
    expect_true(gray_hsv.c2 == 0, "BGR2HSV gray saturation should be 0");
    expect_uchar_near(gray_hsv.c3, 120, 1, "BGR2HSV gray value mismatch");

    Mat black(1, 1, CV_8UC3);
    black.at<cv8uc3_t>(0, 0).c1 = 0;
    black.at<cv8uc3_t>(0, 0).c2 = 0;
    black.at<cv8uc3_t>(0, 0).c3 = 0;
    Mat hsv_black;
    cvtColor(black, hsv_black, CV_BGR2HSV);
    cv8uc3_t black_hsv = hsv_black.at<cv8uc3_t>(0, 0);
    expect_true(black_hsv.c1 == 0 && black_hsv.c2 == 0 && black_hsv.c3 == 0, "BGR2HSV black mismatch");

    struct rgb_h_case_t { uchar r; uchar g; uchar b; int exp_h; };
    rgb_h_case_t rgb_cases[] = {
        {255,   0, 120, 235}, // r max, g min
        {255, 120,   0,  20}, // r max, g != min
        {120, 255,   0,  65}, // g max, b min
        {  0, 255, 120, 105}, // g max, b != min
        {  0, 120, 255, 150}, // b max, r min
        {120,   0, 255, 190}  // b max, r != min
    };
    for (int i = 0; i < 6; i++) {
        Mat rgb(1, 1, CV_8UC3);
        rgb.at<cv8uc3_t>(0, 0).c1 = rgb_cases[i].r;
        rgb.at<cv8uc3_t>(0, 0).c2 = rgb_cases[i].g;
        rgb.at<cv8uc3_t>(0, 0).c3 = rgb_cases[i].b;
        Mat hsv_from_case;
        cvtColor(rgb, hsv_from_case, CV_RGB2HSV);
        cv8uc3_t hv = hsv_from_case.at<cv8uc3_t>(0, 0);
        expect_hue_near(hv.c1, rgb_cases[i].exp_h, 5, "RGB2HSV hue branch mismatch");
        expect_true(hv.c2 > 130, "RGB2HSV saturation should be high");
        expect_true(hv.c3 > 200, "RGB2HSV value should be high");
    }

    int hue_samples[] = {0, 43, 85, 128, 170, 213, 255};
    for (int i = 0; i < 7; i++) {
        Mat hsv1(1, 1, CV_8UC3);
        hsv1.at<cv8uc3_t>(0, 0).c1 = (uchar)hue_samples[i];
        hsv1.at<cv8uc3_t>(0, 0).c2 = 255;
        hsv1.at<cv8uc3_t>(0, 0).c3 = 255;
        Mat rgb1, hsv2;
        cvtColor(hsv1, rgb1, CV_HSV2RGB);
        cvtColor(rgb1, hsv2, CV_RGB2HSV);
        cv8uc3_t hv = hsv2.at<cv8uc3_t>(0, 0);
        int expect_h = (hue_samples[i] == 255) ? 0 : hue_samples[i];
        expect_hue_near(hv.c1, expect_h, 6, "HSV->RGB->HSV hue mismatch");
        expect_uchar_near(hv.c2, 255, 4, "HSV->RGB->HSV saturation mismatch");
        expect_uchar_near(hv.c3, 255, 4, "HSV->RGB->HSV value mismatch");
    }
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

    Mat src_f(2, 2, CV_32FC3);
    float *src_f_ptr = src_f.getData<float>();
    for (int i = 0; i < 4; i++) {
        src_f_ptr[i * 3 + 0] = (float)(i + 1);
        src_f_ptr[i * 3 + 1] = (float)(i + 11);
        src_f_ptr[i * 3 + 2] = (float)(i + 21);
    }
    Mat fch[3];
    split(src_f, fch, 3);
    expect_float_near(fch[1].at<float>(1, 1), 14.0f, 1e-6f, "split float channel mismatch");
    Mat fmerge;
    merge(fch, 3, fmerge);
    float *fm_ptr = fmerge.getData<float>();
    expect_float_near(fm_ptr[(1 * 2 + 1) * 3 + 2], 24.0f, 1e-6f, "merge float channel mismatch");

    EXPECT_THROW(split(Mat(), ch, 3), "split should reject empty input");
    EXPECT_THROW(split(src, (Mat *)0, 3), "split should reject null destination array");
    uchar unsupported_src_data[1] = {7};
    Mat unsupported_src(1, 1, 7, unsupported_src_data);
    Mat unsupported_dst[1];
    EXPECT_THROW(split(unsupported_src, unsupported_dst, 1), "split should reject unsupported depth");

    EXPECT_THROW(merge((Mat *)0, 1, merged), "merge should reject null source array");
    EXPECT_THROW(merge(ch, 0, merged), "merge should reject invalid count");
    EXPECT_THROW(merge(ch, 5, merged), "merge should reject too many channels");
    Mat empty_ch[1];
    EXPECT_THROW(merge(empty_ch, 1, merged), "merge should reject empty channel 0");
    Mat multi_ch[1];
    multi_ch[0] = src;
    EXPECT_THROW(merge(multi_ch, 1, merged), "merge should reject multi-channel source");
    Mat type_mismatch[2];
    type_mismatch[0] = Mat(2, 2, CV_8UC1);
    type_mismatch[1] = Mat(2, 2, CV_32FC1);
    EXPECT_THROW(merge(type_mismatch, 2, merged), "merge should reject channel type mismatch");
    uchar unsupported_merge_data[1] = {9};
    Mat unsupported_merge[1];
    unsupported_merge[0] = Mat(1, 1, 7, unsupported_merge_data);
    EXPECT_THROW(merge(unsupported_merge, 1, merged), "merge should reject unsupported depth");
}

static void test_randn_double_path()
{
    Mat noise64(64, 1, CV_64FC1);
    randn(noise64, 3.0f, 0.5f);
    double first = noise64.at<double>(0, 0);
    bool has_diff = false;
    for (int i = 0; i < noise64.rows; i++) {
        double v = noise64.at<double>(i, 0);
        expect_true(isfinite(v), "randn CV_64F should produce finite output");
        if (v != first) {
            has_diff = true;
        }
    }
    expect_true(has_diff, "randn CV_64F produced constant output");
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

    Mat c3(2, 2, CV_8UC3);
    c3.at<cv8uc3_t>(0, 0).c1 = 0;   c3.at<cv8uc3_t>(0, 0).c2 = 0;   c3.at<cv8uc3_t>(0, 0).c3 = 0;
    c3.at<cv8uc3_t>(0, 1).c1 = 100; c3.at<cv8uc3_t>(0, 1).c2 = 20;  c3.at<cv8uc3_t>(0, 1).c3 = 10;
    c3.at<cv8uc3_t>(1, 0).c1 = 20;  c3.at<cv8uc3_t>(1, 0).c2 = 150; c3.at<cv8uc3_t>(1, 0).c3 = 40;
    c3.at<cv8uc3_t>(1, 1).c1 = 255; c3.at<cv8uc3_t>(1, 1).c2 = 255; c3.at<cv8uc3_t>(1, 1).c3 = 255;
    Mat c3_linear;
    resize(c3, c3_linear, Size(3, 3), 0.0f, 0.0f, INTER_LINEAR);
    cv8uc3_t c = c3_linear.at<cv8uc3_t>(1, 1);
    expect_uchar_near(c.c1, 94, 4, "linear c3 center channel0 mismatch");
    expect_uchar_near(c.c2, 106, 4, "linear c3 center channel1 mismatch");
    expect_uchar_near(c.c3, 76, 4, "linear c3 center channel2 mismatch");

    Mat one(1, 1, CV_8UC1);
    one.at<uchar>(0, 0) = 123;
    Mat one_up;
    resize(one, one_up, Size(4, 4), 0.0f, 0.0f, INTER_LINEAR);
    expect_true(one_up.at<uchar>(3, 3) == 123, "linear 1x1 expand mismatch");

    Mat half;
    resize(src, half, Size(), 0.5f, 0.5f, INTER_NEAREST);
    expect_true(half.rows == 1 && half.cols == 1, "ratio downscale shape mismatch");
    expect_true(half.at<uchar>(0, 0) == 0, "ratio downscale value mismatch");

    Mat bad_type_src(2, 2, CV_32FC1);
    EXPECT_THROW(resize(bad_type_src, nearest, Size(4, 4), 0.0f, 0.0f, INTER_NEAREST), "resize nearest should reject unsupported source type");
    EXPECT_THROW(resize(bad_type_src, nearest, Size(4, 4), 0.0f, 0.0f, INTER_LINEAR), "resize linear should reject unsupported source type");

    EXPECT_THROW(resize(Mat(), nearest, Size(4, 4), 0.0f, 0.0f, INTER_NEAREST), "resize should reject empty input");
    EXPECT_THROW(resize(src, nearest, Size(0, 4), 0.0f, 0.0f, INTER_NEAREST), "resize should reject non-positive output size");
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
    EXPECT_THROW(cvtColor(src8, dst, CV_RGB2GRAY), "cvtColor should reject wrong src type for RGB2GRAY");
    EXPECT_THROW(cvtColor(src_bgr, dst, CV_GRAY2BGR), "cvtColor should reject wrong src type for GRAY2BGR");
    EXPECT_THROW(cvtColor(src8, dst, CV_BGR2YUV_I420), "cvtColor should reject wrong src type for BGR2YUV");
    EXPECT_THROW(cvtColor(src8, dst, CV_RGB2YUV_I420), "cvtColor should reject wrong src type for RGB2YUV");
    EXPECT_THROW(cvtColor(src_bgr, dst, CV_YUV2BGR_I420), "cvtColor should reject wrong src type for YUV2BGR");
    EXPECT_THROW(cvtColor(src_bgr, dst, CV_YUV2RGB_I420), "cvtColor should reject wrong src type for YUV2RGB");
    EXPECT_THROW(cvtColor(src8, dst, CV_RGB2BGR), "cvtColor should reject wrong src type for RGB2BGR");
    EXPECT_THROW(cvtColor(src8, dst, CV_BGR2HSV), "cvtColor should reject wrong src type for BGR2HSV");
    EXPECT_THROW(cvtColor(src8, dst, CV_RGB2HSV), "cvtColor should reject wrong src type for RGB2HSV");
    EXPECT_THROW(cvtColor(src8, dst, CV_HSV2BGR), "cvtColor should reject wrong src type for HSV2BGR");
    EXPECT_THROW(cvtColor(src8, dst, CV_HSV2RGB), "cvtColor should reject wrong src type for HSV2RGB");
    EXPECT_THROW(cvtColor(src_bgr, dst, 999), "cvtColor should reject unknown code");
    EXPECT_THROW(cvtColor(Mat(), dst, CV_BGR2GRAY), "cvtColor should reject empty input");

    Mat odd(3, 3, CV_8UC3);
    EXPECT_THROW(cvtColor(odd, dst, CV_BGR2YUV_I420), "cvtColor should reject odd-size I420 input");
    EXPECT_THROW(cvtColor(odd, dst, CV_RGB2YUV_I420), "cvtColor should reject odd-size I420 input for RGB");

    Mat invalid_i420(5, 2, CV_8UC1);
    EXPECT_THROW(cvtColor(invalid_i420, dst, CV_YUV2BGR_I420), "cvtColor should reject invalid I420 layout for YUV2BGR");
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

    Mat c3(2, 2, CV_8UC3);
    EXPECT_THROW(c3.transpose(), "transpose should reject unsupported 3-channel type");

    Mat nonsquare(2, 3, CV_32FC1);
    EXPECT_THROW(nonsquare.inverse(), "inverse should reject non-square matrix");

    Mat nonf(2, 2, CV_8UC1);
    expect_true(nonf.determinant() == 0.0f, "determinant non-float fallback mismatch");

    Mat a8(2, 2, CV_8UC1), b32(2, 2, CV_32FC1);
    EXPECT_THROW(a8 - b32, "operator- should reject type mismatch");

    EXPECT_THROW(Mat::ones(2, 2, CV_64FC1), "ones should reject unsupported type");
    EXPECT_THROW(Mat::eye(2, 2, CV_8UC3), "eye should reject multi-channel type");

    Mat bad_roi_src(2, 2, CV_16UC1);
    EXPECT_THROW(Mat(bad_roi_src, Rect(0, 0, 1, 1)), "roi constructor should reject unsupported type");

    uchar *external = (uchar *)malloc(4);
    external[0] = 1; external[1] = 2; external[2] = 3; external[3] = 4;
    {
        Mat wrapped(2, 2, CV_8UC1, external);
        expect_true(wrapped.ref != 0, "wrapped mat ref should not be null");
        expect_true(wrapped.ref->count == 1, "wrapped mat initial refcount mismatch");

        Mat cpy = wrapped;
        expect_true(wrapped.ref->count == 2, "wrapped mat copy refcount mismatch");
        cpy.release();
        expect_true(wrapped.ref->count == 1, "wrapped mat refcount after release mismatch");

        Mat assigned;
        assigned = wrapped;
        expect_true(wrapped.ref->count == 2, "wrapped mat assign refcount mismatch");
        assigned.release();
        expect_true(wrapped.ref->count == 1, "wrapped mat refcount after assign release mismatch");
    }
    external[0] = 9;
    expect_true(external[0] == 9, "wrapped external buffer should remain valid");
    free(external);

    Mat size_ctor(Size(3, 2), CV_8UC1);
    expect_true(size_ctor.rows == 2 && size_ctor.cols == 3, "size constructor shape mismatch");

    uchar ext_by_size_data[6] = {1, 2, 3, 4, 5, 6};
    Mat ext_by_size(Size(3, 2), CV_8UC1, ext_by_size_data);
    expect_true(ext_by_size.at<uchar>(1, 2) == 6, "size external constructor mismatch");

    Mat roi_c3_src(3, 3, CV_8UC3);
    for (int i = 0; i < 9; i++) {
        roi_c3_src.at<cv8uc3_t>(i).c1 = (uchar)(i + 1);
        roi_c3_src.at<cv8uc3_t>(i).c2 = (uchar)(i + 11);
        roi_c3_src.at<cv8uc3_t>(i).c3 = (uchar)(i + 21);
    }
    Mat roi_c3(roi_c3_src, Rect(1, 1, 2, 2));
    expect_true(roi_c3.at<cv8uc3_t>(0, 0).c1 == roi_c3_src.at<cv8uc3_t>(1, 1).c1, "roi CV_8UC3 constructor mismatch");

    Mat roi_f_src(3, 3, CV_32FC1);
    for (int i = 0; i < 9; i++) {
        roi_f_src.at<float>(i) = (float)(i * 0.5f);
    }
    Mat roi_f(roi_f_src, Rect(1, 0, 2, 2));
    expect_float_near(roi_f.at<float>(1, 1), roi_f_src.at<float>(1, 2), 1e-6f, "roi CV_32FC1 constructor mismatch");

    Mat roi_d_src(3, 3, CV_64FC1);
    for (int i = 0; i < 9; i++) {
        roi_d_src.at<double>(i) = (double)i + 0.25;
    }
    Mat roi_d(roi_d_src, Rect(0, 1, 2, 2));
    expect_true(fabs(roi_d.at<double>(0, 1) - roi_d_src.at<double>(1, 1)) < 1e-9, "roi CV_64FC1 constructor mismatch");

    Mat add_a(1, 2, CV_32FC1);
    Mat add_b(1, 2, CV_32FC1);
    add_a.at<float>(0, 0) = 1.5f; add_a.at<float>(0, 1) = 2.5f;
    add_b.at<float>(0, 0) = 3.0f; add_b.at<float>(0, 1) = 4.0f;
    add_a += add_b;
    expect_float_near(add_a.at<float>(0, 0), 4.5f, 1e-6f, "operator+= value mismatch");
    expect_float_near(add_a.at<float>(0, 1), 6.5f, 1e-6f, "operator+= value mismatch");
    Mat add_mismatch(2, 1, CV_32FC1);
    EXPECT_THROW(add_a += add_mismatch, "operator+= should reject shape mismatch");

    Mat add_dim_a(1, 2, CV_32FC1);
    Mat add_dim_b(2, 2, CV_32FC1);
    EXPECT_THROW(add_dim_a + add_dim_b, "operator+ should reject dimension mismatch");

    Mat mm32(2, 2, CV_32FC1);
    Mat mm64(2, 2, CV_64FC1);
    EXPECT_THROW(mm32 * mm64, "operator* should reject type mismatch");
    Mat mm8a(2, 2, CV_8UC1), mm8b(2, 2, CV_8UC1);
    EXPECT_THROW(mm8a * mm8b, "operator* should reject unsupported non-float type");

    Mat t_u8_src(2, 3, CV_8U);
    for (int i = 0; i < 6; i++) {
        t_u8_src.at<uchar>(i) = (uchar)(i + 1);
    }
    Mat *t_u8 = &t_u8_src.transpose();
    expect_true(t_u8->rows == 3 && t_u8->cols == 2, "transpose CV_8U shape mismatch");
    expect_true(t_u8->at<uchar>(2, 1) == t_u8_src.at<uchar>(1, 2), "transpose CV_8U value mismatch");
    delete t_u8;

    Mat t_d_src(2, 2, CV_64F);
    t_d_src.at<double>(0, 0) = 1.0;
    t_d_src.at<double>(0, 1) = 2.0;
    t_d_src.at<double>(1, 0) = 3.0;
    t_d_src.at<double>(1, 1) = 4.0;
    Mat *t_d = &t_d_src.transpose();
    expect_true(t_d->at<double>(0, 1) == 3.0, "transpose CV_64F mismatch");
    delete t_d;

    Mat det1(1, 1, CV_32FC1);
    det1.at<float>(0, 0) = 7.5f;
    expect_float_near(det1.determinant(), 7.5f, 1e-6f, "determinant 1x1 mismatch");
}

static void test_mat_cofactor_inverse_numeric()
{
    Mat m1(1, 1, CV_32FC1);
    m1.at<float>(0, 0) = 4.0f;
    Mat *co1 = &m1.cofactor_();
    expect_float_near(co1->at<float>(0, 0), 4.0f, 1e-6f, "cofactor 1x1 mismatch");
    delete co1;
    Mat *inv1 = &m1.inverse();
    expect_float_near(inv1->at<float>(0, 0), 0.25f, 1e-6f, "inverse 1x1 mismatch");
    delete inv1;

    Mat m2(2, 2, CV_32FC1);
    m2.at<float>(0, 0) = 1.0f; m2.at<float>(0, 1) = 2.0f;
    m2.at<float>(1, 0) = 3.0f; m2.at<float>(1, 1) = 4.0f;
    Mat *co2 = &m2.cofactor_();
    expect_float_near(co2->at<float>(0, 0), 4.0f, 1e-6f, "cofactor 2x2 (0,0) mismatch");
    expect_float_near(co2->at<float>(0, 1), -3.0f, 1e-6f, "cofactor 2x2 (0,1) mismatch");
    expect_float_near(co2->at<float>(1, 0), -2.0f, 1e-6f, "cofactor 2x2 (1,0) mismatch");
    expect_float_near(co2->at<float>(1, 1), 1.0f, 1e-6f, "cofactor 2x2 (1,1) mismatch");
    delete co2;
    Mat *inv2 = &m2.inverse();
    expect_float_near(inv2->at<float>(0, 0), -2.0f, 1e-5f, "inverse 2x2 (0,0) mismatch");
    expect_float_near(inv2->at<float>(0, 1), 1.0f, 1e-5f, "inverse 2x2 (0,1) mismatch");
    expect_float_near(inv2->at<float>(1, 0), 1.5f, 1e-5f, "inverse 2x2 (1,0) mismatch");
    expect_float_near(inv2->at<float>(1, 1), -0.5f, 1e-5f, "inverse 2x2 (1,1) mismatch");
    delete inv2;

    Mat m3(3, 3, CV_32FC1);
    float v[9] = {1, 2, 3, 0, 1, 4, 5, 6, 0};
    for (int i = 0; i < 9; i++) {
        m3.at<float>(i) = v[i];
    }
    Mat *co3 = &m3.cofactor_();
    expect_float_near(co3->at<float>(0, 0), -24.0f, 1e-5f, "cofactor 3x3 (0,0) mismatch");
    expect_float_near(co3->at<float>(0, 1), 20.0f, 1e-5f, "cofactor 3x3 (0,1) mismatch");
    expect_float_near(co3->at<float>(2, 2), 1.0f, 1e-5f, "cofactor 3x3 (2,2) mismatch");
    delete co3;
    Mat *inv3 = &m3.inverse();
    expect_float_near(inv3->at<float>(0, 0), -24.0f, 1e-4f, "inverse 3x3 (0,0) mismatch");
    expect_float_near(inv3->at<float>(0, 1), 18.0f, 1e-4f, "inverse 3x3 (0,1) mismatch");
    expect_float_near(inv3->at<float>(0, 2), 5.0f, 1e-4f, "inverse 3x3 (0,2) mismatch");
    expect_float_near(inv3->at<float>(2, 2), 1.0f, 1e-4f, "inverse 3x3 (2,2) mismatch");
    delete inv3;

    Mat ns(2, 3, CV_32FC1);
    Mat *co_ns = &ns.cofactor_();
    expect_true(co_ns->rows == 2 && co_ns->cols == 3, "cofactor non-square shape mismatch");
    delete co_ns;
}

static void test_mat_refcount_stress()
{
    Mat base(4, 4, CV_8UC1);
    for (int i = 0; i < 16; i++) {
        base.at<uchar>(i) = (uchar)i;
    }

    Mat refs[64];
    for (int i = 0; i < 64; i++) {
        refs[i] = base;
    }
    expect_true(base.ref->count == 65, "mat stress initial refcount mismatch");

    for (int i = 0; i < 64; i += 2) {
        refs[i].release();
    }
    expect_true(base.ref->count == 33, "mat stress half-release refcount mismatch");

    for (int i = 1; i < 64; i += 2) {
        refs[i].release();
    }
    expect_true(base.ref->count == 1, "mat stress full-release refcount mismatch");

    uchar *raw = (uchar *)malloc(16);
    for (int i = 0; i < 16; i++) {
        raw[i] = (uchar)(200 + i);
    }
    Mat wrapped(4, 4, CV_8UC1, raw);
    for (int iter = 0; iter < 200; iter++) {
        Mat c1 = wrapped;
        Mat c2;
        c2 = c1;
        Mat roi_copy(c1, Rect(1, 1, 2, 2));
        expect_true(roi_copy.at<uchar>(0, 0) == raw[5], "mat stress roi copy mismatch");
        c2.release();
        c1.release();
    }
    expect_true(wrapped.ref->count == 1, "mat stress wrapped final refcount mismatch");
    wrapped.release();
    raw[0] = 17;
    expect_true(raw[0] == 17, "mat stress external buffer should stay writable");
    free(raw);
}

void unit_test_coverage()
{
    cout << "=== Coverage Supplement Test ===" << endl;
    test_threshold_modes();
    test_bilateral_stability();
    test_filter2d_kernel_types();
    test_box_filter_and_geometry();
    test_hungarian_and_random();
    test_randn_double_path();
    test_hungarian_exhaustive();
    test_kalman_noninteractive();
    test_kalman_control_and_init_paths();
    test_split_merge();
    test_resize_modes_and_errors();
    test_error_paths();
    test_mat_semantics();
    test_mat_cofactor_inverse_numeric();
    test_mat_refcount_stress();
    test_yuv_i420_numeric();
    test_yuv_low_level_paths();
    test_hsv_numeric();
    cout << "✓ Coverage supplement test passed!" << endl;
}
