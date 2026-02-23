#include <cmath>
#include <cstdio>
#include "unit_test.hpp"
#include <tiny_opencv.hpp>
#include <kalman_filter.hpp>

using namespace kcv;

#define IMG_W   512
#define IMG_H   512

static inline float point_dist(point_t a, point_t b)
{
    float dx = (float)a.x - (float)b.x;
    float dy = (float)a.y - (float)b.y;
    return sqrtf(dx * dx + dy * dy);
}

static inline float angle_dist(float a, float b)
{
    float d = fabsf(a - b);
    while (d > (float)M_PI) {
        d = fabsf(d - (float)(2.0 * M_PI));
    }
    return d;
}

static inline point_t phi2xy(Mat &mat)
{
    point_t p;
    p.x = round(IMG_W / 2 + IMG_W / 3 * cos(mat.at<float>(0)));
    p.y = round(IMG_H / 2 - IMG_W / 3 * sin(mat.at<float>(0)));
    return p;
}

int unit_test_kalman_filter_mouse(void *data, draw_frame_t drawcb, get_observation_t observcb, should_stop_t should_stop, int max_steps)
{
    if (observcb == 0 || max_steps <= 0) {
        return -1;
    }

    int stateSize = 4;  // [x, y, v_x, v_y]
    int measSize = 2;   // [z_x, z_y]
    KalmanFilter KF(stateSize, measSize, 0, CV_32F);

    Mat state(stateSize, 1, CV_32F);
    Mat meas(measSize, 1, CV_32F);
    Mat estimated;

    setIdentity(KF.transitionMatrix);
    KF.transitionMatrix.at<float>(0, 2) = 1.0f;
    KF.transitionMatrix.at<float>(1, 3) = 1.0f;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-3));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
    setIdentity(KF.errorCovPost, Scalar::all(1));

    point_t start = observcb(data);
    state.at<float>(0) = (float)start.x;
    state.at<float>(1) = (float)start.y;
    state.at<float>(2) = 0.0f;
    state.at<float>(3) = 0.0f;
    state.copyTo(KF.statePost);

    int steps = 0;
    int samples = 0;
    float err_sum = 0.0f;
    const int warmup = 5;
    for (int i = 0; i < max_steps; i++) {
        state = KF.predict();
        point_t predicted = {(int)state.at<float>(0), (int)state.at<float>(1)};

        point_t observed = observcb(data);
        meas.at<float>(0) = (float)observed.x;
        meas.at<float>(1) = (float)observed.y;

        estimated = KF.correct(meas);
        point_t actual_to = {(int)estimated.at<float>(0), (int)estimated.at<float>(1)};

        if (drawcb) {
            drawcb(data, observed, predicted, actual_to);
        }

        if (i >= warmup) {
            err_sum += point_dist(actual_to, observed);
            samples++;
        }

        steps++;
        if (should_stop && should_stop(data)) {
            break;
        }
    }

    if (samples <= 0) {
        return -1;
    }

    float mean_err = err_sum / samples;
    printf("Kalman(mouse) steps=%d mean_err=%.3f\n", steps, mean_err);
    return (mean_err < 35.0f) ? 0 : -1;
}

int unit_test_kalman_filter_angle(void *data, draw_frame_t drawcb, should_stop_t should_stop, int max_steps)
{
    if (max_steps <= 0) {
        return -1;
    }

    KalmanFilter kalman(2, 1, 0);

    Mat x_k(2, 1, CV_32F);
    randn(x_k, 0.0, 0.1);
    Mat w_k(2, 1, CV_32F);
    Mat z_k = Mat::zeros(1, 1, CV_32F);

    float F[] = {1, 1, 0, 1};
    kalman.transitionMatrix = Mat(2, 2, CV_32F, F).clone();

    setIdentity(kalman.measurementMatrix, Scalar(1));
    setIdentity(kalman.processNoiseCov, Scalar(1e-5));
    setIdentity(kalman.measurementNoiseCov, Scalar(1e-1));
    setIdentity(kalman.errorCovPost, Scalar(1));
    randn(kalman.statePost, 0.0, 0.1);

    int steps = 0;
    int samples = 0;
    float err_sum = 0.0f;
    const int warmup = 10;
    for (int i = 0; i < max_steps; i++) {
        Mat y_k = kalman.predict();

        randn(z_k, 0.0, sqrt(static_cast<double>(kalman.measurementNoiseCov.at<float>(0, 0))));
        z_k = kalman.measurementMatrix * x_k + z_k;

        point_t observed = phi2xy(z_k);
        point_t predicted = phi2xy(y_k);
        point_t actual_to = phi2xy(x_k);
        if (drawcb) {
            drawcb(data, observed, predicted, actual_to);
        }

        Mat corrected = kalman.correct(z_k);
        if (i >= warmup) {
            err_sum += angle_dist(corrected.at<float>(0), x_k.at<float>(0));
            samples++;
        }

        randn(w_k, 0.0, sqrt(static_cast<double>(kalman.processNoiseCov.at<float>(0, 0))));
        x_k = kalman.transitionMatrix * x_k + w_k;

        steps++;
        if (should_stop && should_stop(data)) {
            break;
        }
    }

    if (samples <= 0) {
        return -1;
    }

    float mean_err = err_sum / samples;
    printf("Kalman(angle) steps=%d mean_err=%.3f rad\n", steps, mean_err);
    return (mean_err < 0.6f) ? 0 : -1;
}
