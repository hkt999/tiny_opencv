#include "kalman_filter.hpp"

using namespace KCV;

int main() {
    // State: [x, y, vx, vy], Measurement: [x, y]
    KalmanFilter kf(4, 2);
    float dt = 1.0f;
    kf.transitionMatrix.at<float>(0, 2) = dt;
    kf.transitionMatrix.at<float>(1, 3) = dt;

    // Map state -> measurement
    kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf.measurementMatrix.at<float>(1, 1) = 1.0f;

    float x = 1.0f;
    float y = 2.0f;
    Mat measurement(2, 1, CV_32FC1);
    measurement.at<float>(0, 0) = x;
    measurement.at<float>(1, 0) = y;

    Mat prediction = kf.predict();
    (void)prediction;
    Mat estimate = kf.correct(measurement);
    (void)estimate;

    return 0;
}
