#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include "unit_test.hpp"

//#define PC_OPENCV

#ifdef PC_OPENCV
#include <opencv2/opencv.hpp>
#define phi2xy(mat) \
    cv::Point(cvRound(img.cols / 2 + img.cols / 3 * cos(mat.at<float>(0))), \
              cvRound(img.rows / 2 - img.cols / 3 * sin(mat.at<float>(0))))
using namespace cv;
#else
#include <math.h>
#include <tiny_opencv.hpp>
#include <kalman_filter.hpp>
#define IMG_W   1024
#define IMG_H   1024

using namespace kcv;

point_t phi2xy(Mat &mat)
{
    point_t p;
    p.x = round(IMG_W / 2 + IMG_W / 3 * cos(mat.at<float>(0)));
    p.y = round(IMG_H / 2 - IMG_W / 3 * sin(mat.at<float>(0)));
    return p;
}
#endif

using std::cout;
using std::endl;

void unit_test_kalman_filter(draw_frame_t drawcb, void *data)
{
    // Initialize, create Kalman filter object, window, random number
    // generator etc.
    //
    // Mat img(500, 500, CV_8UC3);
    KalmanFilter kalman(2, 1, 0);

    // state is (phi, delta_phi) - angle and angular velocity
    // Initialize with random guess.
    //
    Mat x_k(2, 1, CV_32F);
    randn(x_k, 0.0, 0.1);
    x_k.at<float>(0,0) = 0;
    x_k.at<float>(0,1) = 0.015814;

    // process noise
    //
    Mat w_k(2, 1, CV_32F);

    // measurements, only one parameter for angle
    //
    Mat z_k = Mat::zeros(1, 1, CV_32F);

    // Transition matrix 'F' describes relationship between
    // model parameters at step k and at step k+1 (this is
    // the "dynamics" in our model.
    //
    float F[] = {1, 1, 0, 1};
    kalman.transitionMatrix = Mat(2, 2, CV_32F, F).clone();

    // Initialize other Kalman filter parameters.
    //
    setIdentity(kalman.measurementMatrix, Scalar(1));
    setIdentity(kalman.processNoiseCov, Scalar(1e-5));
    setIdentity(kalman.measurementNoiseCov, Scalar(1e-1));
    setIdentity(kalman.errorCovPost, Scalar(1));

    // choose random initial state
    //
    randn(kalman.statePost, 0.0, 0.1);
    kalman.statePost.at<float>(0,0) = 0;
    kalman.statePost.at<float>(0,1) = 0.015814;

    for (;;) {
        // predict point position
        //
        Mat y_k = kalman.predict();

        // generate measurement (z_k)
        //
        randn(z_k, 0.0, sqrt(static_cast<double>(kalman.measurementNoiseCov.at<float>(0, 0))));
            
        z_k = kalman.measurementMatrix * x_k + z_k;

        // plot points (e.g., convert
        //
        #ifdef PC_OPENCV
        Mat img = Mat(600, 600, CV_8UC3);
        img = cv::Scalar::all(0);
        cv::circle(img, phi2xy(z_k), 4, cv::Scalar(128, 255, 255));  // observed
        cv::circle(img, phi2xy(y_k), 4, cv::Scalar(255, 255, 255), 2);  // predicted
        cv::circle(img, phi2xy(x_k), 4, cv::Scalar(0, 0, 255));  // actual to
        cv::imshow("Kalman", img);
        #else
        point_t observed = phi2xy(z_k);
        point_t predicted = phi2xy(y_k);
        point_t actual_to = phi2xy(z_k);
        drawcb(data, observed, predicted, actual_to);
/*
        printf("observed:  (%d, %d)\n", observed.x, observed.y);
        printf("predicted: (%d, %d)\n", predicted.x, predicted.y);
        printf("actual_to: (%d, %d)\n", actual_to.x, actual_to.y);
*/
        #endif

        // adjust Kalman filter state
        //
        kalman.correct(z_k);

        // Apply the transition matrix 'F' (e.g., step time forward)
        // and also apply the "process" noise w_k
        //
        randn(w_k, 0.0, sqrt(static_cast<double>(kalman.processNoiseCov.at<float>(0, 0))));

        x_k = kalman.transitionMatrix * x_k + w_k;

#ifdef PC_OPENCV
        // exit if user hits 'Esc'
        if ((cv::waitKey(100) & 255) == 27) {
            break;
        }
#endif
    }
}

