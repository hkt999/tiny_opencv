#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include "unit_test.hpp"


#include <math.h>
#include <tiny_opencv.hpp>
#include <kalman_filter.hpp>
#define IMG_W   512
#define IMG_H   512

using namespace kcv;

inline point_t phi2xy(Mat &mat)
{
    point_t p;
    p.x = round(IMG_W / 2 + IMG_W / 3 * cos(mat.at<float>(0)));
    p.y = round(IMG_H / 2 - IMG_W / 3 * sin(mat.at<float>(0)));
    return p;
}

void unit_test_kalman_filter_mouse(void *data, draw_frame_t drawcb, get_observation_t observcb)
{
	// >>> Kalman Filter Initialization
	int stateSize = 4;  // [x, y, v_x, v_y]
	int measSize = 2;   // [z_x, z_y] // we will only measure mouse cursor x and y
	int contrSize = 0;  // no control input

	unsigned int F_type = CV_32F;

	// initiation of OpenCV Kalman Filter
	KalmanFilter KF(stateSize, measSize, contrSize, F_type);

	// creating state vector
	Mat state(stateSize, 1, F_type);  // [x, y, v_x, v_y] // column Matrix

										  // creating measurement vector
	Mat meas(measSize, 1, F_type);    // [z_x, z_y] // column matrix

										  // Transition state matrix A
										  // Note: set dT at each processing step!
										  // X_k = A*X_k-1
										  // X_k = current state := x_k, y_k, v_x_k
										  // X_k-1 = previous state
										  // A =
										  // [1 0 dT 0]
										  // [0 1 0 dT]
										  // [0 0 1  0]
										  // [0 0 0  1]
										  // observe it is an identity matrix with dT inputs that we will provide later

    state.print("state");
    meas.print("meas");
	setIdentity(KF.transitionMatrix);
    KF.transitionMatrix.print("KF.transitionMatrix");

	// Measurement Matrix (This is C or H matrix)
	// size of C is measSize x stateSize
	// only those values will set which we can get as measurement in a state vector
	// here out of [x, y, v_x and v_y] we can only measure x, y of the mouse cursor coordianates
	// so we set only element "0" and "5".
	// [1 0 0 0]
	// [0 1 0 0]

	// Process Noise Covariance Matrix := stateSize x stateSize
	//  [Ex 0  0    0]
	//  [0 Ey  0    0]
	//  [0 0 E_v_x  0]
	//  [0 0  0  E_v_y]

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

    KF.measurementMatrix.print("KF.measurementMatrix");
    KF.processNoiseCov.print("KF.processNoiseCov");
    KF.measurementNoiseCov.print("KF.measurementNoiseCov");
    KF.errorCovPost.print("KF.errorCovPost");

	// Measure Noise Covariance Matrix
	//cv::setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));

	// <<< Kalman Filter initializationOnThread

    for(;;) {
        state = KF.predict(); // First predict, to update the internal statePre variable

		//Point predictPt(state.at<float>(0), state.at<float>(1));
		// <<< Kalman Prediction
        point_t predicted;
        predicted.x = (int)state.at<float>(0);
        predicted.y = (int)state.at<float>(1);

		// >>> Get Mouse Point
        point_t observed = observcb(data);

		// >>> Passing the measured values to the measurement vector
		meas.at<float>(0) = observed.x;
		meas.at<float>(1) = observed.y;

		// >>> Kalman Update Phase
		Mat estimated = KF.correct(meas);

        point_t actual_to;
        actual_to.x = estimated.at<float>(0);
        actual_to.y = estimated.at<float>(1);

        drawcb(data, observed, predicted, actual_to);
	}
}

void unit_test_kalman_filter_angle(void *data, draw_frame_t drawcb)
{
    // Initialize, create Kalman filter object, window, random number generator etc.
    KalmanFilter kalman(2, 1, 0);

    // Initialize with random guess.
    Mat x_k(2, 1, CV_32F);
    randn(x_k, 0.0, 0.1);

    // process noise
    Mat w_k(2, 1, CV_32F);

    // measurements, only one parameter for angle
    Mat z_k = Mat::zeros(1, 1, CV_32F);

    // Transition matrix 'F' describes relationship between model parameters at step k and at step k+1 (this is
    // the "dynamics" in our model.
    float F[] = {1, 1, 0, 1};
    kalman.transitionMatrix = Mat(2, 2, CV_32F, F).clone();

    // Initialize other Kalman filter parameters.
    setIdentity(kalman.measurementMatrix, Scalar(1));
    setIdentity(kalman.processNoiseCov, Scalar(1e-5));
    setIdentity(kalman.measurementNoiseCov, Scalar(1e-1));
    setIdentity(kalman.errorCovPost, Scalar(1));

    // choose random initial state
    randn(kalman.statePost, 0.0, 0.1);

    for (;;) {
        // predict point position
        Mat y_k = kalman.predict();
        y_k.print("y_k (prediction)");

        // generate measurement (z_k)
        randn(z_k, 0.0, sqrt(static_cast<double>(kalman.measurementNoiseCov.at<float>(0, 0))));
        z_k = kalman.measurementMatrix * x_k + z_k;
        z_k.print("z_k (noise)");
        kalman.measurementMatrix.print("measurementMatrix");
        x_k.print("x_k");
        z_k.print("observed");

        // plot points (e.g., convert)
        point_t observed = phi2xy(z_k);
        point_t predicted = phi2xy(y_k);
        point_t actual_to = phi2xy(x_k);
        drawcb(data, observed, predicted, actual_to);

        // adjust Kalman filter state
        kalman.correct(z_k);

        // Apply the transition matrix 'F' (e.g., step time forward) and also apply the "process" noise w_k
        randn(w_k, 0.0, sqrt(static_cast<double>(kalman.processNoiseCov.at<float>(0, 0))));

        x_k = kalman.transitionMatrix * x_k + w_k;
    }
}

