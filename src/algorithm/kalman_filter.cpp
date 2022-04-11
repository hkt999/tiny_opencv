#include "tiny_opencv.hpp"
#include "kalman_filter.hpp"
#include <stdio.h>
#include <assert.h>

namespace KCV {

KalmanFilter::KalmanFilter()
{
}

KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams, int type)
{
    init(dynamParams, measureParams, controlParams, type);
}

void KalmanFilter::init(int DP, int MP, int CP, int type)
{
    assert( DP > 0 && MP > 0 );
    assert( type == CV_32F || type == CV_64F );
    //CP = max(CP, 0);
    if (CP<0)
        CP = 0;

    statePre = Mat::zeros(DP, 1, type);
    statePost = Mat::zeros(DP, 1, type);
    transitionMatrix = Mat::eye(DP, DP, type);

    processNoiseCov = Mat::eye(DP, DP, type);
    measurementMatrix = Mat::zeros(MP, DP, type);
    measurementNoiseCov = Mat::eye(MP, MP, type);

    errorCovPre = Mat::zeros(DP, DP, type);
    errorCovPost = Mat::zeros(DP, DP, type);
    gain = Mat::zeros(DP, MP, type);

    if( CP > 0 )
        controlMatrix = Mat::zeros(DP, CP, type);
    else
        controlMatrix.release();

    temp1.create(DP, DP, type);
    temp2.create(MP, DP, type);
    temp3.create(MP, MP, type);
    temp4.create(MP, DP, type);
    temp5.create(MP, 1, type);
}

Mat &KalmanFilter::correct(const Mat &measurement)
{
    // temp2 = H*P'(k)
    temp2 = measurementMatrix * errorCovPre;

    // temp3 = temp2*Ht + R

    // gemm(src1, src2, alpha, src3, beta, output, flags )
    // gemm --> dst = alpha * src1.transpose * src2 + beta * src3.transpose
    temp3 = temp2 * measurementMatrix.t() + measurementNoiseCov;

    // temp4 = inv(temp3)*temp2 = Kt(k)
    temp4 = temp3.inverse() * temp2;

    // K(k)
    gain = temp4.t();

    // temp5 = z(k) - H*x'(k)
    temp5 = measurement - measurementMatrix*statePre;

    // x(k) = x'(k) + K(k)*temp5
    statePost = statePre + gain * temp5;

    // P(k) = P'(k) - K(k)*temp2
    errorCovPost = errorCovPre - gain * temp2;

    return statePost;
}

Mat &KalmanFilter::predict(const Mat &control)
{
    // update the state: x'(k) = A*x(k)
    statePre = transitionMatrix * statePost;

/*
    if (!control.empty()) {
        // x'(k) = x'(k) + B*u(k)
        statePre += controlMatrix * control;
    }
*/
    if (control.cols * control.rows > 0) {
        statePre += controlMatrix * control;
    }

    // update error covariance matrices: temp1 = A*P(k)
    temp1 = transitionMatrix * errorCovPost;

    // P'(k) = temp1*At + Q
    // gemm(src1, src2, alpha, src3, beta, output, flags )
    // gemm --> dst = alpha * src1.transpose * src2 + beta * src3.transpose
    // gemm(temp1, transitionMatrix, 1, processNoiseCov, 1, errorCovPre, GEMM_2_T);
    errorCovPre = temp1 * transitionMatrix.t() + processNoiseCov;

    // handle the case when there will be measurement before the next predict.
    statePre.copyTo(statePost);
    errorCovPre.copyTo(errorCovPost);

    return statePre;
}

} // end of namespace