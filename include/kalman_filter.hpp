#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_

#include "tiny_opencv.hpp"

namespace KCV {

class KalmanFilter {
    public:
        // attributes
        Mat controlMatrix;
        Mat errorCovPost; // (P(k)): P(k)=(I-K(k)*H)*P'(k)
        Mat errorCovPre;  // (P'(k)): P'(k)=A*P(k-1)*At + Q0lllllllllllll
        Mat gain;         // (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
        Mat measurementMatrix;
        Mat measurementNoiseCov;
        Mat processNoiseCov;
        Mat statePost; // corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) 
        Mat statePre;  // predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
        Mat transitionMatrix; // state transition matrix

        Mat temp1;
        Mat temp2;
        Mat temp3;
        Mat temp4;
        Mat temp5;
        
    public:
        KalmanFilter();
        KalmanFilter(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);

    public:
        Mat &correct(const Mat &measurement);
        void init(int dynamParams, int measureParams, int controlParams=0, int type=CV_32F);
        Mat &predict(const Mat &control=Mat());
};

} // end of namespace

#endif
