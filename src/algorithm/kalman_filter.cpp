#include "opencv.hpp"
#include "kalman_filter.hpp"

namespace KCV {

KalmanFilter::KalmanFilter()
{

}

KalmanFilter::KalmanFilter(int dynamParams, int measureParams, int controlParams, int type)
{
    init(dynamParams, measureParams, controlParams, type);
}

#if 0
    Mat controlMatrix;
    Mat errorCovPost; // (P(k)): P(k)=(I-K(k)*H)*P'(k)
    Mat errorCovPre;  // (P'(k)): P'(k)=A*P(k-1)*At + Q0lllllllllllll
    Mat gain;         // (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
    Mat measurementMatrix;
    Mat measurementNoiseConv;
    Mat processNoiseCov;
    Mat statePost; // corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k)) 
    Mat statePre;  // predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    Mat transitionMatrix; // state transition matrix
#endif

void KalmanFilter::init(int dynamParams, int measureParams, int controlParams, int type)
{
    // 2, 1, 0, CF_32F
}

Mat &KalmanFilter::correct(const Mat &measurement)
{
    return statePre; // TODO: fix it
}

Mat &KalmanFilter::predict(const Mat &control)
{
    return statePre; // TODO: fix it
}

} // end of namespace