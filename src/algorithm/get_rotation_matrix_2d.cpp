#include "tiny_opencv.hpp"
#include <math.h>

using namespace KCV;

Mat getRotationMatrix2D(Point2f center, double angle, double scale)
{

    angle *= M_PI/180;
    double alpha = cos(angle)*scale;
    double beta = sin(angle)*scale;

	Mat M(2, 3, CV_32FC1);

	float *fdata = (float *)M.data;
	*fdata++ = alpha;
	*fdata++ = beta;
	*fdata++ = (1-alpha) * center.x - beta * center.y;
	*fdata++ = -beta;
	*fdata++ = alpha;
	*fdata++ = beta * center.x + (1-alpha) * center.y;


    return M;
}
