#include <stdlib.h>
#include <iostream>
#include <random>
#include <iomanip>
#include "opencv.hpp"

using namespace KCV;
using namespace std;

void printMatrix(const char *name, Mat &m)
{
    printf("Matrix: %s, dim:(%d, %d)\n", name, m.rows, m.cols);
    for (int i=0; i<m.rows; i++) {
        for (int j=0; j<m.cols; j++) {
            printf(" %15f", m.at<float>(i,j));
        }
        printf("\n");
    }
}
#define ABS(x) (((x)<0)?-(x):(x))
#define ERROR 0.001
int unit_test_inverse_matrix()
{
    constexpr int FLOAT_MIN = 0;
    constexpr int FLOAT_MAX = 100;

    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<> distr(FLOAT_MIN, FLOAT_MAX);

    for (int i=1; i<11; i++) {
        float *data = (float *)malloc(i*i*sizeof(float));
        for (int j=0; j<i*i; j++)
            data[j] = distr(eng);

        printf("calculating inverse (%d x %d)... ", i, i);
        Mat *m = new Mat(i, i, CV_32F, data);
        Mat &inv = m->inverse();
        Mat &res = *m * inv;
        // check result
        for (int j=0; j<res.rows; j++) {
            for (int k=0; k<res.cols; k++) {
                if (j==k) {
                    if (ABS(res.at<float>(j,k) - 1.0) > ERROR) {
                        printf("inverse error (1) %f, (%d, %d) of (%d, %d) -- 1.0\n",
                            ABS(res.at<float>(j,k)), j, k, i, i);
                        exit(1);
                    }
                } else {
                    if (ABS(res.at<float>(j,k)) > ERROR) {
                        printf("inverse error (0) %f, (%d, %d) of (%d, %d) -- 1.0\n",
                            ABS(res.at<float>(j,k)), j, k, i, i);
                        exit(1);
                    }
                }
            }
        }
        delete m;
        printf("Matrix inverse check OK\n");
        free(data);
    }

    return 0; // OK
}

#define DIM_MIN     1
#define DIM_MAX     100
#define NUM_ITER    100
int unit_test_transport_matrix()
{
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<> distr(DIM_MIN, DIM_MAX);
    for (int count=0; count<100; count++) {
        int row = distr(eng), col = distr(eng);
        printf("transpose #%d (%d, %d) check...", count, row, col);
        Mat m(row, col, CV_32F);
        // setup random value
        float *p = m.getData<float>();
        int c = row * col;
        while (c-->0)
            *p++ = distr(eng);

        Mat &t = m.transpose();
        if ((t.cols != m.rows) || (t.rows != m.cols)) {
            printf("transpose doesn't match (%d, %d) - (%d, %d)\n", m.rows, m.cols, t.rows, t.cols);
            return -1;
        }
        for (int i=0; i<row; i++) {
            for (int j=0; j<col; j++) {
                if (m.at<float>(i,j) != t.at<float>(j,i)) {
                    printf("    %f == %f -- FAIL\n", m.at<float>(i,j), t.at<float>(j,i));
                    return -1;
                }
            }
        }
        printf("OK\n");
    }

    return 0;
}

#define NUM_SAMPLES 100
int unit_test_karman_filter()
{
    float x[] = {0, 0};
    float p[] = {1, 0, 0, 1};
    float f[] = {1,1,0,1};
    float q[] = {0.0001, 0, 0, 0.0001};
    float h[] = {1,0};
    float r[] = {1};
    float z[NUM_SAMPLES];

    random_device rd;
    default_random_engine eng(rd());
    normal_distribution<float> normal(0,1);
    for (int i=0; i<NUM_SAMPLES; i++) {
        z[i] = i + normal(eng);
    }

    Mat X(2, 1, CV_32F, x);
    Mat P(2, 2, CV_32F, p);
    Mat F(2, 2, CV_32F, f);
    Mat Q(2, 2, CV_32F, q);
    Mat H(1, 2, CV_32F, h);
    Mat R(1, 1, CV_32F, r);

    try {
        Mat X_, P_, K;
        for (int i=0; i<NUM_SAMPLES; i++) {
            X_ = F * X;
            P_ = F * P * F.transpose() + Q;
            K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();
            Mat Z(1, 1, CV_32F, &z[i]);
            X = X_ + K * (Z - H * X_);
            P = (Mat::eye(2, 2, CV_32F) - K * H) * P_;
            printMatrix("X", X);
        }
    } catch (const Exception& e) {
        printf("error: %s\n", e.what());
    }
    return 0;
}