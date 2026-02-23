#include <stdlib.h>
#include <iostream>
#include <random>
#include <iomanip>
#include "tiny_opencv.hpp"

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
    std::mt19937 eng(1337);
    std::uniform_real_distribution<float> distr(-1.0f, 1.0f);

    for (int i=1; i<11; i++) {
        printf("calculating inverse (%d x %d)... ", i, i);
        Mat m(i, i, CV_32F);
        for (int r = 0; r < i; r++) {
            for (int c = 0; c < i; c++) {
                m.at<float>(r, c) = distr(eng);
            }
        }
        // Keep the matrix well-conditioned for stable numeric checks.
        for (int d = 0; d < i; d++) {
            m.at<float>(d, d) += (float)(i * 4);
        }
        Mat inv = m.inverse();
        Mat res = m * inv;
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
        printf("UnitTest Inverse: OK\n");
    }

    return 0; // OK
}

#define DIM_MIN     1
#define DIM_MAX     100
#define NUM_ITER    100
int unit_test_transpose_matrix()
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

        Mat t = m.transpose();
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
        printf("UnitTest Transpose: OK\n");
    }

    return 0;
}
