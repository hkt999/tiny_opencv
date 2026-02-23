#include "hungarian.hpp"

using namespace KCV;

int main() {
    // Cost matrix (rows = workers, cols = tasks)
    Mat cost(3, 3, CV_32FC1);
    cost.at<float>(0, 0) = 4;  cost.at<float>(0, 1) = 1;  cost.at<float>(0, 2) = 3;
    cost.at<float>(1, 0) = 2;  cost.at<float>(1, 1) = 0;  cost.at<float>(1, 2) = 5;
    cost.at<float>(2, 0) = 3;  cost.at<float>(2, 1) = 2;  cost.at<float>(2, 2) = 2;

    Mat assignment(3, 1, CV_32FC1); // assignment.at<float>(row,0) = chosen column
    HungarianAlgorithm solver;
    double total_cost = solver.Solve(cost, assignment);
    (void)total_cost;

    for (int i = 0; i < cost.rows; i++) {
        int col = static_cast<int>(assignment.at<float>(i, 0));
        (void)col;
    }

    return 0;
}
