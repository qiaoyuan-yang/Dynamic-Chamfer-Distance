#include "benchmark.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

using std::vector;
using std::min;

double euclidean_squared(const vector<double>& p1, const vector<double>& p2) {
    if (p1.size() != p2.size()) {
        throw std::invalid_argument("Points have different dimensions");
    }
    double dist = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

double chamfer_distance(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("Input point sets must not be empty");
    }

    size_t dim = A[0].size();
    for (const auto& a : A) {
        if (a.size() != dim) throw std::invalid_argument("Inconsistent dimension in A");
    }
    for (const auto& b : B) {
        if (b.size() != dim) throw std::invalid_argument("Inconsistent dimension in B");
    }

    double sum_A_to_B = 0.0;
    for (const auto& a : A) {
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& b : B) {
            min_dist = min(min_dist, euclidean_squared(a, b));
        }
        sum_A_to_B += min_dist;
    }

    // double sum_B_to_A = 0.0;
    // for (const auto& b : B) {
    //     double min_dist = std::numeric_limits<double>::max();
    //     for (const auto& a : A) {
    //         min_dist = min(min_dist, euclidean_squared(b, a));
    //     }
    //     sum_B_to_A += min_dist;
    // }

    // return (sum_A_to_B) + (sum_B_to_A);
    return (sum_A_to_B);
}
