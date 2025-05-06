#include "aspect_ratio_depth.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

using namespace std;

// 计算两点之间的欧几里得距离
double euclidean_distance(const vector<double>& p1, const vector<double>& p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// 使用2-近似算法计算数据集直径
double compute_2approx_diameter(const vector<vector<double> >& P) {
    if (P.empty()) {
        return 0.0;
    }
    
    const vector<double>& u = P[0];
    double max_dist = 0.0;
    
    for (const auto& point : P) {
        double dist = euclidean_distance(u, point);
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    return max_dist * 2; // 2-近似值
}

// 计算两个数据集之间的aspect ratio
double calculate_aspect_ratio(const vector<vector<double> >& A, const vector<vector<double> >& B, int max_dist) {
    double min_dist = 1e-4;
    return max_dist / min_dist;
}

// 计算quadtree所需深度
int calculate_quadtree_depth(double aspect_ratio) {
    return ceil(log2(aspect_ratio));
}

// int main() {
//     // 测试数据 - 2维空间
//     int d = 2;
    
//     // 数据集A - 3个点
//     vector<vector<double> > A = {
//         {1.0f, 2.0f},
//         {3.0f, 4.0f},
//         {5.0f, 6.0f}
//     };
    
//     // 数据集B - 2个点
//     vector<vector<double> > B = {
//         {7.0f, 8.0f},
//         {9.0f, 10.0f}
//     };
    
//     double aspect_ratio = calculate_aspect_ratio(A, B);
//     int depth = calculate_quadtree_depth(aspect_ratio, d);
    
//     cout << "Aspect ratio: " << aspect_ratio << endl;
//     cout << "Required quadtree depth: " << depth << endl;
    
//     return 0;
// }
