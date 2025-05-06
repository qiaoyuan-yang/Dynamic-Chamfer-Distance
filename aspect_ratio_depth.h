#ifndef ASPECT_RATIO_DEPTH_H
#define ASPECT_RATIO_DEPTH_H

#include <vector>
#include <cmath>

/**
 * 计算两点之间的欧几里得距离
 * @param p1 第一个点坐标向量
 * @param p2 第二个点坐标向量 
 * @return 两点之间的欧几里得距离
 */
double euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2);

/**
 * 使用2-近似算法计算数据集直径
 * @param P 数据集
 * @return 数据集的2-近似直径
 */
double compute_2approx_diameter(const std::vector<std::vector<double> >& P);

/**
 * 计算两个数据集之间的aspect ratio
 * @param A 第一个数据集
 * @param B 第二个数据集
 * @return 两个数据集的aspect ratio
 */ 
double calculate_aspect_ratio(const std::vector<std::vector<double> >& A,
                            const std::vector<std::vector<double> >& B, int max_dist);

/**
 * 计算quadtree所需深度
 * @param aspect_ratio 两个数据集的aspect ratio
 * @param d 数据维度
 * @return quadtree所需的最小深度
 */
int calculate_quadtree_depth(double aspect_ratio);

#endif // ASPECT_RATIO_DEPTH_H
