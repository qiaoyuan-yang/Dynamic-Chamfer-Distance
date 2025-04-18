#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>

class HSTNode {
public:
    int depth;
    std::vector<float> origin;  // 当前grid左下角坐标
    float width;                // 当前grid宽度
    std::unordered_map<std::vector<int>, std::unique_ptr<HSTNode>> children;
    
    std::vector<std::vector<float>> points_A;  // 叶节点包含的A集的点
    std::vector<std::vector<float>> points_B;  // 叶节点包含的B集的点

    int A_count = 0;  // Number of points from dataset A
    int B_count = 0;  // Number of points from dataset B
    int match_count = 0;  // Number of points from A matched in this cell with any point from B

    HSTNode(int d, const std::vector<float>& o, float w)
        : depth(d), origin(o), width(w) {}
};

// 将点映射到当前格子的子格子编号
std::vector<int> point_to_grid_index(const std::vector<float>& point, const std::vector<float>& origin, float width) {
    std::vector<int> index(point.size());
    for (size_t i = 0; i < point.size(); ++i) {
        index[i] = static_cast<int>((point[i] - origin[i]) / (width / 2));
    }
    return index;
}

// 检查A中的点是否与B中的任何点匹配
bool is_matched(const std::vector<float>& point_a, const std::vector<std::vector<float>>& dataset_b) {
    for (const auto& point_b : dataset_b) {
        bool match = true;
        for (size_t i = 0; i < point_a.size(); ++i) {
            if (point_a[i] != point_b[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

// 递归构建 HST 树
std::unique_ptr<HSTNode> build_2hst(const std::vector<std::vector<float>>& points_A, const std::vector<std::vector<float>>& points_B,
                                    float delta, float random_shift, int max_depth, int depth = 0, const std::vector<float>& origin = {}) {
    size_t d = points_A[0].size();
    std::vector<float> current_origin = origin;
    
    if (origin.empty()) {
        // 根节点左下角坐标
        std::vector<float> min_corner(d, std::numeric_limits<float>::max());
        std::vector<float> max_corner(d, std::numeric_limits<float>::lowest());

        // 计算所有点的最小最大坐标
        for (const auto& pt : points_A) {
            for (size_t i = 0; i < d; ++i) {
                min_corner[i] = std::min(min_corner[i], pt[i]);
                max_corner[i] = std::max(max_corner[i], pt[i]);
            }
        }

        // 计算dataset B的边界
        for (const auto& pt : points_B) {
            for (size_t i = 0; i < d; ++i) {
                min_corner[i] = std::min(min_corner[i], pt[i]);
                max_corner[i] = std::max(max_corner[i], pt[i]);
            }
        }

        // 随机偏移
        for (size_t i = 0; i < d; ++i) {
            current_origin[i] = min_corner[i] - random_shift;
        }
    }

    float width = delta / std::pow(2, depth);
    auto node = std::make_unique<HSTNode>(depth, current_origin, width);

    if (depth >= max_depth || (points_A.size() == 1 && points_B.size() == 1)) {
        node->points_A = points_A;
        node->points_B = points_B;
        node->A_count = points_A.size();
        node->B_count = points_B.size();
        node->match_count = 0;
        for (const auto& pt_A : points_A) {
            if (is_matched(pt_A, points_B)) {
                node->match_count++;
            }
        }
        return node;
    }

    // 将点划分到 2^d 个子格子
    std::unordered_map<std::vector<int>, std::vector<std::vector<float>>> buckets_A;
    for (const auto& pt : points_A) {
        std::vector<int> idx = point_to_grid_index(pt, current_origin, width);
        buckets_A[idx].push_back(pt);
    }

    std::unordered_map<std::vector<int>, std::vector<std::vector<float>>> buckets_B;
    for (const auto& pt : points_B) {
        std::vector<int> idx = point_to_grid_index(pt, current_origin, width);
        buckets_B[idx].push_back(pt);
    }

    // 合并A和B的桶
    for (auto& bucket_A : buckets_A) {
        node->A_count += bucket_A.second.size();
    }
    for (auto& bucket_B : buckets_B) {
        node->B_count += bucket_B.second.size();
    }

    // 递归构建每个子节点
    for (const auto& bucket_A : buckets_A) {
        std::vector<float> child_origin = current_origin;
        for (size_t i = 0; i < d; ++i) {
            // 按照当前格子的宽度，将子格子的origin进行二分划分
            child_origin[i] += (bucket_A.first[i] == 0) ? 0 : width / 2;
        }

        // Check if any point from A in this bucket matches a point in B
        std::vector<std::vector<float>> matched_points_A;
        for (const auto& pt_A : bucket_A.second) {
            if (is_matched(pt_A, points_B)) {
                matched_points_A.push_back(pt_A);
            }
        }

        node->match_count += matched_points_A.size();

        // Recursively build the subtree
        node->children[bucket_A.first] = build_2hst(bucket_A.second, points_B, delta, random_shift, max_depth, depth + 1, child_origin);
    }

    return node;
}

// 获取点从根到叶子的路径
std::vector<std::pair<int, std::vector<int>>> get_path_to_leaf(HSTNode* node, const std::vector<float>& point) {
    std::vector<std::pair<int, std::vector<int>>> path;
    while (!node->children.empty()) {
        std::vector<int> idx = point_to_grid_index(point, node->origin, node->width);
        path.push_back({node->depth, idx});
        node = node->children[idx].get();
    }
    return path;
}

int main() {
    // 示例数据集，二维点
    std::vector<std::vector<float>> points_A = {
        {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0},
        {7.0, 8.0}, {9.0, 10.0}, {2.0, 3.0}, {4.0, 5.0}
    };
    std::vector<std::vector<float>> points_B = {
        {1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5},
        {7.5, 8.5}, {9.5, 10.5}, {2.5, 3.5}, {4.5, 5.5}
    };

    float delta = 10.0f;
    float random_shift = 0.5f;
    int max_depth = 3;  // 最大深度

    // 构建树
    auto root = build_2hst(points_A, points_B, delta, random_shift, max_depth);

    // 获取某个点的路径
    std::vector<float> query_point = {3.0, 4.0};
    auto path = get_path_to_leaf(root.get(), query_point);

    std::cout << "Path to leaf for point (" << query_point[0] << ", " << query_point[1] << "):\n";
    for (const auto& step : path) {
        std::cout << "Depth: " << step.first << " - Grid Index: ";
        for (const auto& idx : step.second) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
