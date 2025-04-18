#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>

class HSTNode {
public:
    int depth;
    std::vector<float> origin;  // The coordinates of the lower left corner of the current grid
    float width;                // width of the grid
    std::unordered_map<std::vector<int>, std::unique_ptr<HSTNode>> children;
    
    int A_count = 0;  // Number of points from dataset A
    int B_count = 0;  // Number of points from dataset B
    int match_count = 0;  // Number of points from A matched in this cell with any point from B

    HSTNode(int d, const std::vector<float>& o, float w)
        : depth(d), origin(o), width(w) {}
};

// Map the point to the subgrid ID of the current grid
std::vector<int> point_to_grid_index(const std::vector<float>& point, const std::vector<float>& origin, float width) {
    std::vector<int> index(point.size());
    for (size_t i = 0; i < point.size(); ++i) {
        index[i] = static_cast<int>((point[i] - origin[i]) / (width / 2));
    }
    return index;
}

// Check if a point in A matches any point in B
bool is_matched(const std::vector<float>& point_a, const std::vector<std::vector<float>>& dataset_b,
                const std::vector<float>& origin, float width) {
    std::vector<int> index_a = point_to_grid_index(point_a, origin, width);
    // Check the points in B. If a B point is found that belongs to the same subgrid, it does not matched in the current cell.
    for (const auto& point_b : dataset_b) {
        std::vector<int> index_b = point_to_grid_index(point_b, origin, width);
        if (index_a == index_b) {
            return false;
        }
    }
    return true;
}

// Recursively build the HST tree
std::unique_ptr<HSTNode> build_2hst(const std::vector<std::vector<float>>& points_A, const std::vector<std::vector<float>>& points_B,
                                    float delta, float random_shift, int max_depth, int depth = 0, const std::vector<float>& origin = {}) {
    size_t d = points_A[0].size();
    std::vector<float> current_origin = origin;
    
    if (origin.empty()) {
        // Coordinates of the lower left corner of the root node
        std::vector<float> min_corner(d, std::numeric_limits<float>::max());

        for (const auto& pt : points_A) {
            for (size_t i = 0; i < d; ++i) {
                min_corner[i] = std::min(min_corner[i], pt[i]);
            }
        }

        for (const auto& pt : points_B) {
            for (size_t i = 0; i < d; ++i) {
                min_corner[i] = std::min(min_corner[i], pt[i]);
            }
        }

        // random shift
        for (size_t i = 0; i < d; ++i) {
            current_origin[i] = min_corner[i] - random_shift;
        }
    }

    float width = delta / std::pow(2, depth);
    auto node = std::make_unique<HSTNode>(depth, current_origin, width);

    if (depth >= max_depth || (points_A.size() == 1)) {
        node->A_count = points_A.size();
        node->B_count = points_B.size();
        node->match_count = 0;
        for (const auto& pt_A : points_A) {
            if (is_matched(pt_A, points_B, current_origin, width)) {
                node->match_count++;
            }
        }
        return node;
    }

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

    for (auto& bucket_A : buckets_A) {
        node->A_count += bucket_A.second.size();
    }
    for (auto& bucket_B : buckets_B) {
        node->B_count += bucket_B.second.size();
    }

    for (const auto& bucket_A : buckets_A) {
        std::vector<float> child_origin = current_origin;
        for (size_t i = 0; i < d; ++i) {
            child_origin[i] += (bucket_A.first[i] == 0) ? 0 : width / 2;
        }

        // Check if any point from A in this bucket matches a point in B
        std::vector<std::vector<float>> matched_points_A;
        for (const auto& pt_A : bucket_A.second) {
            if (is_matched(pt_A, points_B, current_origin, width)) {
                matched_points_A.push_back(pt_A);
            }
        }

        node->match_count += matched_points_A.size();

        // Recursively build the subtree
        node->children[bucket_A.first] = build_2hst(bucket_A.second, points_B, delta, random_shift, max_depth, depth + 1, child_origin);
    }

    return node;
}

// Get the path from the root to the leaf
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
    int max_depth = 3;

    auto root = build_2hst(points_A, points_B, delta, random_shift, max_depth);

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
