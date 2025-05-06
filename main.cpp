#define FALCONN_NO_FHT 1
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include <deque>
#include <fstream>
#include <type_traits>
#include "quad_tree.h"
#include "aspect_ratio_depth.h"
#include "BruteForceANN.h"
#include "data_preprocess.h"
#include "datatype.h"
#include "ANN.h"
#include "FALCONN/external/eigen/Eigen/Dense"
#include "FALCONN/src/include/falconn/lsh_nn_table.h"
#include "baseline/benchmark.h"


using namespace std;
using namespace falconn;
using Eigen::VectorXf;

using Point = VectorXf;
using PointSet = std::vector<Point>;


std::pair<int, float> find_nearest_neighbor(
    const Point& query,
    std::unique_ptr<falconn::LSHNearestNeighborQuery<Point>>& query_table,
    const PointSet& dataset
) {
    int nn_index = query_table->find_nearest_neighbor(query);
    if(nn_index == -1)
        // cout << "query: " << query << endl;
        // cout << "Nearest neighbor search failed: no neighbor found." << endl;
        return {-1,-1};
    const Point& nn_point = dataset[nn_index];
    float distance = (nn_point - query).norm();

    return {nn_index, distance};
}


double sample_points_LSH_ANN(QuadTree& tree, const vector<vector<double>>& A, std::unique_ptr<falconn::LSHNearestNeighborQuery<Point>>& query_table, const PointSet& dataset) {
    vector<vector<double>> sampled_points;
    int n = A.size();
    int sample_times = max(1, static_cast<int>(log2(n)));
    // cout << "A size: " << n << " B size: " << dataset.size() << endl;
    double sum = 0.0;
    random_device rd;
    mt19937 gen(rd());

    while (sampled_points.size() < sample_times) {
        // cout << "sampled_points.size(): " << sampled_points.size() << " sample_times: " << sample_times << endl;
        shared_ptr<HSTNode> node = tree.sample_by_weight();
        // if (node == nullptr) cout << "empty!" << endl;
        if (!node) {
            // cout << 111 << endl;
            continue;
        }
        double weight = node->tree_weight;
        double distance;
        int nn_index;

        while (node) {
            // cout << "node->A_count: " << node->A_count << " " << dataset.size() << " " << n << endl;
            // if (node->A_count == 1 && node->single_A_point.empty()) cout << "leaf empty" << endl;
            if (node->A_count == 1 && !node->single_A_point.empty()) {
                int dim = node->single_A_point.size();
                Point query(dim);
                for (int j = 0; j < dim; ++j) query[j] = node->single_A_point[j];
                auto result = find_nearest_neighbor(query, query_table, dataset);
                nn_index = result.first;
                distance = result.second;
                // cout << "index = -1" << endl;
                if (nn_index != -1) {
                    sum += distance * tree.total_weight / weight;
                    sampled_points.push_back(node->single_A_point);
                }
                break;
            }
            node = tree.sample_node_by_weight(node);
        }
    }

    return sum / sample_times;
}


template<typename T>
double sample_points_from_quadtree(QuadTree& tree, const vector<vector<double>>& A, T&& ann) {
    vector<vector<double>> sampled_points;
    int n = A.size();
    int sample_times = max(1, static_cast<int>(log2(n)));
    double sum = 0.0;
    random_device rd;
    mt19937 gen(rd());

    while (sampled_points.size() < sample_times) {
        shared_ptr<HSTNode> node = tree.sample_by_weight();
        if (!node) continue;

        while (node) {
            if (node->A_count == 1 && !node->single_A_point.empty()) {
                sampled_points.push_back(node->single_A_point);
                point sampled_point(node->single_A_point);
                
                // 使用if constexpr区分指针和非指针类型
                point neighbor = [&]() {
                    if constexpr (std::is_pointer_v<std::decay_t<T>>) {
                        return ann->query(sampled_point);
                    } else {
                        return ann.query(sampled_point);
                    }
                }();
                
                double dist = neighbor.distance(sampled_point);
                sum += dist * tree.total_weight / node->tree_weight;
                break;
            }
            node = tree.sample_node_by_weight(node);
        }
    }

    return sum / sample_times;
}


int main() {
    // Load datasets
    std::vector<std::vector<double>> A = read_csv("dataset/Text_Embeddings/federalist/A.csv");
    std::vector<std::vector<double>> B = read_csv("dataset/Text_Embeddings/federalist/B.csv");
    int window_size = (A.size() + B.size()) / 200; // 100
    int query_interval = (A.size() + B.size()) / 10; // 10
    // std::vector<std::vector<double>> A = read_csv("test_dataset/A.csv");
    // std::vector<std::vector<double>> B = read_csv("test_dataset/B.csv");
    // int window_size = (A.size() + B.size()) / 5; // 100
    // int query_interval = (A.size() + B.size()) / 5; // 10


    if (A.empty() || B.empty()) {
        std::cerr << "Error: Failed to read input files" << std::endl;
        return 1;
    }

    if (A[0].size() != B[0].size()) {
        throw std::runtime_error("Error: Dimension mismatch between A and B. "
                                 "A has dimension " + std::to_string(A[0].size()) +
                                 ", B has dimension " + std::to_string(B[0].size()) + ".");
    }

    std::vector<std::vector<double>> combined = A;
    combined.insert(combined.end(), B.begin(), B.end());

    double max_dist = compute_2approx_diameter(combined);
    double aspect_ratio = calculate_aspect_ratio(A, B, max_dist);
    int depth = calculate_quadtree_depth(aspect_ratio);

    // cout << "width: " << max_dist * 2 << endl;

    QuadTree tree(A[0].size(), max_dist * 2, depth);

    std::vector<std::pair<std::vector<double>, char>> stream;
    // for (auto& pt : A) stream.emplace_back(pt, 'A');
    // for (auto& pt : B) stream.emplace_back(pt, 'B');

    // std::shuffle(stream.begin(), stream.end(), std::default_random_engine(42));
    size_t n = min(A.size(), B.size());
    for (size_t i = 0; i < n; ++i) {
        stream.emplace_back(A[i], 'A');
        stream.emplace_back(B[i], 'B');
    }

    std::deque<std::pair<std::vector<double>, char>> window;
    std::deque<std::vector<double>> current_A, current_B;
    vector<vector<double>> first_A, first_B;
    for (int i = 0; i < window_size; ++i){
        window.push_back(stream[i]);
        if (stream[i].second == 'A') {
            first_A.push_back(stream[i].first);
            current_A.push_back(stream[i].first);
            // cout << "first_A: ";
            // for (auto j = 0; j < stream[i].first.size(); j++){
            //     cout << stream[i].first[j] << " ";
            // }
            // cout << endl;
        }
        else {
            first_B.push_back(stream[i].first);
            current_B.push_back(stream[i].first);
            // cout << "first_B: ";
            // for (auto j = 0; j < stream[i].first.size(); j++){
            //     cout << stream[i].first[j] << " ";
            // }
            // cout << endl;
        }
    }
    tree.build(A, B, first_A, first_B);
    // cout << "first_A.size: " << first_A.size() << " first_B.size: " << first_B.size() << " window size " << window.size() << endl;
    tree.Build_Tree_Sampler();

    std::ofstream cost_log("output/cost_log.csv");
    if (!cost_log.is_open()) {
        std::cerr << "Error: Failed to open cost_log.csv for writing." << std::endl;
        return 1;
    }
    cost_log << "step,ours,chamfer\n";
    // cout << 111 << endl;
    // int count = 0;
    // double total_weight = 0;
    // tree.weight_tree.printInOrder(tree.weight_tree.root, total_weight, count);

    for (size_t i = window_size - 1; i < stream.size(); ++i) {
        // cout << 444 << endl;
        // int count = 0;
        // double total_weight = 0;
        // tree.weight_tree.printInOrder(tree.weight_tree.root, total_weight, count);
        // cout << "insert: "  << stream[i].first << " " << stream[i].second << endl;
        tree.insert(stream[i].first, stream[i].second);
        // cout << 555 << endl;
        // count = 0;
        // total_weight = 0;
        // tree.weight_tree.printInOrder(tree.weight_tree.root, total_weight, count);
        window.push_back(stream[i]);
        if (stream[i].second == 'A') current_A.push_back(stream[i].first);
        else current_B.push_back(stream[i].first);

        if (window.size() > window_size) {
            // cout << 222 << endl;
            // int count = 0;
            // double total_weight = 0;
            // tree.weight_tree.printInOrder(tree.weight_tree.root, total_weight, count);
            // cout << "remove: "  << window.front().first << " " << window.front().second << endl;
            tree.remove(window.front().first, window.front().second);
            // cout << 333 << endl;
            // count = 0;
            // total_weight = 0;
            // tree.weight_tree.printInOrder(tree.weight_tree.root, total_weight, count);
            if (window.front().second == 'A') current_A.pop_front();
            else current_B.pop_front();
            window.pop_front();
        }

        if (i % query_interval == 0 && i > 0) {
            try {
                std::vector<std::vector<double>> A_now(current_A.begin(), current_A.end());
                std::vector<std::vector<double>> B_now(current_B.begin(), current_B.end());
                cout << "A_now.size: " << A_now.size() << " B_now.size: " << B_now.size();


                // BruteForceANN ann_brute;
                // for (const auto& coords : B_now) ann_brute.insert(point(coords));
                // double cost1 = sample_points_from_quadtree(tree, A_now, ann_brute);

                // ANN* ann_lsh = new ANN(B_now.size(), dataset_dmin, max_dist, dataset_rho);
                // std::vector<point> pointset = convert_to_points(B_now);
                // for (const point& p : pointset) ann_lsh->insert(p);
                // double cost2 = sample_points_from_quadtree(tree, A_now, ann_lsh);

                int dim = A_now[0].size();
                int num_probes = 64;
                PointSet dataset;
                for (const auto& row : B_now) {
                    Point p(row.size());
                    for (int j = 0; j < row.size(); ++j) p[j] = static_cast<float>(row[j]);
                    dataset.push_back(p);
                }
                int num_points = dataset.size();
                LSHConstructionParameters params = get_default_parameters<Point>(num_points, dim, DistanceFunction::EuclideanSquared, true);
                params.lsh_family = LSHFamily::Hyperplane;
                params.distance_function = DistanceFunction::EuclideanSquared;
                params.k = 10;
                params.l = 50;
                params.storage_hash_table = StorageHashTable::LinearProbingHashTable;
                params.num_setup_threads = 0;
                params.seed = 123;
                auto table = construct_table<Point>(dataset, params);
                auto query_table = table->construct_query_object();
                query_table->set_num_probes(num_probes);
                double cost3 = sample_points_LSH_ANN(tree, A_now, query_table, dataset);

                double chamfer = chamfer_distance(A_now, B_now);

                std::cout << "Step " << i << " Results:\n";
                // std::cout << "The cost for BruteForceANN is: " << cost1 << std::endl;
                // std::cout << "The cost for base_LSH is: " << cost2 << std::endl;
                std::cout << "Our cost is: " << cost3 << std::endl;
                std::cout << "Chamfer Distance: " << chamfer << std::endl;

                cost_log << i << "," << cost3 << "," << chamfer << "\n";
                // delete ann_lsh;
            } catch (const std::exception& e) {
                std::cerr << "Error at step " << i << ": " << e.what() << std::endl;
            }
        }
    }

    cost_log.close();
    return 0;
}
