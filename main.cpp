#define FALCONN_NO_FHT 1
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <algorithm>
#include <memory>
#include <deque>
#include <fstream>
#include <type_traits>
#include "quad_tree.h"
#include "aspect_ratio_depth.h"
#include "BruteForceNN.h"
#include "data_preprocess.h"
#include "datatype.h"
#include "ANN-Oracle.h"
#include "baseline/benchmark.h"

using namespace std;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;


double sample_point_uniformly(
    const std::vector<std::vector<double>>& A,
    ANNOracle* ann_oracle,
    int sample_times,
    int n,
    BruteForceNN& ann  // 注意：改成引用，避免拷贝
) {
    if (sample_times > static_cast<int>(A.size())) {
        throw std::invalid_argument("sample_times is larger than dataset size");
    }

    std::uniform_int_distribution<size_t> dist(0, A.size() - 1);

    double sum = 0.0;
    for (int i = 0; i < sample_times; ++i) {
        static thread_local std::mt19937 gen(std::random_device{}());
        size_t idx = dist(gen);
        const auto& sample_pt = A[idx];

        double d = ann.query(point(sample_pt));
        sum += d;
    }

    return sum * static_cast<double>(n) / sample_times;
}


double sample_points_from_quadtree(QuadTree& tree, const vector<vector<double>>& A, ANNOracle* ann_oracle, 
    int& sample_times, BruteForceNN ann) {
    cout << "our sample times: " << sample_times << endl;

    vector<vector<double>> sampled_points;
    int times = sample_times;
    set<vector<double>> set_sampler;
    double sum = 0.0;

    while (sampled_points.size() < sample_times) {
        shared_ptr<HSTNode> node = tree.sample_by_weight();
        if (!node) continue;
        double width = node->width;
        // if (weight < 1e-4) continue;
        // cout << " first sampled cell match count: " << node->match_count << " width: " << node->width << 
        //     " tree_weight: " << node->tree_weight << " total weight: " << tree.total_weight << endl;

        while (node) {
            if (node->A_count == 1 && !node->single_A_point.empty()) {
                sampled_points.push_back(node->single_A_point);
                set_sampler.insert(node->single_A_point);
                Point sampled_point(node->single_A_point);

                // cout << "sampled_point: " << sampled_point << endl;
                
                // double dist = ann_oracle->distance(sampled_point);
                double dist = ann.query(point(node->single_A_point));

                sum += dist * tree.total_weight / width ;
                // cout << "our dist: " << dist << " tree.total_weight: " << tree.total_weight << " width: " << width << endl;
                break;
            }
            node = tree.sample_node_by_weight(node);
        }
    }
    sample_times = set_sampler.size();
    cout << "actually sampled point size: " << set_sampler.size() << endl;

    return sum / times;
}

double compute_avl_total_weight(const std::shared_ptr<BBTNode>& node) {
    if (!node) return 0.0;

    double left_sum = compute_avl_total_weight(node->left);
    double right_sum = compute_avl_total_weight(node->right);
    return left_sum + right_sum + node->weight;
}



int main() {
    // Load datasets
    std::vector<std::vector<double>> B = read_csv("dataset/Text_Embeddings/federalist/A_raw.csv");
    std::vector<std::vector<double>> A = read_csv("dataset/Text_Embeddings/federalist/B_raw.csv");
    // A 小
    int window_size = A.size();
    int query_interval = std::max(1, static_cast<int>((B.size() - A.size()) / 50)); // 10
    // std::vector<std::vector<double>> A = read_csv("test_dataset/A.csv");
    // std::vector<std::vector<double>> B = read_csv("test_dataset/B.csv");
    // int window_size = A.size();
    // int query_interval = std::max(1, static_cast<int>((B.size() - A.size()) / 50)); // 10

    cout << "window_size: " << window_size << " query_interval: " << query_interval << endl;


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
    cout << "diameter: " << max_dist << endl;
    double aspect_ratio = calculate_aspect_ratio(A, B, max_dist);
    int depth = calculate_quadtree_depth(aspect_ratio);

    // cout << "width: " << max_dist * 2 << endl;
    cout << "aspect_ratio: " << aspect_ratio << endl;
    cout << "max depth: " << depth << endl;

    std::vector<std::pair<std::vector<double>, char>> stream;
    for (size_t i = 0; i < B.size(); ++i) {
        stream.emplace_back(B[i], 'B');
    }

    ANNOracle* ann_oracle = new ANNOracle();

    std::deque<std::pair<std::vector<double>, char>> window;
    std::deque<std::vector<double>> current_B;
    for (int i = 0; i < window_size; ++i){
        window.push_back(stream[i]);
        if (stream[i].second == 'B') {
            current_B.push_back(stream[i].first);
            ann_oracle->insert(stream[i].first);
        }
    }
    std::vector<vector<double>> first_B(current_B.begin(), current_B.end());
    // cout << "first_B: " << first_B.size() << endl;

    Duration our_time(0), uniform_time(0), benchmark_time(0);

    auto start_ours1 = Clock::now();
    QuadTree tree(DIM, max_dist * 2, depth);
    vector<vector<double>> A_test = {};
    vector<vector<double>> B_test = {};
    tree.build(A, B, A_test, B_test);
    tree.Build_Tree_Sampler();
    for (auto a : A) {
        tree.insert(a, 'A');
    }
    for (auto b : first_B) {
        tree.insert(b, 'B');
    }
    auto end_ours1 = Clock::now();
    our_time += Duration(end_ours1 - start_ours1);

    auto start_benchmark1 = Clock::now();
    DynamicChamfer dc(A, first_B);
    auto end_benchmark1 = Clock::now();
    benchmark_time += Duration(end_benchmark1 - start_benchmark1);


    std::ofstream cost_log("output/cost_log.csv");
    if (!cost_log.is_open()) {
        std::cerr << "Error: Failed to open cost_log.csv for writing." << std::endl;
        return 1;
    }
    cost_log << "step,ours,chamfer,our_err,uni_err\n";



    //// 测试
    BruteForceNN ann;
    for (auto node : B){
        ann.insert(point(node));
    }
    int sample_time = static_cast<int>(log2(A.size()) * 10);
    double our_cost1 = sample_points_from_quadtree(tree, A, ann_oracle, sample_time, ann);

    double uniform_cost1 = sample_point_uniformly(A, ann_oracle, sample_time, A.size(), ann);
    
    double chamfer1 = dc.current();
    double our_err1 = (our_cost1 - chamfer1) / chamfer1;
    double uni_err1 = (uniform_cost1 - chamfer1) / chamfer1;
    std::cout << "Our cost is11: " << our_cost1 << std::endl;
    std::cout << "Chamfer Distance11: " << chamfer1 << std::endl;
    cout << "our_err11: " << our_err1 << endl;
    cout << "uni_err11: " << uni_err1 << endl;


    cout << "AVL weight: " << compute_avl_total_weight(tree.weight_tree.getRoot()) << " " << tree.total_weight << endl;
    
    // tree.dfs_quadtree(tree.getRoot());
    // double temp_total_weight = 0.0; int temp_count = 0;
    // tree.weight_tree.printInOrder(tree.weight_tree.root, temp_total_weight, temp_count);


    for (size_t i = window_size; i < stream.size(); ++i) {
        window.push_back(stream[i]);

        // tree
        auto start_our2 = Clock::now();
        tree.insert(stream[i].first, stream[i].second);

        if (window.size() > window_size) {
            tree.remove(window.front().first, window.front().second);
        }
        auto end_our2 = Clock::now();
        our_time += Duration(end_our2 - start_our2);

        // benchmark
        auto start_benchmark2 = Clock::now();
        if (stream[i].second == 'A') dc.insert_A(stream[i].first);
        else dc.insert_B(stream[i].first);
        if (window.size() > window_size) {
            if (window.front().second == 'A') dc.delete_A(0);
            else dc.delete_B(0);
        }
        auto end_benchmark2 = Clock::now();
        benchmark_time += Duration(end_benchmark2 - start_benchmark2);

        // tree,benchamrk,uniform
        auto start1 = Clock::now();
        if (stream[i].second != 'A') ann_oracle->insert(stream[i].first);
        if (window.size() > window_size) {
            if (window.front().second != 'A') ann_oracle->erase(stream[i].first);
        }
        auto end1 = Clock::now();
        auto time1 = Duration(end1 - start1);
        our_time += time1;
        uniform_time += time1;
        benchmark_time += time1;

        // others
        if (stream[i].second == 'B') current_B.push_back(stream[i].first);

        if (window.size() > window_size) {
            if (window.front().second == 'B') current_B.pop_front();
            window.pop_front();
        }

        if (i % query_interval == 0 && i > 0) {
            // 直接用迭代器区间构造：
            std::vector<std::vector<double>> B_vec(
                current_B.begin(),
                current_B.end()
            );
            cout << "AVL weight: " << compute_avl_total_weight(tree.weight_tree.getRoot()) << " " << tree.total_weight << endl;
            try {
                BruteForceNN ann;
                for (auto node : B_vec){
                    ann.insert(point(node));
                }
                // tree.dfs_quadtree(tree.getRoot());
                // double temp_total_weight = 0.0; int temp_count = 0;
                // tree.weight_tree.printInOrder(tree.weight_tree.root, temp_total_weight, temp_count);

                // BruteForceANN ann;
                // for (const auto& coords : B) {
                //     ann.insert(point(coords));
                // }


                int n = A.size(); // 这里A是整个数据集
                int sample_times = max(1, static_cast<int>(log2(n) * 10));
                cout << "n: " << n << " sample times: " << sample_times << endl;
                
                auto start_our3 = Clock::now();
                double our_cost = sample_points_from_quadtree(tree, A, ann_oracle, sample_times, ann);
                auto end_our3 = Clock::now();
                our_time += Duration(end_our3 - start_our3);

                auto start_uniform1 = Clock::now();
                double uniform_cost = sample_point_uniformly(A, ann_oracle, sample_times, n, ann);
                auto end_uniform1 = Clock::now();
                uniform_time += Duration(end_uniform1 - start_uniform1);
                
                auto start_benchmark3 = Clock::now();
                double chamfer = dc.current();
                auto end_benchmark3 = Clock::now();
                benchmark_time += Duration(end_benchmark3 - start_benchmark3);

                double our_err = (our_cost - chamfer) / chamfer;

                double uni_err = (uniform_cost - chamfer) / chamfer;

                std::cout << "Step " << i << " Results:\n";
                std::cout << "Our cost is: " << our_cost << std::endl;
                std::cout << "Chamfer Distance: " << chamfer << std::endl;
                cout << "our_err: " << our_err << endl;
                cout << "uni_err: " << uni_err << endl;

                cost_log << i << "," << our_cost << "," << chamfer << "," << our_err << "," << uni_err << "\n";
                // delete ann_lsh;
            } catch (const std::exception& e) {
                std::cerr << "Error at step " << i << ": " << e.what() << std::endl;
            }
        }
    }

    cost_log.close();
    cout << "our time: " << our_time.count() << endl;
    cout << "uniform time: " << uniform_time.count() << endl;
    cout << "benchmark time: " << benchmark_time.count() << endl;
    return 0;
}
