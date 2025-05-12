#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <memory>
#include <deque>
#include <future>
#include <fstream>
#include <type_traits>
#include "quad_tree.h"
#include "aspect_ratio_depth.h"
#include "BruteForceNN.h"
#include "data_preprocess.h"
#include "datatype.h"
#include "ANN-Oracle.h"
#include "baseline/benchmark.h"
#include <sys/resource.h>
#include <mach/mach.h>
// #include <omp.h>


using namespace std;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
static thread_local std::mt19937 gen(std::random_device{}());


std::size_t getCurrentRSSBytes()
{
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) != KERN_SUCCESS)
        return 0;
    return info.resident_size;          // 单位：Bytes
}


double sample_point_uniformly(
    const std::vector<std::vector<double>>& A,
    ANNOracle* ann_oracle,
    int sample_times,
    int n,
    BruteForceNN& ann 
) {
    // cout << "uniform sample times: " << sample_times << endl;
    std::uniform_int_distribution<size_t> dist(0, A.size() - 1);

    double sum = 0.0;
    for (int i = 0; i < sample_times; ++i) {
        size_t idx = dist(gen);
        const auto& sample_pt = A[idx];
        
        // double d = ann_oracle->distance(sample_pt);
        double d = ann.query(point(sample_pt));
        sum += d;
    }

    return sum * static_cast<double>(n) / sample_times;
}


double sample_points_from_quadtree(QuadTree& tree, const vector<vector<double>>& A, ANNOracle* ann_oracle, 
    int& sample_times, BruteForceNN ann) {
    // double sample_points_from_quadtree(QuadTree& tree, const vector<vector<double>>& A, ANNOracle* ann_oracle, 
    //     int& sample_times) {
    // cout << "our sample times: " << sample_times << endl;

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
            // cout << node->A_count << " " << node->single_A_point.size() << endl;
            if (!node->single_A_point.empty()) {
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
    // sample_times = set_sampler.size();
    // cout << "actually sampled point size: " << set_sampler.size() << endl;

    return sum / times;
}

double compute_avl_total_weight(const std::shared_ptr<BBTNode>& node) {
    if (!node) return 0.0;

    double left_sum = compute_avl_total_weight(node->left);
    double right_sum = compute_avl_total_weight(node->right);
    return left_sum + right_sum + node->weight;
}



int main() {
    // 1. 读取原始数据集
    // A with small size; B with big size
    // auto dataA = read_csv("dataset/Text_Embeddings/federalist/A_raw.csv");
    // auto dataB = read_csv("dataset/Text_Embeddings/federalist/B_raw.csv");
    auto dataA = read_csv("dataset/MNIST/test.csv");
    auto dataB = read_csv("dataset/MNIST/train.csv");
    // auto dataA = read_csv("dataset/SIFT/test.csv");
    // auto dataB = read_csv("dataset/SIFT/train.csv");
    // auto dataA = read_csv("dataset/Gaussian/A.csv");
    // auto dataB = read_csv("dataset/Gaussian/B.csv");
    // auto dataA = read_csv("test_dataset/A3.csv");
    // auto dataB = read_csv("test_dataset/B3.csv");
    std::vector<std::vector<double>> A, B;
    if (dataA.size() >= dataB.size()) {
        A = std::move(dataA);
        B = std::move(dataB);
    } else {
        A = std::move(dataB);
        B = std::move(dataA);
    }
    if (A.empty() || B.empty()) {
        std::cerr << "Error: Failed to read input files" << std::endl;
        return 1;
    }
    if (A[0].size() != B[0].size()) {
        throw std::runtime_error("Error: Dimension mismatch between A and B. "
                                 "A has dimension " + std::to_string(A[0].size()) +
                                 ", B has dimension " + std::to_string(B[0].size()) + ".");
    }
    // std::vector<std::vector<double>> A = read_csv("test_dataset/A3.csv");
    // std::vector<std::vector<double>> B = read_csv("test_dataset/B3.csv");

    // 2. 计算滑动窗口参数,首先定windoesize，然后根据需要50个query output定query_interval
    int window_size = B.size() / 100;  // 3  100
    int query_interval = max(static_cast<int>(B.size() / 56), 1);  //75  56
    cout << "window_size: " << window_size << " query_interval: " << query_interval << endl;

    // 3. 计算数据集全局参数
    Duration our_time(0), uniform_time(0), benchmark_time(0);
    std::vector<std::vector<double>> combined = A;
    combined.insert(combined.end(), B.begin(), B.end());
    double max_dist = compute_2approx_diameter(combined);
    double aspect_ratio = calculate_aspect_ratio(A, B, max_dist);
    int depth = calculate_quadtree_depth(aspect_ratio);
    cout << "diameter: " << max_dist
         << " aspect_ratio: " << aspect_ratio
         << " max depth: " << depth << endl;

    // 4. 并行构建 R 个 Quadtree 副本
    auto start_ours1 = Clock::now();
    const int R = 5;
    vector<future<QuadTree>> build_futs;
    build_futs.reserve(R);
    for (int r = 0; r < R; ++r) {
        build_futs.emplace_back(
            async(launch::async, [&, r]() {
                QuadTree tree(DIM, max_dist * 2, depth);
                vector<vector<double>> A_empty, B_empty;
                tree.build(A, B, A_empty, B_empty);
                tree.Build_Tree_Sampler();
                for (auto& a : A) tree.insert(a, 'A');
                
                return tree;
            })
        );
    }
    vector<QuadTree> trees;
    trees.reserve(R);
    for (auto& fut : build_futs) {
        trees.push_back(std::move(fut.get()));
    }
    auto end_ours1 = Clock::now();
    // our_time += Duration(end_ours1 - start_ours1);

    // 5. 初始化单个ANNOracle
    ANNOracle* ann_oracle = new ANNOracle();

    // 6. 初始化baseline和数据结构
    auto start_benchmark1 = Clock::now();
    DynamicChamfer dc(A, {});
    auto end_benchmark1 = Clock::now();
    // benchmark_time += Duration(end_benchmark1 - start_benchmark1);
    std::deque<std::pair<std::vector<double>, char>> window;
    std::deque<std::vector<double>> current_B;
    std::ofstream cost_log("output/cost_log.csv");
    if (!cost_log.is_open()) {
        std::cerr << "Error: Failed to open cost_log.csv for writing." << std::endl;
        return 1;
    }
    cost_log << "step,ours,chamfer,our_err,uni_err\n";
    

    // 7. 主窗口循环
    for (size_t i = 0; i < B.size(); ++i) {
        auto p = B[i];
        window.emplace_back(p, 'B');

        // 更新五棵树
        auto start_our2 = Clock::now();
        vector<future<void>> update_futs;
        update_futs.reserve(R);
        for (int r = 0; r < R; ++r) {
            update_futs.emplace_back(
                async(launch::async, [&, r]() {
                    trees[r].insert(p, 'B');
                    if (window.size() > window_size) {
                        trees[r].remove(window.front().first,
                                        window.front().second);
                    }
                })
            );
        }
        for (auto& f : update_futs) f.get();
        auto end_our2 = Clock::now();
        our_time += Duration(end_our2 - start_our2);

        // benchmark
        auto start_benchmark2 = Clock::now();
        // if (stream[i].second == 'A') dc.insert_A(stream[i].first); else 
        dc.insert_B(p);
        if (window.size() > window_size) {
            // if (window.front().second == 'A') dc.delete_A(0); else 
            dc.delete_B(0);
        }
        auto end_benchmark2 = Clock::now();
        benchmark_time += Duration(end_benchmark2 - start_benchmark2);

        // tree,benchamrk,uniform
        auto start1 = Clock::now();
        // if (stream[i].second == 'B') 
        // ann_oracle->insert(p);
        if (window.size() > window_size) {
            // if (window.front().second == 'B') 
            // ann_oracle->erase(p);
        }
        auto end1 = Clock::now();
        auto time1 = Duration(end1 - start1);
        our_time += time1;
        uniform_time += time1;
        benchmark_time += time1;

        // others
        // if (stream[i].second == 'B') 
        current_B.push_back(p);
        if (window.size() > window_size) {
            if (window.front().second == 'B') current_B.pop_front();
            window.pop_front();
        }

        if (i >= window_size && i % query_interval == 0) {
            std::size_t mem = getCurrentRSSBytes();
            std::cout << "[Iter " << i
                    << "]  RSS = " << mem / 1024.0 / 1024.0 << " MB\n";

            std::vector<std::vector<double>> B_vec(
                current_B.begin(),
                current_B.end()
            );
            try {
                BruteForceNN ann;
                for (auto node : B_vec){
                    ann.insert(point(node));
                }

                int n = A.size(); // 这里A是整个数据集
                int sample_times = max(1, static_cast<int>(log2(n) * 20)); // text embedding
                // int sample_times = 50;
                cout << "n: " << n << " sample times: " << sample_times << endl;
                
                auto start_our3 = Clock::now();
                // double our_cost = sample_points_from_quadtree(tree, A, ann_oracle, sample_times, ann);
                // double our_cost = sample_points_from_quadtree(tree, A, ann_oracle, sample_times);
                // 并行调用 R 次采样
                vector<future<double>> sample_futs;
                sample_futs.reserve(R);
                for (int r = 0; r < R; ++r) {
                    sample_futs.emplace_back(
                        async(launch::async, [&, r]() {
                            return sample_points_from_quadtree(
                                trees[r], A, ann_oracle, sample_times, ann
                            );
                        })
                    );
                }
                vector<double> our_costs(R);
                for (int r = 0; r < R; ++r) {
                    our_costs[r] = sample_futs[r].get();
                }
                // auto mid = our_costs.begin() + (our_costs.size() - 1)/2;
                auto mid = our_costs.begin() + our_costs.size()/2;
                nth_element(our_costs.begin(), mid, our_costs.end());
                double our_median_cost = *mid;
                auto end_our3 = Clock::now();
                our_time += Duration(end_our3 - start_our3);

                auto start_uniform1 = Clock::now();
                double uniform_cost = sample_point_uniformly(A, ann_oracle, sample_times, n, ann);
                // double uniform_cost = sample_point_uniformly(A, ann_oracle, sample_times, n);
                auto end_uniform1 = Clock::now();
                uniform_time += Duration(end_uniform1 - start_uniform1);
                
                auto start_benchmark3 = Clock::now();
                double chamfer = dc.current();
                auto end_benchmark3 = Clock::now();
                benchmark_time += Duration(end_benchmark3 - start_benchmark3);

                double our_err = (our_median_cost - chamfer) / chamfer;
                double uni_err = (uniform_cost - chamfer) / chamfer;

                std::cout << "Step " << i << " Results:\n";
                std::cout << "Our cost is: " << our_median_cost << std::endl;
                std::cout << "Chamfer Distance: " << chamfer << std::endl;
                cout << "our_err: " << our_err << endl;
                cout << "uni_err: " << uni_err << endl;

                cost_log << i << "," << our_median_cost << "," << chamfer << "," << our_err << "," << uni_err << std::endl;
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
