#include <vector>
#include <unordered_map>
#include <memory>
#include <bitset>
#include <limits>
#include <stdexcept>
#include <cmath>
#include "balanced_binary_tree.h"
#include "dataset_config.h"

struct HSTNode {
    int depth;
    std::vector<int> id_in_depth;
    std::vector<double> origin;
    bool is_leaf;
    double width;
    int A_count = 0;
    int B_count = 0;
    int match_count = 0;
    double tree_weight = 0.0; // 存储节点的weight值
    std::unordered_map<std::bitset<DIM>, std::shared_ptr<HSTNode>> children;
    std::shared_ptr<HSTNode> parent;
    AVLTree node_sampler; // 节点自身的sampler
    double node_total_weight = 0.0; // 节点sampler的总weight
    std::vector<double> single_A_point; // 当A_count=1时存储的唯一A点
    
    HSTNode(int d, const std::vector<int>& i, const std::vector<double>& o, double w);
    void Build_Node_Sampler();
};

class QuadTree {
public:
    QuadTree(int d, double delta, int max_depth);
    void build(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B, 
        const vector<vector<double>>& c_A, const vector<vector<double>>& c_B);
    std::shared_ptr<HSTNode> sample_by_weight(); // 基于全局weight_tree采样
    std::shared_ptr<HSTNode> sample_node_by_weight(std::shared_ptr<HSTNode> node); // 基于node_sampler采样子节点
    std::shared_ptr<HSTNode> getRoot() { return root; }
    double calculateTotalWeight(); // 计算总weight的函数
    void Build_Tree_Sampler();
    void insert(std::vector<double>& point, char label);
    std::shared_ptr<HSTNode> remove(const std::vector<double>& point, char label);
    double total_weight = 0.0; // 维护总weight
    AVLTree weight_tree;

    void dfs_quadtree(const std::shared_ptr<HSTNode>& root) {
        if (!root) return;
        // 访问当前节点
        std::cout
          << "quadtree match_count: " << root->match_count
          << " A count: "      << root->A_count
          << " B count: "      << root->B_count
          << " width: "       << root->width
          << " weight: "      << root->tree_weight
          << std::endl;
        for (auto i : root->origin){
            cout << i << " ";
        }
        cout << endl;
    
        for (const auto& [key, child] : root->children) {
            dfs_quadtree(child);
        }
    }

private:
    int dim;
    double delta;
    int max_depth;
    std::vector<double> random_shift;
    std::shared_ptr<HSTNode> root;
    std::vector<double> root_origin;

    template <std::size_t N = DIM>
    std::bitset<N> to_bitset(const std::vector<int>& vec);
    std::vector<int> point_to_grid_index(const std::vector<double>& point, const std::vector<double>& origin, double width);
    bool is_matched(const std::vector<double>& a, const std::vector<std::vector<double>>& B, const std::vector<double>& origin, double width);
    std::shared_ptr<HSTNode> build_subtree(const std::vector<std::vector<double>>& points_A, const std::vector<std::vector<double>>& points_B, 
        int depth, const std::vector<double>& current_origin, const std::vector<double>& root_origin, double width, std::shared_ptr<HSTNode>& parent_node);
    std::vector<int> calculate_current_id(const std::vector<double>& current_origin, const std::vector<double>& root_origin, double width);
    void add_points_to_subbuckets(const std::vector<std::vector<double>>& points_A, const std::vector<std::vector<double>>& points_B, 
        std::unordered_map<std::bitset<DIM>, std::vector<std::pair<std::string, std::vector<double>>>>& buckets,
        const std::vector<double>& current_origin, double width);
    void process_subgrid(std::shared_ptr<HSTNode>& node, const std::unordered_map<std::bitset<DIM>, std::vector<std::pair<std::string, std::vector<double>>>>& buckets,
        const std::vector<double>& current_origin, const std::vector<double>& root_origin, double width, int depth);
    std::shared_ptr<HSTNode> insert_recursive(std::shared_ptr<HSTNode>& node, const std::vector<double>& point, char label, int depth);
    std::shared_ptr<HSTNode> remove_recursive(std::shared_ptr<HSTNode> node, const std::vector<double>& point, char label, int depth);
    
    void Update_Sampler();
    
    void delete_update_A(std::shared_ptr<HSTNode> node, int C);
    void delete_update_B(std::shared_ptr<HSTNode> node, int C);
    void insert_update_A(std::shared_ptr<HSTNode> node, int C, const std::vector<double>& point);
    void insert_update_B(std::shared_ptr<HSTNode> node, int C);
};
