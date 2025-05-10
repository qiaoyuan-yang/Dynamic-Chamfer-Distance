#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>
#include <cassert>

#include "quad_tree.h"

using namespace std;

static std::random_device rd;
static std::mt19937 gen_sample(rd());


HSTNode::HSTNode(int d, const vector<int>& i, const vector<double>& o, double w)
    : depth(d), id_in_depth(i), origin(o), width(w), node_sampler(), node_total_weight(0) {}

void HSTNode::Build_Node_Sampler() {
    node_sampler.clear();
    node_total_weight = 0.0;

    for (auto& child : children) {
        double weight = child.second->B_count > 0 ? 0 : child.second->A_count;
        node_sampler.insert(make_pair(child.second->depth, child.second->id_in_depth), weight, child.second);
        node_total_weight += weight;
    }
}


QuadTree::QuadTree(int d, double delta, int max_depth)
    : dim(d), delta(delta), max_depth(max_depth) {
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.0, delta / 2);
    random_shift.resize(dim);
    for (int i = 0; i < dim; ++i) {
        random_shift[i] = dist(gen);
    }
}


// Transform vector<int> to bitset to be the key of unordered_map
template <std::size_t N>
bitset<N> QuadTree::to_bitset(const vector<int>& vec) {
    if (vec.size() > N) throw out_of_range("Error: child_idx size exceeds the limit of DIM bits.");
    bitset<N> bs;
    for (size_t i = 0; i < vec.size(); ++i){
        if (vec[i] != 0 && vec[i] != 1) {
            throw invalid_argument("Error: Invalid index value. Got vec[" + to_string(i) + "] = " + to_string(vec[i]) + ", must be 0 or 1.");
        }        
        bs[i] = vec[i];
    }
    return bs;
}


// Map the point to the subgrid ID of the current grid
vector<int> QuadTree::point_to_grid_index(const vector<double>& point, const vector<double>& origin, double width) {
    vector<int> child_idx(point.size());
    for (size_t i = 0; i < point.size(); ++i){
        child_idx[i] = static_cast<int>((point[i] - origin[i]) / (width / 2.0));
    }
    return child_idx;
}


// Check if a point in A matches any point in B
bool QuadTree::is_matched(const vector<double>& point_a, const vector<vector<double>>& dataset_b,
                const vector<double>& origin, double width) {
    if (dataset_b.size() == 0) return false;
    vector<int> child_idx_a = point_to_grid_index(point_a, origin, width);
    // Check the points in B. If a B point is found that belongs to the same subgrid, it does not match in the current cell.
    for (const auto& b : dataset_b) {
        if (point_to_grid_index(b, origin, width) == child_idx_a) {
            return false;  // They are in the same subgrid, so not a match.
        }
    }
    return true;  // No points from B in the same subgrid, hence it's a match.
}


// Recursively build the HST tree
void QuadTree::build(const vector<vector<double>>& A, const vector<vector<double>>& B, 
    const vector<vector<double>>& c_A, const vector<vector<double>>& c_B) {
    vector<double> min_corner(dim, numeric_limits<double>::max());
    for (const auto& pt : A) {
        for (size_t i = 0; i < dim; ++i)
            min_corner[i] = min(min_corner[i], pt[i]);
    }
    for (const auto& pt : B) {
        for (size_t i = 0; i < dim; ++i)
            min_corner[i] = min(min_corner[i], pt[i]);
    }

    root_origin.resize(dim);
    for (int i = 0; i < dim; ++i){
        root_origin[i] = min_corner[i] - random_shift[i];
        // cout << "root origin: " << root_origin[i] << endl;
    }

    if (c_A.size() == 0 && c_B.size() == 0){
        vector<int> root_id(dim, 0);
        root = make_shared<HSTNode>(0, root_id, root_origin, delta);
        root->Build_Node_Sampler();
        root->A_count = 0;
        root->B_count = 0;
        root->match_count = 0;
        root->is_leaf = false;
        root->parent = nullptr;
        // cout << "root->width: " << root->width << endl;
    }else{
        shared_ptr<HSTNode> parent_node = nullptr;
        root = build_subtree(c_A, c_B, 0, root_origin, root_origin, delta, parent_node);
    }
}


vector<int> QuadTree::calculate_current_id(const vector<double>& current_origin, const vector<double>& root_origin, double width) {
    vector<int> current_id(dim);
    for (size_t i = 0; i < dim; ++i) {
        double quo = (current_origin[i] - root_origin[i]) / width;
        if (fabs(quo - round(quo)) > 1e-4) {
            throw runtime_error("Warning: grid id is not an integer, quo = " + to_string(quo));
        }        
        current_id[i] = round(quo);
    }
    return current_id;
}


void QuadTree::add_points_to_subbuckets(const vector<vector<double>>& points_A, const vector<vector<double>>& points_B, 
                                               unordered_map<bitset<DIM>, vector<pair<string, vector<double>>>>& buckets,
                                               const vector<double>& current_origin, double width) {
    // Add points from A into the subgrids
    // cout << "width: " << width << endl;
    for (const auto& pt : points_A) {
        vector<int> child_idx = point_to_grid_index(pt, current_origin, width);
        auto child_idx_bitset = to_bitset(child_idx);
        // if (std::fabs(width - 18.0011) < 1e-4) cout << "create: " << child_idx_bitset << endl;
        buckets[child_idx_bitset].emplace_back("A", pt);
    }

    // Add points from B into the subgrids
    for (const auto& pt : points_B) {
        vector<int> child_idx = point_to_grid_index(pt, current_origin, width);
        auto child_idx_bitset = to_bitset(child_idx);
        // if (std::fabs(width - 18.0011) < 1e-4) {
            // cout << "point: ";
            // for (int i = 0; i < pt.size(); i++) cout << pt[i] << " ";
            // cout << endl;
            // cout << "create: " << child_idx_bitset << endl;
        // }
        // if (buckets.find(child_idx_bitset) != buckets.end()) {
        buckets[child_idx_bitset].emplace_back("B", pt);
        // }
    }
}


void QuadTree::process_subgrid(shared_ptr<HSTNode>& node, const unordered_map<bitset<DIM>, vector<pair<string, vector<double>>>>& buckets,
                               const vector<double>& current_origin, const vector<double>& root_origin, double width, int depth) {
    // For each subgrid, recursively build the tree
    for (const auto& bucket : buckets) {
        vector<double> child_origin = current_origin;
        for (size_t i = 0; i < dim; ++i) {
            child_origin[i] += (bucket.first[i] == 0) ? 0 : width / 2;
        }

        // Create a new HSTNode for this subgrid
        vector<vector<double>> points_A_in_subgrid;
        vector<vector<double>> points_B_in_subgrid;

        // Separate the points into A and B points for the next level
        for (const auto& point : bucket.second) {
            if (point.first == "A") {
                points_A_in_subgrid.push_back(point.second);
            } else if (point.first == "B") {
                points_B_in_subgrid.push_back(point.second);
            }
        }
        // Recursively build the subtree for this subgrid
        auto child_node = build_subtree(points_A_in_subgrid, points_B_in_subgrid,
            depth + 1, child_origin, root_origin, width / 2, node);
        // Only add as child if node was created (has points)
        if (child_node) {
            node->children[bucket.first] = child_node;
            assert(child_node.get() != node.get());
        }
    }
}


shared_ptr<HSTNode> QuadTree::build_subtree(const vector<vector<double>>& points_A, const vector<vector<double>>& points_B, 
    int depth, const vector<double>& current_origin, const vector<double>& root_origin, double width, shared_ptr<HSTNode>& parent_node) {
    // 对depth的id
    vector<int> current_id = calculate_current_id(current_origin, root_origin, width);
    
    auto node = make_shared<HSTNode>(depth, current_id, current_origin, width);
    node->Build_Node_Sampler();
    node->A_count = points_A.size();
    // 如果只有一个A点，记录下来
    if (node->A_count == 1) {
        node->single_A_point = points_A[0];
    }

    node->B_count = points_B.size();
    assert(parent_node.get() != node.get());
    node->parent = parent_node;
    node->is_leaf = false;

    // 计算当前层的匹配数
    for (const auto& A : points_A) {
        if (is_matched(A, points_B, current_origin, width)) {
            node->match_count++;
        }
    }

    assert(depth <= max_depth && "depth exceeded max_depth");
    if (depth == max_depth) {
        node->is_leaf = true;
        return node;
    }
    
    // 如果当前格子没有点，则不创建节点
    if (points_A.empty() && points_B.empty()) {
        return nullptr;
    }

    // Divide points A and B into subgrids
    unordered_map<bitset<DIM>, vector<pair<string, vector<double>>>> buckets;
    add_points_to_subbuckets(points_A, points_B, buckets, current_origin, width);

    // Process each subgrid recursively
    process_subgrid(node, buckets, current_origin, root_origin, width, depth);

    return node;
}

void QuadTree::Update_Sampler(){}

std::shared_ptr<HSTNode> QuadTree::sample_node_by_weight(std::shared_ptr<HSTNode> node) {
    if (!node || node->node_total_weight <= 0) return nullptr;
    
    std::uniform_real_distribution<> dis(0.0, node->node_total_weight);
    double random_weight = dis(gen_sample);
    
    // 从node_sampler中采样子节点
    std::shared_ptr<BBTNode> sampled = node->node_sampler.sample(random_weight);
    if (!sampled) return nullptr;
    
    // 查找对应的子节点
    for (auto& child : node->children) {
        if (child.second->depth == sampled->node_id.first && 
            child.second->id_in_depth == sampled->node_id.second) {
            return child.second;
        }
    }
    return nullptr;
}

double QuadTree::calculateTotalWeight() {
    double total = 0;
    std::function<void(std::shared_ptr<HSTNode>)> calc = [&](std::shared_ptr<HSTNode> node) {
        if (!node) return;
        total += node->width * node->match_count;
        // cout << "depth: " << node->depth << " width: " << node->width << " match_count: " << node->match_count << " weight: " << node->tree_weight << endl;
        for (auto& child : node->children) {
            calc(child.second);
        }
    };
    calc(root);
    return total;
}

void QuadTree::Build_Tree_Sampler() {
    // 清空AVL树
    weight_tree.clear();
    total_weight = 0;

    // 构建AVL树
    std::function<void(std::shared_ptr<HSTNode>)> build_weights = [&](std::shared_ptr<HSTNode> node) {
        if (!node) return;
        
        // 计算当前节点weight
        double weight = node->width * node->match_count;
        node->tree_weight = weight;
        total_weight += weight;
        
        // 添加到AVL树
        auto node_id = std::make_pair(node->depth, node->id_in_depth);
        weight_tree.insert(node_id, weight, node);
        
        // 递归处理子节点
        for (auto& child : node->children) {
            build_weights(child.second);
        }
    };
    
    build_weights(root);
}

int countNonZeroNodes_AVL(const std::shared_ptr<BBTNode>& node, double threshold = 1e-4) {
    if (!node) return 0;

    int count = (node->weight > threshold) ? 1 : 0;
    count += countNonZeroNodes_AVL(node->left, threshold);
    count += countNonZeroNodes_AVL(node->right, threshold);
    return count;
}

int countNonZeroNodes_quadtree(const std::shared_ptr<HSTNode>& node, double threshold = 1e-4) {
    if (!node) return 0;

    int count = (node->tree_weight > threshold) ? 1 : 0;
    for (auto& child : node->children) {
        count += countNonZeroNodes_quadtree(child.second, threshold);
    }

    return count;
}

std::shared_ptr<HSTNode> QuadTree::sample_by_weight() {
    // int count1 = countNonZeroNodes_AVL(weight_tree.getRoot());
    // int count2 = countNonZeroNodes_quadtree(root);

    // std::cout << "Tree 1 non-zero nodes: " << count1 << std::endl;
    // std::cout << "Tree 2 non-zero nodes: " << count2 << std::endl;


    if (!root || total_weight <= 0) return nullptr;
    
    std::uniform_real_distribution<> dis(0.0, total_weight);
    double random_weight = dis(gen_sample);
    // cout << "random_weight: " << random_weight << endl;
    
    // Use AVL tree's improved sampling method
    std::shared_ptr<BBTNode> result = weight_tree.sample(random_weight);
    if (!result) return nullptr;

    // 直接从 bbt->hst_ptr 拿到 HSTNode
    return result ? result->hst_ptr : nullptr;


    // if (!result) return nullptr;

    // // Find corresponding HSTNode
    // std::function<std::shared_ptr<HSTNode>(std::shared_ptr<HSTNode>)> find_node = 
    //     [&](std::shared_ptr<HSTNode> node) -> std::shared_ptr<HSTNode> {
    //         if (!node) return nullptr;
    //         if (node->depth == result->node_id.first && 
    //             node->id_in_depth == result->node_id.second) {
    //             return node;
    //         }
    //         for (auto& child : node->children) {
    //             auto found = find_node(child.second);
    //             if (found) return found;
    //         }
    //         return nullptr;
    //     };

    // return find_node(root);
}


void QuadTree::insert_update_A(shared_ptr<HSTNode> node, int C, const vector<double>& point){
    // 更新受影响节点的weight
    while(node != nullptr) {
        if (node->A_count == 0) {  // First A point in this node
            node->single_A_point = point;
        } else if (node->A_count == 1) {  // Second A point, clear single_A_point
            node->single_A_point.clear();
        }
        node->A_count += 1;
        // 增量更新node_sampler
        if (node->parent) {
            if (node->B_count == 0) {
                // 获取节点在parent sampler中的旧weight
                double old_weight = node->A_count - 1; // 之前的值
                double new_weight = node->A_count;     // 新值
                // 更新parent的sampler
                // node->parent->node_sampler.update(
                //     make_pair(node->depth, node->id_in_depth),
                //     old_weight,
                //     new_weight
                // );
                node->parent->node_sampler.remove(make_pair(node->depth, node->id_in_depth), old_weight);
                node->parent->node_sampler.insert(make_pair(node->depth, node->id_in_depth), new_weight, node);
                node->parent->node_total_weight += 1;
            }
        }
        
        if(node->B_count > 0 && C == 1) {
            node->match_count += 1;
            C = 0;
        }
        
        // 保存旧weight并更新total_weight
        double old_weight = node->tree_weight;
        total_weight -= old_weight;
        
        // 计算新weight
        double new_weight = node->width * node->match_count;
        total_weight += new_weight;
        node->tree_weight = new_weight;
        
        auto node_id = std::make_pair(node->depth, node->id_in_depth);
        
        if (fabs(old_weight - new_weight) > 1e-4) {
            // 使用旧值和新值更新AVL树
            weight_tree.remove(node_id, old_weight);
            weight_tree.insert(node_id, new_weight, node);
        }
        
        node = node->parent;
    }
}


void QuadTree::insert_update_B(shared_ptr<HSTNode> node, int C){
    // 更新当前节点
    node->B_count += 1;
    if (node->B_count == 1){
        node->match_count = node->A_count - C;
        C = node->A_count;
    }else{
        node->match_count = node->match_count - C;
        C = 0;
    }
    
    // 增量更新父节点的sampler
    if (node->parent) {
        if (node->B_count == 1) {
            // 从父节点sampler中移除该节点
            node->parent->node_sampler.remove(
                make_pair(node->depth, node->id_in_depth), node->A_count
            );
            node->parent->node_total_weight -= node->A_count;
        }
    }
    // 保存旧weight并更新total_weight
    double old_weight = node->tree_weight;
    total_weight -= old_weight;
    
    // 计算新weight
    double new_weight = node->width * node->match_count;
    node->tree_weight = new_weight;
    total_weight += new_weight;
    
    auto node_id = std::make_pair(node->depth, node->id_in_depth);
    
    if (fabs(old_weight - new_weight) > 1e-4) {
        // 使用旧值和新值更新AVL树
        // weight_tree.update(node_id, old_weight, new_weight);
        weight_tree.remove(node_id, old_weight);
        weight_tree.insert(node_id, new_weight, node);
    }

    // 递归更新父节点
    auto node_parent = node->parent;
    if (node_parent != nullptr) this->insert_update_B(node_parent, C);
}


void QuadTree::insert(vector<double>& point, char label) {
    shared_ptr<HSTNode> leaf_node = insert_recursive(root, point, label, 0);
    if (label == 'A'){
        insert_update_A(leaf_node, 1, point);
    }else if (label == 'B'){
        insert_update_B(leaf_node, 0);
    }else{
        throw std::invalid_argument("Error: wrong label. Valid labels are 'A' or 'B'.");
    }
}


// TODO: the decomposition does not continue after reaching the original leaf node.
shared_ptr<HSTNode> QuadTree::insert_recursive(shared_ptr<HSTNode>& node, const vector<double>& point, char label, int depth) {
    if (depth >= max_depth) {
        return node;
    }

    auto idx = point_to_grid_index(point, node->origin, node->width);
    auto bs = to_bitset(idx);

    if (node->children.find(bs) == node->children.end()) {
        vector<double> child_origin = node->origin;
        for (int i = 0; i < dim; ++i)
            child_origin[i] += (bs[i] == 0 ? 0 : node->width / 2);
        double new_width = node->width / 2;
        vector<int> new_id = calculate_current_id(child_origin, root_origin, new_width);
        auto child_node = make_shared<HSTNode>(depth + 1, new_id, child_origin, new_width);
        child_node->Build_Node_Sampler();
        child_node->parent = node;
        node->children[bs] = child_node;
        assert(child_node.get() != node.get());
        
        // // 插入对应的AVL节点，初始match_count为0
        // auto node_id = make_pair(child_node->depth, child_node->id_in_depth);
        // weight_tree.insert(node_id, 0);
    }

    return insert_recursive(node->children[bs], point, label, depth + 1);
}

shared_ptr<HSTNode> QuadTree::remove(const vector<double>& point, char label) {
    shared_ptr<HSTNode> leaf_node = remove_recursive(root, point, label, 0);
    if (label == 'A') {
        delete_update_A(leaf_node, 1);
    } else if (label == 'B') {
        delete_update_B(leaf_node, 0);
    } else {
        throw invalid_argument("Error: wrong label. Valid labels are 'A' or 'B'.");
    }
    return leaf_node;
}

shared_ptr<HSTNode> QuadTree::remove_recursive(shared_ptr<HSTNode> node, const vector<double>& point, char label, int depth) {
    if (depth >= max_depth) {
        return node;
    }

    auto idx = point_to_grid_index(point, node->origin, node->width);
    auto bs = to_bitset(idx);

    // cout << "bs: " << node->width << endl;

    if (node->children.find(bs) == node->children.end()) {
        for (const auto& pair : node->children) {
            // std::cout << "child: " << pair.first << std::endl;  // 输出 key
        }
        // cout << "bs: " << bs << " " << node->width << endl;
        throw runtime_error("Point not found in tree");
    }
    return remove_recursive(node->children[bs], point, label, depth + 1);
}

void QuadTree::delete_update_A(shared_ptr<HSTNode> node, int C) {
    // 更新受影响节点的weight
    while (node != nullptr) {
        // 更新single_A_point
        if (node->A_count == 1) {  // 删除后剩0个A点
            node->single_A_point.clear();
        } else if (node->A_count == 2) {  // 删除前有2个A点，删除后剩1个
            node->single_A_point.clear();
            for (auto& child : node->children) {
                if (child.second->A_count > 0) {
                    if (child.second->A_count == 1) {
                        node->single_A_point = child.second->single_A_point;
                    }
                    break;
                }
            }
        } else if (node->A_count == 0) {
            node->single_A_point.clear();
        }
        node->A_count--;
        // 增量更新父节点的sampler
        if (node->parent) {
            if (node->B_count == 0) {
                // 获取节点在parent sampler中的旧weight
                double old_weight = node->A_count + 1; // 之前的值
                double new_weight = node->A_count;     // 新值
                // 更新parent的sampler
                // node->parent->node_sampler.update(
                //     make_pair(node->depth, node->id_in_depth),
                //     old_weight,
                //     new_weight
                // );
                node->parent->node_sampler.remove(make_pair(node->depth, node->id_in_depth), old_weight);
                node->parent->node_sampler.insert(make_pair(node->depth, node->id_in_depth), new_weight, node);
                node->parent->node_total_weight -= 1;
            }
        }
        if (node->B_count > 0 && C == 1) {
            node->match_count--;
            C = 0;
        }
        
        // 保存旧weight并更新total_weight
        double old_weight = node->tree_weight;
        total_weight -= old_weight;
        
        // 计算新weight
        double new_weight = node->width * node->match_count;
        node->tree_weight = new_weight;
        total_weight += new_weight;
        
        auto node_id = std::make_pair(node->depth, node->id_in_depth);
        
        if (fabs(old_weight - new_weight) > 1e-4) {
            // 使用旧值和新值更新AVL树
            // weight_tree.update(node_id, old_weight, new_weight);
            weight_tree.remove(node_id, old_weight);
            weight_tree.insert(node_id, new_weight, node);
        }
        
        node = node->parent;
    }
}

void QuadTree::delete_update_B(shared_ptr<HSTNode> node, int C) {
    // 更新当前节点
    node->B_count--;
    if (node->B_count == 0){
        node->match_count = 0;
        C = node->A_count;
    }else{
        node->match_count += C;
        C = 0;
    }

    // 递归更新父节点
    auto parent_node = node->parent;
    if (parent_node != nullptr) this->delete_update_B(parent_node, C);
    

    // 增量更新父节点的sampler
    if (node->parent) {
        if (node->B_count == 0) { // 即将从1变为0
            // 重新添加该节点到父节点sampler
            node->parent->node_sampler.insert(
                make_pair(node->depth, node->id_in_depth),
                node->A_count,
                node
            );
            node->parent->node_total_weight += node->A_count;
        }
    }
    // 直接更新当前节点的weight
    double old_weight = node->tree_weight;
    total_weight -= old_weight;
    double new_weight = node->width * node->match_count;
    node->tree_weight = new_weight;
    total_weight += new_weight;

    if (fabs(old_weight - new_weight) > 1e-4) {
        auto node_id = std::make_pair(node->depth, node->id_in_depth);
        // 使用旧值和新值更新AVL树
        // weight_tree.update(node_id, old_weight, new_weight);
        weight_tree.remove(node_id, old_weight);
        weight_tree.insert(node_id, new_weight, node);
    }
}
