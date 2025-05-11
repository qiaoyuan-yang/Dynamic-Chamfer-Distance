#ifndef BALANCED_BINARY_TREE_H
#define BALANCED_BINARY_TREE_H

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>

class HSTNode;  // 告诉编译器：有这么个类

// Node structure for Balanced Binary Tree (AVL Tree)
class BBTNode {
public:
    std::pair<int, std::vector<int>>       node_id; // Quadtree node's unique ID
    double                       weight; // Quadtree node's weight
    double                       subtree_weight_sum; // Sum of weights in this subtree
    std::shared_ptr<HSTNode>          hst_ptr;              // 直接指向对应的 HSTNode
    std::shared_ptr<BBTNode>          left, right, parent; // Left, Right children and Parent pointer
    int height; // Height of the node (used for balancing)
    
    BBTNode(std::pair<int, std::vector<int>> id, double w, std::shared_ptr<HSTNode> hst) : 
        node_id(id), weight(w), subtree_weight_sum(w), hst_ptr(hst),
        height(1), left(nullptr), right(nullptr), parent(nullptr) {}
};

// AVL Tree structure
class AVLTree {
public:
    std::shared_ptr<BBTNode> root;
    AVLTree() : root(nullptr) {}

    void insert(std::pair<int, std::vector<int>> id, double weight, std::shared_ptr<HSTNode> hst_ptr);
    void remove(std::pair<int, std::vector<int>> id, double weight);
    void clear();
    void printInOrder(std::shared_ptr<BBTNode> node, double& total_weight, int& count) const;
    std::shared_ptr<BBTNode> getRoot() const { return root; }

    // 加权随机采样，返回包含 hst_ptr 的 BBTNode
    std::shared_ptr<BBTNode> sample(double random_weight) const;

    bool contains(std::pair<int, std::vector<int>> id, double weight) {
        auto node = root;
        while (node) {
            if (isLess(weight, id, node->weight, node->node_id)) {
                node = node->left;
            }
            else if (isGreater(weight, id, node->weight, node->node_id)) {
                node = node->right;
            }
            else {
                // (weight, id) 与当前节点完全相等
                return true;
            }
        }
        return false;
    }
    

private:
    // Helper functions for AVL Tree
    int getHeight(const std::shared_ptr<BBTNode>& node);
    int getBalanceFactor(const std::shared_ptr<BBTNode>& node);
    std::shared_ptr<BBTNode> rightRotate(std::shared_ptr<BBTNode> y) {
        std::shared_ptr<BBTNode> x = y->left;
        std::shared_ptr<BBTNode> T2 = x->right;

        // Perform rotation
        x->right = y;
        y->left = T2;

        // Update parents
        x->parent = y->parent;
        y->parent = x;
        if (T2) T2->parent = y;

        // Update heights
        y->height = std::max(getHeight(y->left), getHeight(y->right)) + 1;
        x->height = std::max(getHeight(x->left), getHeight(x->right)) + 1;

        // Update subtree weights
        y->subtree_weight_sum = y->weight;
        if (y->left) y->subtree_weight_sum += y->left->subtree_weight_sum;
        if (y->right) y->subtree_weight_sum += y->right->subtree_weight_sum;
        
        x->subtree_weight_sum = x->weight;
        if (x->left) x->subtree_weight_sum += x->left->subtree_weight_sum;
        if (x->right) x->subtree_weight_sum += x->right->subtree_weight_sum;

        return x;
    }

    std::shared_ptr<BBTNode> leftRotate(std::shared_ptr<BBTNode> x) {
        std::shared_ptr<BBTNode> y = x->right;
        std::shared_ptr<BBTNode> T2 = y->left;

        // Perform rotation
        y->left = x;
        x->right = T2;

        // Update parents
        y->parent = x->parent;
        x->parent = y;
        if (T2) T2->parent = x;

        // Update heights
        x->height = std::max(getHeight(x->left), getHeight(x->right)) + 1;
        y->height = std::max(getHeight(y->left), getHeight(y->right)) + 1;

        // Update subtree weights
        x->subtree_weight_sum = x->weight;
        if (x->left) x->subtree_weight_sum += x->left->subtree_weight_sum;
        if (x->right) x->subtree_weight_sum += x->right->subtree_weight_sum;
        
        y->subtree_weight_sum = y->weight;
        if (y->left) y->subtree_weight_sum += y->left->subtree_weight_sum;
        if (y->right) y->subtree_weight_sum += y->right->subtree_weight_sum;

        return y;
    }

    // 定义比较算法：先比 weight，weight 相等时再比 id
    bool isLess (double w1, const std::pair<int, std::vector<int>>& id1,
        double w2, const std::pair<int, std::vector<int>>& id2) {
        if (w1 < w2) return true;
        if (w1 > w2) return false;
        return id1 < id2;
    };
    bool isGreater (double w1, const std::pair<int, std::vector<int>>& id1,
            double w2, const std::pair<int, std::vector<int>>& id2) {
        if (w1 > w2) return true;
        if (w1 < w2) return false;
        return id1 > id2;
    };

    std::shared_ptr<BBTNode> insertNode(
        std::shared_ptr<BBTNode>& node,
        const std::pair<int, std::vector<int>>& id,
        double weight,
        std::shared_ptr<HSTNode> hst_ptr,
        std::shared_ptr<BBTNode> parent)
    {
        // 过滤过轻节点
        if (weight <= 1e-4) return node;
    
        // 空树情况：新建节点
        if (!node) {
            auto newNode = std::make_shared<BBTNode>(id, weight, hst_ptr);
            newNode->parent = parent;
            return newNode;
        }
    
        // 按 (weight, id) 决定插入左子树还是右子树
        if (isLess(weight, id, node->weight, node->node_id)) {
            node->left = insertNode(node->left, id, weight, hst_ptr, node);
        }
        else if (isGreater(weight, id, node->weight, node->node_id)) {
            node->right = insertNode(node->right, id, weight, hst_ptr, node);
        }
        else {
            // 完全相同（weight 和 id 都相等），不插入重复节点
            return node;
        }
    
        // 更新高度和子树权重和
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));
        node->subtree_weight_sum = node->weight;
        if (node->left)  node->subtree_weight_sum += node->left->subtree_weight_sum;
        if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;
    
        // 计算平衡因子并做 AVL 旋转
        int balance = getBalanceFactor(node);
    
        // LL 型
        if (balance > 1 &&
            isLess(weight, id, node->left->weight, node->left->node_id))
        {
            return rightRotate(node);
        }
        // RR 型
        if (balance < -1 &&
            isGreater(weight, id, node->right->weight, node->right->node_id))
        {
            return leftRotate(node);
        }
        // LR 型
        if (balance > 1 &&
            isGreater(weight, id, node->left->weight, node->left->node_id))
        {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
        // RL 型
        if (balance < -1 &&
            isLess(weight, id, node->right->weight, node->right->node_id))
        {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
    
        return node;
    }
    

    std::shared_ptr<BBTNode> deleteNode(
        std::shared_ptr<BBTNode>& node,
        std::pair<int, std::vector<int>> id,
        double weight)
    {
        // 空节点或权重过小，直接返回
        if (!node || weight <= 1e-4) return node;
    
        // 根据 (weight, id) 决定向左还是向右递归删除
        if (isLess(weight, id, node->weight, node->node_id)) {
            node->left = deleteNode(node->left, id, weight);
        }
        else if (isGreater(weight, id, node->weight, node->node_id)) {
            node->right = deleteNode(node->right, id, weight);
        }
        else {
            // 找到待删节点（权重和 id 完全相等）
            if (!node->left || !node->right) {
                // 只有一个子节点或无子节点
                std::shared_ptr<BBTNode> temp = node->left ? node->left : node->right;
                if (!temp) {
                    // 无子节点：直接删除
                    temp = node;
                    node = nullptr;
                } else {
                    // 有一个子节点：用子节点替代
                    *node = *temp;
                }
            } else {
                // 有两个子节点：找到后继（右子树最小）
                std::shared_ptr<BBTNode> temp = minValueNode(node->right);
                // 用后继的数据覆盖当前节点
                node->node_id = temp->node_id;
                node->weight  = temp->weight;
                node->hst_ptr = temp->hst_ptr;
                // 删除后继节点
                node->right = deleteNode(node->right, temp->node_id, temp->weight);
            }
        }
    
        if (!node) return node;
    
        // 更新高度和子树权重和
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));
        node->subtree_weight_sum = node->weight;
        if (node->left)  node->subtree_weight_sum += node->left->subtree_weight_sum;
        if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;
    
        // 重新平衡 AVL 树
        int balance = getBalanceFactor(node);
        // LL 型
        if (balance > 1 && getBalanceFactor(node->left) >= 0) {
            return rightRotate(node);
        }
        // LR 型
        if (balance > 1 && getBalanceFactor(node->left) < 0) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
        // RR 型
        if (balance < -1 && getBalanceFactor(node->right) <= 0) {
            return leftRotate(node);
        }
        // RL 型
        if (balance < -1 && getBalanceFactor(node->right) > 0) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
    
        return node;
    }    

    std::shared_ptr<BBTNode> minValueNode(std::shared_ptr<BBTNode> node) {
        std::shared_ptr<BBTNode> current = node;
        while (current->left) {
            current = current->left;
        }
        return current;
    }
};

#endif
