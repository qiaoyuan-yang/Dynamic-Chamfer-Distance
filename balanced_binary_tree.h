#ifndef BALANCED_BINARY_TREE_H
#define BALANCED_BINARY_TREE_H

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>


using namespace std;

class HSTNode;  // 告诉编译器：有这么个类
// Node structure for Balanced Binary Tree (AVL Tree)
class BBTNode {
public:
    pair<int, vector<int>>       node_id; // Quadtree node's unique ID
    double                       weight; // Quadtree node's weight
    double                       subtree_weight_sum; // Sum of weights in this subtree
    shared_ptr<HSTNode>          hst_ptr;              // 直接指向对应的 HSTNode
    shared_ptr<BBTNode>          left, right, parent; // Left, Right children and Parent pointer
    int height; // Height of the node (used for balancing)
    
    BBTNode(pair<int, vector<int>> id, double w, shared_ptr<HSTNode> hst) : 
        node_id(id), weight(w), subtree_weight_sum(w), hst_ptr(hst),
        height(1), left(nullptr), right(nullptr), parent(nullptr) {}
};

// AVL Tree structure
class AVLTree {
public:
    shared_ptr<BBTNode> root;
    AVLTree() : root(nullptr) {}

    void insert(pair<int, vector<int>> id, double weight, shared_ptr<HSTNode> hst_ptr);
    void remove(pair<int, vector<int>> id, double weight);
    bool contains(pair<int, vector<int>> id, double weight);
    void clear();
    void printInOrder(shared_ptr<BBTNode> node, double& total_weight, int& count) const;
    shared_ptr<BBTNode> getRoot() const { return root; }

    // 加权随机采样，返回包含 hst_ptr 的 BBTNode
    shared_ptr<BBTNode> sample(double random_weight) const;

private:
    // Helper functions for AVL Tree
    int getHeight(const shared_ptr<BBTNode>& node);
    int getBalanceFactor(const shared_ptr<BBTNode>& node);
    shared_ptr<BBTNode> rightRotate(shared_ptr<BBTNode> y) {
        shared_ptr<BBTNode> x = y->left;
        shared_ptr<BBTNode> T2 = x->right;

        // Perform rotation
        x->right = y;
        y->left = T2;

        // Update parents
        x->parent = y->parent;
        y->parent = x;
        if (T2) T2->parent = y;

        // Update heights
        y->height = max(getHeight(y->left), getHeight(y->right)) + 1;
        x->height = max(getHeight(x->left), getHeight(x->right)) + 1;

        // Update subtree weights
        y->subtree_weight_sum = y->weight;
        if (y->left) y->subtree_weight_sum += y->left->subtree_weight_sum;
        if (y->right) y->subtree_weight_sum += y->right->subtree_weight_sum;
        
        x->subtree_weight_sum = x->weight;
        if (x->left) x->subtree_weight_sum += x->left->subtree_weight_sum;
        if (x->right) x->subtree_weight_sum += x->right->subtree_weight_sum;

        return x;
    }

    shared_ptr<BBTNode> leftRotate(shared_ptr<BBTNode> x) {
        shared_ptr<BBTNode> y = x->right;
        shared_ptr<BBTNode> T2 = y->left;

        // Perform rotation
        y->left = x;
        x->right = T2;

        // Update parents
        y->parent = x->parent;
        x->parent = y;
        if (T2) T2->parent = x;

        // Update heights
        x->height = max(getHeight(x->left), getHeight(x->right)) + 1;
        y->height = max(getHeight(y->left), getHeight(y->right)) + 1;

        // Update subtree weights
        x->subtree_weight_sum = x->weight;
        if (x->left) x->subtree_weight_sum += x->left->subtree_weight_sum;
        if (x->right) x->subtree_weight_sum += x->right->subtree_weight_sum;
        
        y->subtree_weight_sum = y->weight;
        if (y->left) y->subtree_weight_sum += y->left->subtree_weight_sum;
        if (y->right) y->subtree_weight_sum += y->right->subtree_weight_sum;

        return y;
    }

    shared_ptr<BBTNode> insertNode(shared_ptr<BBTNode>& node, pair<int, vector<int>> id, double weight, 
        shared_ptr<HSTNode> hst_ptr, shared_ptr<BBTNode> parent) {
        if (weight <= 1e-4) return node;
        if (!node) {
            auto newNode = make_shared<BBTNode>(id, weight, hst_ptr);
            newNode->parent = parent;
            return newNode;
        }

        if (id < node->node_id) {
            node->left = insertNode(node->left, id, weight, hst_ptr, node);
        } else if (id > node->node_id) {
            node->right = insertNode(node->right, id, weight, hst_ptr, node);
        } else {
            return node; // Duplicate nodes not allowed
        }

        // Update height and subtree weight
        node->height = 1 + max(getHeight(node->left), getHeight(node->right));
        node->subtree_weight_sum = node->weight;
        if (node->left) node->subtree_weight_sum += node->left->subtree_weight_sum;
        if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;

        // Rebalance tree
        int balance = getBalanceFactor(node);
        if (balance > 1 && id < node->left->node_id) {
            return rightRotate(node);
        }
        if (balance < -1 && id > node->right->node_id) {
            return leftRotate(node);
        }
        if (balance > 1 && id > node->left->node_id) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
        if (balance < -1 && id < node->right->node_id) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    shared_ptr<BBTNode> deleteNode(shared_ptr<BBTNode>& node, pair<int, vector<int>> id, double weight) {
        if (!node || weight <= 1e-4) return node;

        if (id < node->node_id) {
            node->left = deleteNode(node->left, id, weight);
        } else if (id > node->node_id) {
            node->right = deleteNode(node->right, id, weight);
        } else {
            if (!node->left || !node->right) {
                shared_ptr<BBTNode> temp = node->left ? node->left : node->right;
                if (!temp) {
                    temp = node;
                    node = nullptr;
                } else {
                    *node = *temp;
                }
            } else {
                shared_ptr<BBTNode> temp = minValueNode(node->right);
                node->node_id = temp->node_id;
                node->weight = temp->weight;
                node->right = deleteNode(node->right, temp->node_id, temp->weight);
            }
        }

        if (!node) {
            // TODO: 为什么会出现remove点找不到的情况？
            // cout << "222 " << weight << endl;
            return node;
        }

        // Update height and subtree weight
        node->height = 1 + max(getHeight(node->left), getHeight(node->right));
        node->subtree_weight_sum = node->weight;
        if (node->left) node->subtree_weight_sum += node->left->subtree_weight_sum;
        if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;

        // Rebalance tree
        int balance = getBalanceFactor(node);
        if (balance > 1 && getBalanceFactor(node->left) >= 0) {
            return rightRotate(node);
        }
        if (balance > 1 && getBalanceFactor(node->left) < 0) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
        if (balance < -1 && getBalanceFactor(node->right) <= 0) {
            return leftRotate(node);
        }
        if (balance < -1 && getBalanceFactor(node->right) > 0) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }

        return node;
    }

    shared_ptr<BBTNode> minValueNode(shared_ptr<BBTNode> node) {
        shared_ptr<BBTNode> current = node;
        while (current->left) {
            current = current->left;
        }
        return current;
    }

    // void updateNode(shared_ptr<BBTNode> node, pair<int, vector<int>> old_id, double old_weight, double new_weight) {
    //     if (!node) return;

    //     if (old_id < node->node_id) {
    //         updateNode(node->left, old_id, old_weight, new_weight);
    //     } else if (old_id > node->node_id) {
    //         updateNode(node->right, old_id, old_weight, new_weight);
    //     } else {
    //         node->weight = new_weight;
    //     }

    //     // Update subtree weight sum
    //     if (node) {
    //         node->subtree_weight_sum = node->weight;
    //         if (node->left) node->subtree_weight_sum += node->left->subtree_weight_sum;
    //         if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;
    //     }
    // }
};

#endif
