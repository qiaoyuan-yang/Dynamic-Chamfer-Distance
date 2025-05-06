#include "balanced_binary_tree.h"

// TODO: When building, sort by weight first, then by id. The smaller id in the left.
// Get the height of a node
int AVLTree::getHeight(const shared_ptr<BBTNode>& node) {
    return node ? node->height : 0;
}

// Get the balance factor of a node
int AVLTree::getBalanceFactor(const shared_ptr<BBTNode>& node) {
    return node ? getHeight(node->left) - getHeight(node->right) : 0;
}

// Insert a node with the given ID and weight
void AVLTree::insert(pair<int, vector<int>> id, double weight) {
    root = insertNode(root, id, weight, nullptr); // Pass nullptr as root's parent
}

// Remove a node with the given ID
void AVLTree::remove(pair<int, vector<int>> id, double weight) {
    root = deleteNode(root, id, weight);
}

// Update a node in the AVL Tree
// void AVLTree::update(pair<int, vector<int>> old_id, double old_weight, double new_weight) {
//     updateNode(root, old_id, old_weight, new_weight);
// }

// Check if a node exists in the AVL Tree
bool AVLTree::contains(pair<int, vector<int>> id, double weight) {
    shared_ptr<BBTNode> node = root;
    while (node) {
        if (node->weight == weight) {
            if (node->node_id == id)
                return true;
            else
                node = (id < node->node_id) ? node->left : node->right;
        } else if (weight < node->weight) {
            node = node->left;  
        } else {
            node = node->right;
        }
    }
    return false;
}

// Print the AVL Tree in order
void AVLTree::clear() {
    root = nullptr;
}

void AVLTree::printInOrder(shared_ptr<BBTNode> node, double& total_weight, int& count) const {
    if (node) {
        printInOrder(node->left, total_weight, count);
        
        total_weight += node->weight;
        count++;

        cout << "Node (" << node->node_id.first << ", [";
        for (size_t i = 0; i < node->node_id.second.size(); ++i) {
            cout << node->node_id.second[i];
            if (i < node->node_id.second.size() - 1) cout << ", ";
        }
        cout << "]), Weight: " << node->weight << endl;

        printInOrder(node->right, total_weight, count);
    }
}

