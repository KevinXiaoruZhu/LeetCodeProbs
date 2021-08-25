//
// Created by Xiaoru_Zhu on 2020/10/17.
//

#ifndef ALGORITHMPRACTICE_TREE_H
#define ALGORITHMPRACTICE_TREE_H
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <stack>
#include <set>
#include <algorithm>
#include <vector>
#include <string>
#include <stack>
#include <queue>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


// #105 Construct Binary Tree from Inorder and Preoder Traversal
// Note: The pre-order traverse must visit the root of tree first, and left subtree then right subtree
//       The in order traverse must visit the left subtree first, and root then
TreeNode* buildTreeHelper(const vector<int>& preorder, int pLeft, int pRight, const vector<int>& inorder, int iLeft, int iRight){
    if (pLeft > pRight || iLeft > iRight) return nullptr;
    int i = 0, root_val = preorder[pLeft];
    for (i = iLeft; i <= iRight; ++i) {
        if (inorder[i] == root_val) break;
    }
    auto* root = new TreeNode(root_val);
    root->left = buildTreeHelper(preorder, pLeft + 1, pLeft + i - iLeft, inorder, iLeft, i - 1);
    root->right = buildTreeHelper(preorder, pLeft + i - iLeft + 1, pRight, inorder, i + 1, iRight);
    return root;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    return buildTreeHelper(preorder, 0, (int)preorder.size() - 1, inorder, 0, (int)inorder.size() - 1);
}

// #106 Construct Binary Tree from Inorder and Postorder Traversal
TreeNode* buildTreeHelperInPost(const vector<int>& inorder, int iLeft, int iRight, const vector<int>& postorder, int pLeft, int pRight){
    if (iLeft > iRight || pLeft > pRight) return nullptr;
    int i = 0, root_val = postorder[pRight];
    auto* root = new TreeNode(root_val, nullptr, nullptr);
    for (i = iLeft; i <= iRight; ++i){
        if (inorder[i] == root_val) break;
    }
    root->left = buildTreeHelperInPost(inorder, iLeft, i - 1, postorder, pLeft, pLeft + i - iLeft - 1);
    root->right = buildTreeHelperInPost(inorder, i + 1, iRight, postorder, pLeft + i - iLeft, pRight - 1);
    return root;
}
TreeNode* buildTreeInPost(vector<int>& inorder, vector<int>& postorder) {
    return buildTreeHelperInPost(inorder, 0, (int)inorder.size() - 1, postorder, 0, (int) postorder.size() - 1);
}


#endif //ALGORITHMPRACTICE_TREE_H
