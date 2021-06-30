//
// Created by Xiaoru_Zhu on 2020/6/2.
//

#ifndef ALGORITHMPRACTICE_DATA_STRUCTURES_H
#define ALGORITHMPRACTICE_DATA_STRUCTURES_H

#include <cstdio>
#include <vector>
#include <queue>
#include <functional>
#include <unordered_set>

// Segment Tree (int) (sum)
const int MAX_SEG_TREE_LEN = 1000;

class SegTree {
public:
    explicit SegTree(const std::vector<int>& arr, const std::function<int(int, int)>& func) {
        this->arr.assign(arr.begin(), arr.end());
        this->tree.assign(MAX_SEG_TREE_LEN, 0);
        this->func = func;
        seg_tree_build(0, 0, (int) this->arr.size() -1);
    }

    void update(int idx, int val) {
        if (idx >= (int) this->arr.size()) return;
        this->arr[idx] = val;
        seg_tree_update(0, 0, (int) this->arr.size() - 1, idx, val);
    }

    // get range sum
    int query(const int L, const int R) {
        if (L > R || L < 0 || R >= (int) this->arr.size()) return 0x3f3f3f3f;
        return seg_tree_query(0, 0, (int) this->arr.size() - 1, L, R);
    }

private:
    std::vector<int> arr;
    std::vector<int> tree;
    std::function<int(int, int)> func;
    void seg_tree_build(int node, int start, int end) {

        if(start == end){
            tree[node] = arr[start]; // start or end
            return;
        }

        int mid = (start + end) / 2; // start + (end - start) / 2

        // Like the Heap Sort tree
        int left_child = 2 * node + 1;
        int right_child = 2 * node + 2;

        // Recursion
        seg_tree_build(left_child, start, mid);
        seg_tree_build(right_child, mid + 1, end);
        tree[node] = tree[left_child] + tree[right_child];
        // tree[node] = this->func(tree[left_child], tree[right_child]);
    }
    void seg_tree_update(int node, int start, int end, int idx, int val){
        if (start == end){
            // value of start and end must equal to idx
            arr[idx] = val;
            tree[node] = val;
            return;
        }
        int mid = (start + end) / 2;
        int left_child = 2 * node + 1;
        int right_child = 2 * node + 2;

        // Recursion for only one path of the segment tree
        if(idx >= start && idx <= mid){
            seg_tree_update(left_child, start, mid, idx, val);
        } else {
            seg_tree_update(right_child, mid + 1, end, idx, val);
        }
        tree[node] = tree[left_child] + tree[right_child];
        // tree[node] = this->func(tree[left_child], tree[right_child]);
    }
    int seg_tree_query(int node, int start, int end, const int L, const int R){

        // The querying range is out of node range, return 0
        if (L > end || R < start) return 0;
        // Return tree[node] if:
        //  1. The current node refers to a single element in the original array
        //  OR
        //  2. The range of current node is completely included in the querying range
        if (start == end || (L <= start && end <= R)) return tree[node];

        int mid = (start + end) / 2; // (end - start) / 2 + start => to prevent overflow
        int left_child = 2 * node + 1;
        int right_child = 2 * node + 2;

        int sum_left = seg_tree_query(left_child, start, mid, L, R);
        int sum_right = seg_tree_query(right_child, mid + 1, end, L, R);
        return sum_left + sum_right;
        // return this->func(sum_left, sum_right);
    }
};
/*
 * Param1 - arr: The original array
 * Param2 - tree: The segment tree
 * Param3 - node: The current root node of the segment subtree
 * Param4 - start: The left side of range for current node in the original array
 * Param5 - end: The right side of range for current node in the original array
 * */
void seg_tree_build(int arr[], int tree[], int node, int start, int end) {

    if(start == end){
        tree[node] = arr[start]; // start or end
        return;
    }

    int mid = (start + end) / 2; // start + (end - start) / 2

    // Like the Heap Sort tree
    int left_child = 2 * node + 1;
    int right_child = 2 * node + 2;

    // Recursion
    seg_tree_build(arr, tree, left_child, start, mid);
    seg_tree_build(arr, tree, right_child, mid + 1, end);
    tree[node] = tree[left_child] + tree[right_child];
}
/*
 * Param1 - arr: The original array
 * Param2 - tree: The segment tree
 * Param3 - node: The current root node of the segment subtree
 * Param4 - start: The left side of range for current node in the original array
 * Param5 - end: The right side of range for current node in the original array
 * Param6 - idx: The idx of updating element in the original array
 * Param7 - val: The value of the updating element
 * */
void seg_tree_update(int arr[], int tree[], int node, int start, int end, int idx, int val){
    if (start == end){
        // value of start and end must equal to idx
        arr[idx] = val;
        tree[node] = val;
        return;
    }
    int mid = (start + end) / 2;
    int left_child = 2 * node + 1;
    int right_child = 2 * node + 2;

    // Recursion for only one path of the segment tree
    if(idx >= start && idx <= mid){
        seg_tree_update(arr, tree, left_child, start, mid, idx, val);
    } else {
        seg_tree_update(arr, tree, right_child, mid + 1, end, idx, val);
    }
    tree[node] = tree[left_child] + tree[right_child];
}
/*
 * Param1 - arr: The original array
 * Param2 - tree: The segment tree
 * Param3 - node: The current root node of the segment subtree
 * Param4 - start: The left side of range for current node in the original array
 * Param5 - end: The right side of range for current node in the original array
 * Param6 - L: The left side of querying range
 * Param7 - R: The right side of querying range
 * */
int seg_tree_query(int arr[], int tree[], int node, int start, int end, const int L, const int R){

    // The querying range is out of node range, return 0
    if (L > end || R < start) return 0;
    // Return tree[node] if:
    //  1. The current node refers to a single element in the original array
    //  OR
    //  2. The range of current node is completely included in the querying range
    if (start == end || (L <= start && end <= R)) return tree[node];

    int mid = (start + end) / 2; // (end - start) / 2 + start => to prevent overflow
    int left_child = 2 * node + 1;
    int right_child = 2 * node + 2;

    int sum_left = seg_tree_query(arr, tree, left_child, start, mid, L, R);
    int sum_right = seg_tree_query(arr, tree, right_child, mid + 1, end, L, R);
    return sum_left + sum_right;
}

void seg_tree_test(){
    int arr[] = {1, 3, 5, 7 ,9, 11};
    int size = 6;
    int tree[MAX_SEG_TREE_LEN] = {0};

    seg_tree_build(arr, tree, 0, 0, size -1);
    seg_tree_query(arr, tree, 0, 0, size - 1, 1, 4);

    seg_tree_update(arr, tree, 0, 0, size - 1, 4, 19);
    seg_tree_query(arr, tree, 0, 0, size - 1, 1, 4);
}

// Disjoint Set
void union_find_init(std::vector<int>& parent, std::vector<int>& rank){
    // Size is the number of the Vertices of Graph
    for(int i = 0; i < (int)parent.size(); ++i){
        parent[i] = -1;
        rank[i] = 0;
    }
}
int union_find_find_root(int x, const std::vector<int>& parent){
    int x_root = x;
    while (parent[x_root] != -1){
        x_root = parent[x_root];
    }
    return x_root;
}
/*
 * Return:
 *  1 - circle exists
 *  0 - current x, y is not in a circle
 * */
int union_find_union_vertices(int x, int y, std::vector<int>& parent, std::vector<int>& rank){
    int x_root = union_find_find_root(x, parent);
    int y_root = union_find_find_root(y, parent);
    if (x_root == y_root){
        return 1;
    } else {

        if (rank[x_root] > rank[y_root]){
            parent[y_root] = x_root;
        } else if (rank[x_root] < rank[y_root]){
            parent[x_root] = y_root;
        } else {
            parent[x_root] = y_root;
            rank[y_root]++;
        }
        return 0;
    }
}
// Check if there are circles in a graph
/*
 * Return:
 *  true - circle exists
 *  false - circle does not exist
 * */
bool union_find_circle_test(){
    std::vector<int> parent(6, 0);
    std::vector<int> rank(6, 0);
    // Init graph
    std::vector<std::pair<int, int>> edges = {
            {0, 1}, {1, 2}, {1, 3}, {3, 4}, {2, 5}
    };
    union_find_init(parent, rank);

    for(auto & edge : edges){
        int x = edge.first;
        int y = edge.second;
        if (union_find_union_vertices(x, y, parent, rank)) {
            return true;
        }
    }
    return false;
}


// Trie (Prefix Tree)
const int ALPHABET_SIZE = 26;
typedef struct trie_node {
    int count;   // 记录该节点代表的单词的个数
    trie_node* children[ALPHABET_SIZE]; // 各个子节点
}* trie;
trie_node* create_trie_node() {
    trie_node* pNode = new trie_node();
    pNode->count = 0;
    for (int i = 0; i < ALPHABET_SIZE; ++i)
        pNode->children[i] = nullptr;
    return pNode;
}
void trie_insert(trie root, char* key) {
    trie_node* node = root;
    char* p = key;
    while (*p) {
        if (node->children[*p - 'a'] == NULL) {
            node->children[*p - 'a'] = create_trie_node();
        }
        node = node->children[*p - 'a'];
        ++p;
    }
    node->count += 1;
}
/**
 * Return:
 *  0: not exist
 *  Other Number: appearing times
 */
int trie_search(trie root, const char* key) {
    trie_node* node = root;
    const char* p = key;
    while (*p && node != nullptr) {
        node = node->children[*p - 'a'];
        ++p;
    }

    if (node == nullptr)
        return 0;
    else
        return node->count;
}
int trie_test() {
    // Keywords set
    char keys[][8] = {"the", "a", "there", "answer", "any", "by", "bye", "their"};
    trie root = create_trie_node();

    // Build trie (Prefix tree)
    for (auto & key : keys)
        trie_insert(root, key);

    // Search string test
    char s[][32] = {"Present in trie", "Not present in trie"};
    printf("%s --- %s\n", "the", trie_search(root, (char *)"the") > 0 ? s[0] : s[1]);
    printf("%s --- %s\n", "these", trie_search(root, (char *)"these") > 0 ? s[0] : s[1]);
    printf("%s --- %s\n", "their", trie_search(root, (char *)"their") > 0 ? s[0] : s[1]);
    printf("%s --- %s\n", "thaw", trie_search(root, (char *)"thaw") > 0 ? s[0] : s[1]);

    return 0;
}

std::function<bool(std::pair<int, int>, std::pair<int, int>)> cmp = [](const std::pair<int, int> a, const std::pair<int, int> b) -> bool{ return a.second > b.second; };
// Priority  queue with lazy deletion (int, min)
class MinHeap {
public:
    MinHeap() {
        deleteSet = new std::unordered_set<int>();
        // std::function<bool(const int, const int)> cmp = [](const int a, const int b) -> bool{ return a > b; };
        // std::priority_queue<int, std::vector<int>, std::function<bool(const int, const int)> > minHeap(cmp);
        minHeap = new std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, decltype(cmp)> (cmp);
    }

    void push(int idx, int val) {
        minHeap->push({idx, val});
    }

    std::pair<int, int> top() {
        lazyDeletion();
        return minHeap->top();
    }

    void pop() {
        lazyDeletion();
        minHeap->pop();
    }

    void removeByIdx(int idx){
        deleteSet->insert(idx);
    }

    bool isEmpty() {
        return minHeap->empty();
    }

private:
    // pair.first: index in original array; pair.second: value in original array
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, decltype(cmp) > *minHeap;
    // int: index in original array
    std::unordered_set<int> *deleteSet;

    void lazyDeletion() {
        while (!minHeap->empty() && deleteSet->count(minHeap->top().first)) {
            auto pair = minHeap->top();
            minHeap->pop();
            deleteSet->erase(pair.first);
        }
    }
};


#endif //ALGORITHMPRACTICE_DATA_STRUCTURES_H
