//
// Created by Xiaoru_Zhu on 2020/6/2.
//

#ifndef ALGORITHMPRACTICE_DATA_STRUCTURES_H
#define ALGORITHMPRACTICE_DATA_STRUCTURES_H


// Segment Tree
const int MAX_SEG_TREE_LEN = 1000;
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

    int mid = (start + end) / 2;

    // Like the Heap Sort tree
    int left_child = 2 * node + 1;
    int right_child = 2 * node + 1;

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
    int right_child = 2 * node + 1;

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
int seg_tree_query(int arr[], int tree[], int node, int start, int end, int L, int R){

    // The querying range is out of node range, return 0
    if (L > end || R < start) return 0;
    // Return tree[node] if:
    //  1. The current node refers to a single element in the original array
    //  OR
    //  2. The range of current node is completely included in the querying range
    if (start == end || (L <= start && end <= R)) return tree[node];

    int mid = (start + end) / 2; // (end - start) / 2 + start => to prevent overflow
    int left_child = 2 * node + 1;
    int right_child = 2 * node + 1;

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


#endif //ALGORITHMPRACTICE_DATA_STRUCTURES_H
