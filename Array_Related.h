//
// Created by Xiaoru_Zhu on 2019/11/6.
//
#ifndef ALGORITHMPRACTICE_ARRAY_RELATED_H
#define ALGORITHMPRACTICE_ARRAY_RELATED_H

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
#include <functional>
#include "Data_Structures.h"

using namespace std;
const int INT_MAX = INT32_MAX;
const int INT_MIN = INT32_MIN;
// #1
/*
 * Given an array of integers, return indices of the two numbers such that they add up to a specific target.
 * You may assume that each input would have exactly one solution, and you may not use the same element twice.
 * Example:
 *  Given nums = [2, 7, 11, 15], target = 9,
 *  Because nums[0] + nums[1] = 2 + 7 = 9,
 *  return [0, 1].
 */
static vector<int> twoSum(vector<int>& nums, int target) {
    vector<int> ret_vec;
    unordered_map<int, int> arr_map;
    //  int idx1 = -1, idx2 = -1;
    for(int i = 0; i != nums.size(); ++i)
        arr_map.emplace(nums[i], i);

    for(int i = 0; i != nums.size(); ++i){
        int t = target - nums[i];
        if(arr_map.count(t) && arr_map[t] != i){
            ret_vec.emplace_back(i);
            ret_vec.emplace_back(arr_map[t]);
            break;
        }
    }
    return ret_vec;
}

// #2
/*
You are given two non-empty linked lists representing two non-negative integers.
The digits are stored in reverse order and each of their nodes contain a single digit.
Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Example:
    Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    Output: 7 -> 0 -> 8
    Explanation: 342 + 465 = 807.
 */
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};
static ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* ret_list = new ListNode(-1);
    ListNode* ret_curr = ret_list;

    int digit = 0;
    bool flag = false;
    ListNode* curr_l1 = l1;
    ListNode* curr_l2 = l2;
    while (curr_l1 || curr_l2 || flag){
        int digit_value = 0;
        if(curr_l1) digit_value += curr_l1->val;
        if(curr_l2) digit_value += curr_l2->val;
        if(flag) ++digit_value;

        if(digit_value >= 10){
            digit_value -= 10;
            flag = true;
        }else{
            flag = false;
        }

        // ret_curr->val = digit_value;
        ret_curr->next = new ListNode(digit_value);
        ret_curr = ret_curr->next;

        if(curr_l1) curr_l1 = curr_l1->next;
        if(curr_l2) curr_l2 = curr_l2->next;
    }
    return ret_list->next;
}


// #4
/*
There are two sorted arrays nums1 and nums2 of size m and n respectively.
Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
You may assume nums1 and nums2 cannot be both empty.

    Example 1:
    nums1 = [1, 3]
    nums2 = [2]

    The median is 2.0

    Example 2:
    nums1 = [1, 2]
    nums2 = [3, 4]

    The median is (2 + 3)/2 = 2.5
*/
static int find_no_k_element(vector<int>& nums1, int i, vector<int>& nums2, int j, int k);
static double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int m = nums1.size(), n = nums2.size(), left = (m + n + 1) / 2, right = (m + n + 2) / 2;
    return (find_no_k_element(nums1, 0, nums2, 0, left) +
            find_no_k_element(nums1, 0, nums2, 0, right))
            / 2.0;
}
static int find_no_k_element(vector<int>& nums1, int i, vector<int>& nums2, int j, int k){
    if(i >= nums1.size()) return nums2[j + k -1];
    if(j >= nums2.size()) return nums1[i + k -1];
    if(k == 1) return min(nums1[i], nums2[j]);
    int mid1 = ( (i + k / 2 - 1) < nums1.size() ? nums1[i + k / 2 - 1] : INT_MAX);
    int mid2 = ( (j + k / 2 - 1) < nums2.size() ? nums2[j + k / 2 - 1] : INT_MAX);
    if(mid1 < mid2)
        return find_no_k_element(nums1, i + k / 2, nums2, j, k - k / 2);
    else
        return find_no_k_element(nums1, i, nums2, j + k / 2, k - k / 2);
}


// #7
// reverse INT32 Number
static int reverse(int x) {
    bool is_positive = x > 0;
    long long abs_x = x; // cannot be int here because the INT_MIN == -2 ^ 32 -1 which does not have opposite number
    if(!is_positive) abs_x *= -1; // abs_x = -x;
    long long ret_num = 0;
    while(abs_x){
        ret_num = ret_num * 10 + abs_x % 10;
        abs_x /= 10;
    }
    if(ret_num > INT_MAX) return 0;
    if(!is_positive) ret_num = -ret_num;
    return (int)ret_num;
}

// #9
// Palindrome number
static bool isPalindrome(int x) {
    int div = 1, tmp = x;
    while (tmp >= 10) {
        tmp /= 10;
        div *= 10;
    }
    // while (x / div >= 10) div *= 10;
    // if(div == 1) return true;
    // if(div == 10) return x % 11 == 0;
    while (x > 0) {
        if (x / div != x % 10) return false;
        x = (x % div) / 10;
        div /= 10;
    }
    return true;
}


// #11
//
/*
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai).
n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0).
Find two lines, which together with x-axis forms a container, such that the container contains the most water.
*/
static int maxArea(vector<int>& height) {
    int max_area = 0, i = 0, j = (int)height.size() - 1; // i, j is the left and right pointer of the given array
    while(i < j){
        max_area = std::max(max_area, std::min(height[i], height[j]) * (j - i));
        height[i] < height[j] ? ++i : --j;
    }
    return max_area;
}

// #12
// convert int number ( [1, 4000) ) into Roman
static string intToRoman(int num) {
    string ret_str; int tmp;

    tmp = num / 1000; // in the test case tmp must less than 4
    if(tmp) for(int i = 0; i < tmp; ++i) ret_str += 'M';
    num %= 1000;

    tmp = num / 100;
    if(tmp == 9) ret_str += "CM";
    else if(tmp > 4 && tmp < 9){
        ret_str += 'D';
        for (int i = 6; i <= tmp; ++i) ret_str += 'C';
    }
    else if(tmp == 4) ret_str += "CD";
    else if(tmp < 4)
        for (int i = 1; i <= tmp; ++i) ret_str += 'C';
    num %= 100;

    tmp = num / 10;
    if(tmp == 9) ret_str += "XC";
    else if(tmp > 4 && tmp < 9){
        ret_str += 'L';
        for (int i = 6; i <= tmp; ++i) ret_str += 'X';
    }
    else if(tmp == 4) ret_str += "XL";
    else if(tmp < 4)
        for (int i = 1; i <= tmp; ++i) ret_str += 'X';
    num %= 10;

    tmp = num;
    if(tmp == 9) ret_str += "IX";
    else if(tmp > 4 && tmp < 9){
        ret_str += 'V';
        for (int i = 6; i <= tmp; ++i) ret_str += 'I';
    }
    else if(tmp == 4) ret_str += "IV";
    else if(tmp < 4)
        for (int i = 1; i <= tmp; ++i) ret_str += 'I';

    return ret_str;
}

// #13
// convert Roman into int number ( [1, 4000) )
// check if the corresponding value of the current char is less than the next,
static int romanToInt(string s) {
    int ret_num = 0, pos = 0;
    unordered_map<char, int> roman_map{{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50},
                                       {'C', 100}, {'D', 500}, {'M', 1000}};
    while(pos < s.size()){
        if(pos == s.size() - 1 || roman_map[s[pos]] >= roman_map[s[pos + 1]]) ret_num += roman_map[s[pos]];
        else ret_num -= roman_map[s[pos]];
        ++pos;
    }
    return ret_num;
}

// #15
// 3 sum
// in order to avoid duplicate tuples, sort the array firstly, and use 2 pointers i,j (front and rear)
// to improve the performance
static vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> ret_vec;
    vector<int> tmp_vec;
    std::sort(nums.begin(), nums.end());
    if (nums.empty() || nums.back() < 0 || nums.front() > 0) return {};
    for (int k  = 0; k < (int)nums.size() - 2; ++k) {
        if(nums[k] > 0) break;
        if (k > 0 && nums[k - 1] == nums[k]) continue;
        int opposite = 0 - nums[k], i = k + 1, j = (int)nums.size() - 1;
        while(i < j){
            if(nums[i] + nums[j] == opposite){
                tmp_vec = {nums[k], nums[i], nums[j]};
                ret_vec.emplace_back(tmp_vec);
                //ret_vec.push_back({nums[k], nums[i], nums[j]});
                while(i < j && nums[i] == nums[i + 1]) ++i; // i<j: should check if the pointer surpass the boundary
                while(i < j && nums[j] == nums[j - 1]) --j;
                ++i; --j;
            }
            else if (nums[i] + nums[j] < opposite) ++i;
            else --j;
        }

    }
    return ret_vec;
}

// #16
// threeSumClosest
static int threeSumClosest(vector<int>& nums, int target) {
    int diff = INT_MAX, closest = INT_MAX;
    std::sort(nums.begin(), nums.end());
    // diff = std::abs(target - closest);
    for(int k = 0; k < int(nums.size() - 2); ++k){
        int i = k + 1, j = (int)nums.size() - 1;
        while(i < j){
            int sum = nums[k] + nums[i] + nums[j];
            if(diff > std::abs(target - sum)){
                diff = std::abs(target - sum);
                closest = sum;
            }
            if(sum < target) ++i;
            else --j;
        }
    }

    return closest;
}

// #18
// four sum prob
static vector<vector<int>> fourSum(vector<int>& nums, int target) {
    vector<vector<int>> ret_vec;
    std::sort(nums.begin(), nums.end());
    for(int i = 0; i < (int)nums.size() - 3; ++i){
        if(i > 0 && nums[i - 1] == nums[i]) continue;
        for(int j = i + 1; j < (int)nums.size() - 2; ++j){
            if(i + 1 < j && nums[j-1] == nums[j]) continue;
            int left = j + 1, right = (int)nums.size() - 1;
            while(left < right){
                int tmp_sum = nums[i] + nums[j] + nums[left] + nums[right];
                if(tmp_sum == target) {
                    vector<int> tmp{nums[i], nums[j], nums[left], nums[right]};
                    ret_vec.emplace_back(tmp);
                    while(left < right && nums[left] == nums[left + 1]) ++left;
                    while(left < right && nums[right] == nums[right - 1]) --right;
                    ++left; --right;
                } else if(tmp_sum > target) --right;
                else ++left;
            }
        }
    }
    return ret_vec;
}

// #19
// Remove Nth Node From End of List
static ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode *prvs = head, *curr = head, *tmp = nullptr;
    for(int i = 0; i < n; ++i) curr = curr->next;
    if(curr == nullptr) return head->next;
    while(curr->next != nullptr){
        curr = curr->next;
        prvs = prvs->next;
    }
    //tmp = prvs->next;
    //delete tmp;
    prvs->next = prvs->next->next;
    return head;
}

// #20
// Valid Parentheses
static bool isValid(string s) {
    std::stack<char> parentheses;
    std::unordered_map<char, char> relation_map;
    relation_map['['] = ']'; relation_map['('] = ')'; relation_map['{'] = '}';
    for(char ch : s){
        if(relation_map.count(ch)) parentheses.push(ch);
        else{
            if(parentheses.empty()) return false; // case : "]}}"
            if(ch != relation_map[parentheses.top()]) return false;
            parentheses.pop();
        }
    }
    return parentheses.empty();
}

// #21
// Merge two sorted linked list
static ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    if(!l1) return l2;
    if(!l2) return l1;

    if(l1->val <= l2->val){
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    }else{
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    }
}

// #22 *************
// Generate Parentheses
static void generateParenthesisDFS(int left, int right, const string& out, vector<string> &vec);
static vector<string> generateParenthesis(int n) {
    std::vector<string> ret_vec;
    generateParenthesisDFS(n, n, "", ret_vec);
    return ret_vec;
}
static void generateParenthesisDFS(int left, int right, const string& out, vector<string> &vec) {
    if(left > right) return;
    if(left == 0 && right == 0) vec.emplace_back(out);
    else{
        if(left > 0) generateParenthesisDFS(left - 1, right, out + '(', vec);
        if(right > 0) generateParenthesisDFS(left, right - 1, out + ')', vec);
    }
}

// #23 **********
// 多路归并链表 - by using priority queue
static ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](const ListNode* a, const ListNode* b) -> bool{ return a->val > b->val; };
    // '>' in the lambda fuc body means the smaller guy has the higher priority!
    std::priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
    //std::vector<ListNode*>::iterator it = lists.begin();
    for(ListNode* node : lists){
        if(node) pq.push(node);
    }
    ListNode *dummy = new ListNode(-1), *cur = dummy;
    while(!pq.empty()){
        cur->next = pq.top(); pq.pop();
        cur = cur->next;
        if(cur->next) pq.push(cur->next); // if cur->next is not null
    }
    return dummy->next;
}

// #24
// swap pairs in linked list
static ListNode* swapPairs(ListNode* head) {
    auto *dummy = new ListNode(-1); dummy->next = head;
    auto *pre = dummy;
    while(pre->next && pre->next->next){
        auto* tmp = pre->next->next;
        pre->next->next = tmp->next;
        tmp->next = pre->next;
        pre->next = tmp;

        pre = pre->next->next;
    }
    return dummy->next;
}

// #25 ***********
// Reverse Nodes in k-Group
static void reverseK(ListNode* front, ListNode* back);
static ListNode* reverseKGroup(ListNode* head, int k) {
    if(k == 1) return head;
    ListNode *dummy = new ListNode(-1), *curr = dummy;
    dummy->next = head;
    for (int i = 0; i < k; ++i) {
        if (!curr->next) return head;
        curr = curr->next;
    }
    reverseK(head, curr);
    head->next = reverseKGroup(head->next, k);
    delete dummy;
    return curr;
}
static void reverseK(ListNode* front, ListNode* back){
    ListNode *end = back->next, *pre = back->next, *curr = front, *nxt = nullptr;
    while(curr != end){
        nxt = curr->next;
        curr->next = pre;
        pre = curr;
        curr = nxt;
    }
}

// #26
// remove dupliacted elements in a sorted array
static int removeDuplicates(vector<int>& nums) {
    int curr = nums[0];
    vector<int>::iterator pos = nums.begin();
    ++pos;
    while(pos != nums.end()){
        if(*pos == curr){
            pos = nums.erase(pos);
        }else{
            curr = *pos; ++pos;
        }
    }
    return (int)nums.size();
}

// #31
// Next Permutation
static void nextPermutation(vector<int> &num) {
    int i, j, n = num.size();
    for (i = n - 2; i >= 0; --i) {
        if (num[i + 1] > num[i]) {
            for (j = n - 1; j > i; --j) {
                if (num[j] > num[i]) break;
            }
            std::swap(num[i], num[j]);
            std::reverse(num.begin() + i + 1, num.end());
            return;
        }
    }
    std::reverse(num.begin(), num.end());
}

// #32
static int longestValidParentheses(string s) {
    std::stack<int> stk;
    int start = 0, ret_num = 0;;
    for(int i = 0; i < (int)s.length(); ++i){
        if(s[i] == '(') {
            stk.push(i);
        }else {
            if(stk.empty()) start = i + 1;
            else{
                stk.pop();
                ret_num = stk.empty() ? std::max(ret_num, i - start + 1) : std::max(ret_num, i - stk.top());
            }
        }
    }
    return ret_num;
}

// #46
// Permutations I
   static void dfs_perm(vector<vector<int>>& ret, int level, vector<int>& nums, vector<int>& seq, bool visited[]);
       static vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> ret;
    vector<int> seq;
    int num_size = (int)nums.size();
    bool visited[200] = {false};
    dfs_perm(ret, 0, nums, seq, visited);
    return ret;
}
static void dfs_perm(vector<vector<int>>& ret, int level, vector<int>& nums, vector<int>& seq, bool visited[]){
    int arr_size = (int)nums.size();
    if (level == arr_size) {
        ret.push_back(seq);
        return;
    }
    for(int i = 0; i < arr_size; ++i){
        if (visited[i]) continue;
        visited[i] = true;
        seq.push_back(nums[i]);
        // vector<int>::iterator it = seq.begin();
        dfs_perm(ret, level + 1, nums, seq, visited);
        seq.pop_back();
        visited[i] = false;
    }
}

// #47
// Permutations II
// https://www.youtube.com/watch?v=imLl2s9Ujis
static void dfs_perm2(vector<int>& nums, int level, vector<int>& visited, vector<int>& seq, vector<vector<int>>& res);
    static vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> res;
    vector<int> seq, visited(nums.size(), 0);
    sort(nums.begin(), nums.end());
    dfs_perm2(nums, 0, visited, seq, res);
    return res;
}
static void dfs_perm2(vector<int>& nums, int level, vector<int>& visited, vector<int>& seq, vector<vector<int>>& res) {
    if (level >= nums.size()) {res.push_back(seq); return;}
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (visited[i] == 1) continue;
        // 在递归函数中要判断前面一个数和当前的数是否相等，如果相等，且其对应的 visited 中的值为1，当前的数字才能使用，否则需要跳过，这样就不会产生重复排列了
        if (i > 0 && nums[i] == nums[i - 1] && visited[i - 1] == 0) continue;
        visited[i] = 1;
        seq.push_back(nums[i]);
        dfs_perm2(nums, level + 1, visited, seq, res);
        seq.pop_back();
        visited[i] = 0;
    }
}


// #33
// Binary search in totated array
static int searchBin(vector<int>& nums, int target) {
    int ret_idx = -1, l = 0, r = (int)nums.size() - 1;
    while(l <= r){
        int m = (r + l) / 2;
        if(target == nums[m]) return m;
        if(nums[m] < nums[r]){
            if(nums[m] < target && target <= nums[r]) l = m + 1;
            else r = m - 1;
        }else{
            if(nums[l] <= target && target < nums[m]) r = m -1;
            else l = m + 1;
        }
    }

    return ret_idx;
}

// #34
// O(logN) time binary search target number range (return {left_idx, right_idx})
static int first_greater_equal(vector<int>& nums, int target);
static vector<int> searchRange(vector<int>& nums, int target) {
    if (nums.empty()) return {-1, -1};
    int fisrt_target_idx = first_greater_equal(nums, target);
    if(fisrt_target_idx == (int)nums.size() || nums[fisrt_target_idx] != target ){ // the sequence is important.
        return {-1, -1};
    }else{
        return {fisrt_target_idx, first_greater_equal(nums, target + 1) - 1};
    }

}
static int first_greater_equal(vector<int>& nums, int target) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = (left + right) / 2;
        if (nums[mid] < target) left = mid + 1;
        else right = mid;
    }
    return right;
}

// #36
// Determine if a 9x9 Sudoku board is valid
static bool isValidSudoku(vector<vector<char>>& board) {
    // bool** cell_flag_ptr = new bool*[9];
    // for(int i = 0; i < 9; ++i) cell_flag_ptr[i] = new bool[9];
    vector<vector<bool>> cell_flag(9, vector<bool>(9, false));
    vector<vector<bool>> row_flag(9, vector<bool>(9, false));
    vector<vector<bool>> clm_flag(9, vector<bool>(9, false));

    for(int i = 0; i < 9; ++i)
        for(int j = 0; j < 9; ++j){
            if(board[i][j] == '.') continue;
            else{
                int value = board[i][j] - '1'; // 1-9 ==> 0-8 idx
                if(cell_flag[3 * (i / 3) + (j / 3)][value] || row_flag[i][value] || clm_flag[j][value] ) return false;
                cell_flag[3 * (i / 3) + (j / 3)][value] = true;
                row_flag[i][value] = true;
                clm_flag[j][value] = true;
            }
        }
    return true;
}

// #37
// solution for Sudoku
static bool is_valid(vector<vector<char>>& board, int i, int j, char val);
static bool helper(vector<vector<char>>& board, int i, int j);
static void solveSudoku(vector<vector<char>>& board) {
    helper(board, 0, 0);
}
static bool helper(vector<vector<char>>& board, int i, int j) {
    if(i == 9) return true;
    if(j == 9) return helper(board, i + 1, 0);
    if(board[i][j] != '.') return helper(board, i, j + 1);
    for(char ch = '1'; ch <= '9'; ++ch){
        if(!is_valid(board, i, j, ch)) continue;
        board[i][j] = ch;
        if(helper(board, i, j + 1)) return true;
        board[i][j] = '.';
    }
    return false;

}
static bool is_valid(vector<vector<char>>& board, int i, int j, char val) {
    for (int x = 0; x < 9; ++x) {
        if (board[x][j] == val) return false;
    }
    for (int y = 0; y < 9; ++y) {
        if (board[i][y] == val) return false;
    }
    int row = i - i % 3, col = j - j % 3;
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 3; ++y) {
            if (board[x + row][y + col] == val) return false;
        }
    }
    return true;
}

// #39
// Combination Sum 1 (repeatable element)
static void dfs_comb_sum(vector<int>& candidates, int target, int start, vector<int>& tmp_vec, vector<vector<int>>& rst);
    static vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> rst;
    vector<int> tmp_vec;
    dfs_comb_sum(candidates, target, 0, tmp_vec, rst);
    return rst;
}

static void dfs_comb_sum(vector<int>& candidates, int target, int start, vector<int>& tmp_vec, vector<vector<int>>& rst){
    if(target < 0) return;
    if(target == 0) {rst.push_back(tmp_vec); return;}

    for(int i = start; i < (int)candidates.size(); ++i){
        tmp_vec.push_back(candidates[i]);
        dfs_comb_sum(candidates, target - candidates[i], i, tmp_vec, rst);
        tmp_vec.pop_back();
    }
}

// #40
// Combination Sum 2 (unrepeatable element)
static void comb2_dfs(vector<vector<int>>& rst, vector<int>& candidates, vector<int>& vec_tmp, int start, int target);
    static vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<vector<int>> rst;
    vector<int> vec_tmp;
    sort(candidates.begin(), candidates.end()); // sort for preventing duplicate combinations
    comb2_dfs(rst, candidates, vec_tmp, 0, target);
    return rst;
}

static void comb2_dfs(vector<vector<int>>& rst, vector<int>& candidates, vector<int>& vec_tmp, int start, int target){
    if(target < 0) return;
    if(target == 0) {rst.push_back(vec_tmp); return;}
    for(int i = start; i < (int)candidates.size(); ++i){
        // at the same for loop, to prevent duplicate combinations
        if(i > start && candidates[i - 1] == candidates[i]) continue;

        vec_tmp.push_back(candidates[start]);
        comb2_dfs(rst, candidates, vec_tmp, i + 1, target-candidates[start]); // i+1
        vec_tmp.pop_back();
    }
}

// #41 hard (time limit: O(n), space limit: O(1))
// Utilize Bucket sort Algorithm
// [3, 4, -1 , 1]
// [0, 0, 0, 0] ==> [1, -1, 3, 4] (Bucket)
// [1, 2, 3, 4] (index, not exist)
static int firstMissingPositive(vector<int>& nums) {
    int n = nums.size();
    for (int i = 0; i < n; ++i) {
        // Important! element nums[i] should go to the position "nums[i] - 1" (idx starts from 0)
        while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
            std::swap(nums[i], nums[nums[i] - 1]);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (nums[i] != i + 1) return i + 1;
    }
    return n + 1;
}

// #42 ********* Trapping Rain Water
// Use left and right pointer
// The current capacity depends on the difference between the min value of the left/right barrier(pointer)
// and the current barrier
static int trap(vector<int>& height) {
    int l = 0, r = (int)height.size(), l_max = 0, r_max = 0, ret_capacity = 0;
    while (l < r){
        if(height[l] < height[r]){
            l_max = std::max(l_max, height[l]);
            ret_capacity += l_max - height[l];
            ++l;
        }else{
            r_max = std::max(r_max, height[r]);
            ret_capacity += r_max - height[r];
            --r;
        }
    }
    return ret_capacity;
}

// #48 rotate 2d array
static void rotate(vector<vector<int>> & matrix){
    int n = (int) matrix.size();
    for(int i = 0; i < n / 2; ++i)
        for(int j = i; j < n - 1 - i; ++j){
            int tmp_num = matrix[i][j];
            matrix[i][j] = matrix[n - 1 - j][i];
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
            matrix[j][n - 1 - i] = tmp_num;
        }
}

// #56 Merge Intervals
// 1. check if the vector is empty first
// 2. check if the range of current element contains the next one's
static vector<vector<int>> merge_intervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};

    // std::function<bool(const vector<int>& a, const vector<int>& b)> f =  [](const vector<int>& a, const vector<int>& b) -> bool { return a[0] < b[0]; };
    auto cmp = [](const vector<int>& a, const vector<int>& b) -> bool { return a[0] < b[0]; };
    std::sort(intervals.begin(), intervals.end(), cmp);

    vector<vector<int>> ret;
    ret.reserve(intervals.size());
    ret.push_back(intervals[0]);
    //vector<int> tmp(intervals[0]);

    for (int i = 1; i < intervals.size(); ++i) {

        if (ret.back()[0] <= intervals[i][0] && ret.back()[1] >= intervals[i][1]) {
            continue;
        } else if (ret.back()[1] >= intervals[i][0]){
            ret.back()[1] = intervals[i][1];
        } else {
            ret.push_back(intervals[i]);
        }
    }

    return ret;
}


// #59 Spiral Matrix
static vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> ret(n, vector<int>(n));
    int up = 0, down = n - 1, left = 0, right = n-1, val = 1;

    while(true){
        for (int j = left; j <=right; ++j) ret[up][j] = val++;
        if (++up > down) break;
        for (int i = up; i <= down; ++i) ret[i][right] = val++;
        if (--right < left) break;
        for (int j = right; j >= left; --j) ret[down][j] = val++;
        if (--down < up) break;
        for (int i = down; i >= up; --i) ret[i][left] = val++;
        if (++left > right) break;
    }
    return ret;
}

// #62 Unique Paths - DP
/*
 * Initiate 2D array d[m][n] = number of paths. To start, put number of paths equal to 1 for the first row and the first column.
 * Iterate over all "inner" cells: d[col][row] = d[col - 1][row] + d[col][row - 1].
 * Return d[m - 1][n - 1]
 **/
static int uniquePaths(int m, int n) {
    int dp[n];
    for (int i = 0; i < n; ++i) dp[i] = 1;
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            dp[j] += dp[j - 1];
        }
    }
    return dp[n - 1];
}

// #63 Unique Paths II (With Obstacle) - DP
static int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int m = (int)obstacleGrid.size(), n = (int)obstacleGrid[0].size();
    if (m == 0 || n == 0) return 0;

    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m; ++i){
        if (!obstacleGrid[i][0]) dp[i][0] = 1;
        else break;
    }
    for (int i = 0; i < n; ++i){
        if (!obstacleGrid[0][i]) dp[0][i] = 1;
        else break;
    }
    if (m == 1) return dp[0][n-1];
    if (n == 1) return dp[m-1][0];
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) {
            if (obstacleGrid[i][j]) {
                dp[i][j] = 0; continue;
            }
            dp[i][j] += dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m-1][n-1];
}

// #64 Minimum Path Sum - DP
static int minPathSum(vector<vector<int>>& grid) {
    int m = (int)grid.size(), n = (int)grid[0].size();
    if (m == 0 || n == 0) return 0;
    int val = 0;
    if (m == 1){
        for (int i = 0; i < n; ++i) val += grid[0][i];
        return val;
    }
    if (n == 1){
        for (int i = 0; i < m; ++i) val += grid[i][0];
        return val;
    }
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < n; ++i){
        val += grid[0][i];
        dp[0][i] = val;
    }
    val = 0;
    for (int i = 0; i < m; ++i){
        val += grid[i][0];
        dp[i][0] = val;
    }
    for (int i = 1; i < m; ++i)
        for (int j = 1; j < n; ++j) {
            int tmp;
            dp[i-1][j] >= dp[i][j-1] ? tmp = dp[i][j-1] : tmp = dp[i-1][j];
            dp[i][j] = grid[i][j] + tmp;
        }

    return dp[m-1][n-1];

}

// #73 Set Matrix Zeroes
// Not allowed to use extra space (in-place algorithm)
// We need to utilize the first row & column to record the zero info for the whole matrix.
static void setZeroes(vector<vector<int>>& matrix) {
    if (matrix.empty() || matrix[0].empty()) return;
    int m = (int)matrix.size(), n = (int)matrix[0].size();
    bool row_flag = false, col_flag = false;

    for (int i = 0; i < m; ++i)
        if (matrix[i][0] == 0) { row_flag = true; break;}

    for (int i = 0; i < n; ++i)
        if (matrix[0][i] == 0) { col_flag = true; break;}

    for (int i = 1; i < m; ++i)
        for (int j = 1; j < n; ++j){
            if (!matrix[i][j]){
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }

    for (int i = 1; i < m; ++i)
        for (int j = 1; j < n; ++j){
            if (!matrix[i][0] || !matrix[0][j]){
                matrix[i][j] = 0;
            }
        }

    if (row_flag)
        for (int i = 0; i < m; ++i) matrix[i][0] = 0;

    if (col_flag)
        for (int i = 0; i < n; ++i) matrix[0][i] = 0;

}

// #74 Search a 2D Matrix - Binary Search
static bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;

    int m = (int)matrix.size(), n = (int)matrix[0].size();
    int mid, left, right;

    if (target < matrix[0][0]) return false;
    left = 0; right = m;
    while(left < right){
        mid = (left + right) / 2;
        if (matrix[mid][0] == target) return true;
        if (matrix[mid][0] > target){
            right = mid;
            continue;
        }
        if (matrix[mid][0] < target){
            left = mid + 1;
            continue;
        }
    }

    int target_row = (right > 0) ? (right - 1) : right;

    left = 0; right = n;
    while (left < right) {
        mid = (left + right) / 2;
        if (matrix[target_row][mid] == target) return true;
        if (matrix[target_row][mid] > target){
            right = mid;
            continue;
        }
        if (matrix[target_row][mid] < target){
            left = mid + 1;
            continue;
        }
    }

    return false;
}

// #74 Sort Colors (3 diff: 0, 1, 2)
static void sortColors1(vector<int>& nums) {
    int color_num[3] = {0};
    for (int num: nums) ++color_num[num];
    int pos = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < color_num[i]; ++j){
            nums[pos++] = i;
        }
}
static void sortColors2(vector<int>& nums) {
    int left = 0, right = (int)nums.size() - 1;
    for (int i = 0; i <= right; ++i) {
        if (nums[i] == 0) std::swap(nums[i], nums[left++]);
        else if (nums[i] == 2) std::swap(nums[i--], nums[right--]);
    }
}

// #81 Search in Rotated Sorted Array II
//
static bool searchRotatedArrII(vector<int>& nums, int target) {
    int left = 0, right = (int)nums.size() - 1, mid;
    while (left <= right) {
        mid = left + (right - left) / 2;
        if (nums[mid] == target) return true;
        if (nums[mid] > nums[left]) {
            if (target < nums[mid] && target >= nums[left]) {
                right = mid - 1;
            } else { // if (target > nums[mid]) {
                left = mid + 1;
            }
        } else if (nums[mid] < nums[left]) {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else { //if (target < nums[mid]) {
                right = mid - 1;
            }
        } else ++left;
    }
    return false;
}

// #84 Hard - Largest Rectangle in Histogram
// Given n non-negative integers representing the histogram's bar height where the width of each bar is 1
// find the area of largest rectangle in the histogram.
/*
*  单调栈保存着一个数组以下这样的信息：
*  如果是找某个位置左右两边大于此数且最下标靠近它的数位置，那么扫描到下标i的时候的单调栈保存的是0~i-1区间中数字的的递增序列的下标。
*  找某个位置左右两边小于此数且最下标靠近它的数的位置的情况类似
*  作用：可以O(1)时间得知某个位置左右两侧比他大（或小）的数的位置

*  什么时候能用单调栈？
*  在你有高效率获取某个位置左右两侧比他大（或小）的数的位置的需求的时候。
*  对于出栈元素来说：找到右侧第一个比自身小的元素。
*  对于新元素来说：等待所有破坏递增顺序的元素出栈后，找到左侧第一个比自身小的元素。
*  */
static int largestRectangleArea(vector<int> &height) {
    int res = 0;
    stack<int> stk;
    height.push_back(0);
    for (int i = 0; i < height.size(); ++i) {
        while (!stk.empty() && height[stk.top()] >= height[i]) {
            int h = height[stk.top()];
            stk.pop();
            int diff = !stk.empty() ? stk.top() : -1;
            res = std::max(res, h * (i - diff - 1));
        }

        stk.push(i);
    }
    return res;
}

// #88 easy - Merge Sorted Array
// Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
static void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int i = m - 1, j = n - 1, k = m + n - 1;
    while (j >= 0) {
        // if (i >= 0 && nums1[i] >= nums2[j]) { nums1[k--] = nums1[i--];
        // } else {
        //     nums1[k--] = nums2[j--];
        // }
        nums1[k--] = (i >= 0 && nums1[i] >= nums2[j]) ? nums1[i--] : nums2[j--];
    }
}

// #120
// Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
// For example, given the following triangle
//      [
//          [2],
//          [3,4],
//          [6,5,7],
//          [4,1,8,3]
//      ]
//The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
static int minimumTotal(vector<vector<int>>& triangle) {
    int n = (int)triangle.size();
    if (n == 0 || triangle[0].empty()) return 0;
    // if (n == 1) return triangle[0][0];
    for (int i = n - 2; i >= 0; --i) {
        for (int k = 0; k <= i; ++k) {
            triangle[i][k] += std::min(triangle[i+1][k], triangle[i+1][k+1]);
        }
    }
    return triangle[0][0];
}

// #121 Best Time to Buy and Sell Stock
static int maxProfit(vector<int>& prices) {
    int res = 0, buy = INT_MAX;
    for (auto price : prices) {
        buy = min(buy, price);
        res = max(res, price - buy);
    }
    return res;
}

// #122 Best Time to Buy and Sell Stock II
// You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).
// Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
static int maxProfitII(vector<int>& prices) {
    int res = 0, n = (int) prices.size();
    for (int i = 0; i < n - 1; ++i) {
        if (prices[i] < prices[i + 1]) {
            res += prices[i + 1] - prices[i];
        }
    }
    return res;
}

// # 123 Best Time to Buy and Sell Stock III - hard
// Say you have an array for which the ith element is the price of a given stock on day i.
// Design an algorithm to find the maximum profit. You may complete at most two transactions.
static int maxProfitIII(vector<int>& prices) {
    if (prices.empty()) return 0;
    int buy = 0x3f3f3f3f, profit = 0;
    int res_arr[prices.size()];
    for (int i = 0; i < (int)prices.size(); ++i){
        buy = std::min(buy, prices[i]);
        profit = std::max(profit, prices[i] - buy);
        res_arr[i] = profit;
    }
    int sell = 0, max_profit = 0; profit = 0;
    for (int j = (int)prices.size() - 1; j >= 0; --j) {
        sell = std::max(sell, prices[j]);
        profit = std::max(profit, sell - prices[j]);
        res_arr[j] += profit;
        max_profit = std::max(max_profit, res_arr[j]);
    }
    return max_profit;
}

// Two Pointers
// 209. Minimum Size Subarray Sum



/* LintSolutionArray */

// Prefix Sum Array
//   1 2 3  4  5  6
// 0 1 3 6 10 15 21
static vector<int> getPrefixSum(const vector<int>& nums){
    vector<int> prefix_sum(nums.size() + 1, 0);
    for (int i = 0; i < (int)nums.size(); ++i){
        prefix_sum[i + 1] = prefix_sum[i] + nums[i];
    }
    return prefix_sum;
}

// Prefix Sum Related

// #Lint1844 mid
/**
* @param nums: a list of integer
* @param k: an integer
* @return: return an integer, denote the minimum length of continuous subarrays whose sum equals to k
*/
static int subarraySumEqualsKII(vector<int> &nums, int k) {
    int rst = 0x3f3f3f3f;

    // prefix array length: len(nums) + 1
    // sum from element i to j: prefix[j+1] - prefix[i]
    vector<int> prefix_sum(nums.size() + 1, 0);
    for (int i = 0; i < (int)nums.size(); ++i){
        prefix_sum[i + 1] = prefix_sum[i] + nums[i];
    }

    std::unordered_map<int, int> sum2idx;
    sum2idx[0] = 0;
    for (int end = 0; end < nums.size(); ++end) {
        if (sum2idx.find(prefix_sum[end + 1] - k) != sum2idx.end()) {
            // found one start point that meet the sum-k req for the current end point
            // sum2idx[prefix_sum[end + 1] - k] is the start point idx
            int len = end + 1 - sum2idx[prefix_sum[end + 1] - k];
            rst = std::min(rst, len);
        }
        sum2idx[prefix_sum[end + 1]] = end + 1;
    }
    if (rst == 0x3f3f3f3f) {
        return -1;
    }
    return rst;
}

// #Lint1840 mid
/**
     * @param n: the row of the matrix
     * @param m: the column of the matrix
     * @param after: the matrix
     * @return: restore the matrix
     */
// algorithm from before to after matrix:
//    for (int i = 0; i < before.size(); ++i)
//        for (int j = 0; j < before[0].size(); ++j) {
//            int s = 0;
//            for (int i1 = 0; i1 <=i; ++i1)
//                for (int j1 = 0; j1 <= j; ++j1) {
//                    s = s + before[i1][j1];
//                }
//            after[i][j] = s;
//        }
vector<vector<int>> matrixRestoration(int n, int m, vector<vector<int>> &after) {
    vector<vector<int>> before(n, vector<int>(m, 0));
    // analysis (without corner cases):
    //      before[i][j] = after[i][j] - after[i-1][j] - after[i][j-1] + after[i-1][j-1];
    for (int i = n - 1; i >= 0; --i)
        for (int j = m - 1; j >= 0; --j) {
            if (i == 0 && j ==0) {
                before[i][j] = after[i][j];
                continue;
            }
            if (i == 0) {
                before[i][j] = after[i][j] - after[i][j-1];
            }

            if (j == 0) {
                before[i][j] = after[i][j] - after[i-1][j];
            }

            if (i != 0 && j != 0) {
                before[i][j] = after[i][j] - after[i-1][j] - after[i][j-1] + after[i-1][j-1];
            }
        }

    return before;
}

// #406 · Minimum Size Subarray Sum (Compared to #1507 (prefixSum array is not strictly increasing))
// Given an array of n positive integers and a positive integer s, find the minimal length of a subarray of which the sum ≥ s. If there isn't one, return -1 instead.

// Input: [2,3,1,2,4,3], s = 7
// Output: 2
// Explanation: The subarray [4,3] has the minimal length under the problem constraint.
/**
 * @param nums: an array of integers
 * @param s: An integer
 * @return: an integer representing the minimum size of subarray
 */
int minIndex(const vector<int> &prefix, int start, int s);
int minimumSize(vector<int> &nums, int s) {
    if (nums.empty()) return -1;

    int n = (int) nums.size(), res = 0x3f3f3f3f;

    vector<int> prefixSum(n+1, 0);
    for (int i = 1; i <= n; ++i) {
        prefixSum[i] = nums[i-1] + prefixSum[i-1];
        // printf("%d\t", prefixSum[i]);
    }
    // printf("\n");

    for (int start = 0; start < n; ++start){
        printf("start: %d\n", start);

        // Binary search the right subarray, because the prefixSum is strictly increasing
        int idx = minIndex(prefixSum, start, s);
        if (idx != -1) {
            printf("return idx: %d\n", idx);
            res = std::min(res, idx - start + 1);
        }
    }

    if (res == 0x3f3f3f3f) return -1;

    return res;
}
int minIndex(const vector<int>& prefix, const int start, const int s) {
    int n = (int)prefix.size() - 1, left = start, right = n - 1, mid = 0;

    while (left + 1 < right) {
        mid = left + (right - left) / 2;
        // printf("mid index: %d\n", mid);
        if(prefix[mid + 1] - prefix[start] == s) {
            return mid;
        } else if (prefix[mid + 1] - prefix[start] > s) {
            right = mid;
        } else {
            left = mid;
        }
    }

    if(prefix[left + 1] - prefix[start] >= s)
        return left;

    if(prefix[right + 1] - prefix[start] >= s)
        return right;

    return -1;
}

// #Lint1507 hard
/**
 * @param A: the array
 * @param K: sum
 * @return: the length
 */
static bool shortestSubarray_isValid(const vector<int> &prefixSum, int K, int length);
static int shortestSubarray(vector<int> &A, int K) {
    vector<int> prefixSum = getPrefixSum(A);
    int start = 1, mid, end = (int) A.size();

    // 二分答案 binary search on answer
    while(start + 1 < end) {
        mid = start + (end - start)/2;
        if (shortestSubarray_isValid(prefixSum, K, mid)) {
            end = mid;
        } else {
            start = mid;
        }
    }

    if (shortestSubarray_isValid(prefixSum, K, start))
        return start;

    if (shortestSubarray_isValid(prefixSum, K, end))
        return end;

    return -1;
}
static bool shortestSubarray_isValid(const vector<int> &prefixSum, const int K, const int length) {
    MinHeap minHeap;
    for (int end = 0; end < (int) prefixSum.size(); ++end) {
        minHeap.removeByIdx(end - length - 1); // range [end-length, end-1] subarray
        if (!minHeap.isEmpty() && prefixSum[end] - minHeap.top().second >= K)
            return true;
        minHeap.push(end, prefixSum[end]);
    }
    return false;
}

// #617
// Given an array with positive and negative numbers,
//   find the maximum average subarray which length should be greater or equal to given length k.

//   Input:
//   [1,12,-5,-6,50,3]
//   3
//   Output:
//   15.667
//   Explanation:
//   (-6 + 50 + 3) / 3 = 15.667
/**
 * @param nums: an array with positive and negative numbers
 * @param k: an integer
 * @return: the maximum average
 */
bool findMaxAverage_canFind(const vector<int> &nums, int k, double avg);
double findMaxAverage(vector<int>& nums, int k) {
    if (nums.empty()) {
        return 0;
    }

    // binary search on ans, condition + min/max
    int n = nums.size();

    double low = nums.front(), high = nums.front();
    for (int i = 0; i < n; i++) {
        low = min((double)nums[i], low);
        high = max((double)nums[i], high);
    }

    double mid;
    while (high - low > 1e-6) {
        mid = low + (high - low) / 2;
        if (findMaxAverage_canFind(nums, k, mid)) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}
bool findMaxAverage_canFind(const vector<int> &nums, int k, double avg) {
    int n = (int) nums.size();
    double rightSum = 0, leftSum = 0, minLeftSum = 0;
    for (int i = 0; i < k; ++i)
        rightSum += nums[i] - avg;

    if (rightSum >= 0)
        return true;

    for (int i = k; i < n; ++i) {
        rightSum += nums[i] - avg;
        leftSum += nums[i - k] - avg;
        minLeftSum = std::min(minLeftSum, leftSum);

        if (rightSum - minLeftSum >= 0)
            return true;
    }

    return false;
}


// #391 · Number of Airplanes in the Sky
//Input: [(1, 10), (2, 3), (5, 8), (4, 7)]
//Output: 3
//Explanation:
// The first airplane takes off at 1 and lands at 10.
// The second airplane takes off at 2 and lands at 3.
// The third airplane takes off at 5 and lands at 8.
// The forth airplane takes off at 4 and lands at 7.
// During 5 to 6, there are three airplanes in the sky.
/**
 * @param intervals: An interval array
 * @return: Count of airplanes are in the sky.
 */
// Definition of Interval:
class Interval {
public:
    int start, end;
    Interval(int start, int end) {
        this->start = start;
        this->end = end;
    }
};
int countOfAirplanes(vector<Interval> &airplanes) {
    priority_queue<int,vector<int>,greater<>> q;
    std::function<bool(const Interval& a, const Interval& b)> sortCmp = [] (const Interval& a, const Interval& b) -> bool {
        if (a.start == b.start) {
            return  a.end < b.end;
        }
        return a.start < b.start;
    };
    sort(airplanes.begin(), airplanes.end(), sortCmp);
    int res = 0;
    int n = airplanes.size();
    for (int i = 0; i < n; i++) {
        while (!q.empty() && q.top() <= airplanes[i].start)
            q.pop();
        q.push(airplanes[i].end);
        res = max(res, (int)q.size());
    }
    return res;
}

/**
     * @param str: the string
     * @param dict: the dictionary
     * @return: return words which  are subsequences of the string
     */
vector<string> findWords(string &str, vector<string> &dict) {
    if (str.empty() || dict.empty())
        return {};

    int n = (int) dict.size(), len = (int) str.size();
    std::vector<int> index(n, 0);
    std::vector<string> res = {};
    res.reserve(n);

    for (int i = 0; i < len; ++i)
        for (int j = 0; j < n; ++j) {
            if (index[j] == -1)
                continue;

            if (str[i] == dict[j][index[j]])
                ++index[j];

            if (index[j] == (int) dict[j].size())
                index[j] = -1;
        }

    for (int i = 0; i < n; ++i) {
        if (index[i] == -1)
            res.emplace_back(dict[i]);
    }

    return res;
}

// #149 · Best Time to Buy and Sell Stock
// Say you have an array for which the ith element is the price of a given stock on day i.
//   If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

//Input: [3, 2, 3, 1, 2]
//Output: 1
//Explanation: You can buy at the third day and then sell it at the 4th day. The profit is 2 - 1 = 1
/**
     * @param prices: Given an integer array
     * @return: Maximum profit
     */
int maxProfit_(vector<int> &prices) {
    // write your code here
    int n = (int) prices.size(), maxVal = prices[n - 1], maxProfit = 0;
    std::vector<int> maxStock(n, 0);
    for (int i = n - 1; i >= 1; --i) {
        maxVal = std::max(maxVal, prices[i]);
        maxStock[i] = maxVal;
    }

    for (int i = 0; i < n - 1; ++i) {
        maxProfit = std::max(maxProfit, maxStock[i+1] - prices[i]);
    }

    return maxProfit;
}


// #62 · Search in Rotated Sorted Array - O(logN) time limit
// Suppose a sorted array is rotated at some pivot unknown to you beforehand.
// (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
// You are given a target value to search. If found in the array return its index, otherwise return -1.
// You may assume no duplicate exists in the array.
int searchRotatedArray(vector<int> &A, int target) {
    // write your code here
    if (A.empty()) return -1;

    int n = (int) A.size(), left = 0, right = n - 1, mid = 0;

    while (left + 1 < right) {
        mid = left + (right - left) / 2;

        if (A[mid] == target) {
            return mid;
        } else if (A[mid] > A[left]) {
            //此时left和mid肯定处在同一个递增数组上
            //那么就直接运用原始的二分查找
            if ( (A[left] <= target) && (target < A[mid]) ) {
                right = mid;
            } else {
                left = mid;
            }
        } else {
            //此时mid处于第二个递增数组 left处于第一个递增数组 自然的mid和right肯定处于第二个递增数组上
            //还是直接运用原始的二分查找思想
            if ( (A[mid] < target) && (target <= A[right]) ) {
                left = mid;
            } else {
                right = mid;
            }
        }
    }

    if (A[left] == target)
        return left;

    if (A[right] == target)
        return right;

    return -1;
}


// #183 · Wood Cut
// Given n pieces of wood with length L[i] (integer array). Cut them into small pieces to guarantee you could have equal or more than k pieces with the same length. What is the longest length you can get from the n pieces of wood? Given L & k, return the maximum length of the small pieces.
// The unit of length is centimeter.The length of the woods are all positive integers,you couldn't cut wood into float length.If you couldn't get >= k pieces, return 0.

// Input:
// L = [232, 124, 456]
// k = 7
// Output: 114
// Explanation: We can cut it into 7 pieces if any piece is 114cm long, however we can't cut it into 7 pieces if any piece is 115cm long.

/**
 * @param L: Given n pieces of wood with length L[i]
 * @param k: An integer
 * @return: The maximum length of the small pieces
 */
bool woodCut_check (const vector<int> &L, int k, int mid);
int woodCut(vector<int> &L, int k) {
    if (L.empty()) return 0;

    int low = 1, high = L.front(), mid = 0;
    for (auto it : L) {
        high = std::max(high, it);
    }

    // Binary search on answer
    while (low + 1 < high) {
        mid = low + (high - low) / 2;

        if (woodCut_check(L, k, mid)) {
            low = mid;
        } else {
            high = mid;
        }

    }

    if (woodCut_check(L, k, high)) {
        return high;
    }

    if (woodCut_check(L, k, low)) {
        return low;
    }

    return 0;
}
bool woodCut_check (const vector<int> &L, int k, int mid) {
    int num = 0, n = (int) L.size();
    for (int i = 0; i < n; ++i) {
        num += L[i] / mid;
    }
    return num >= k;
}

// #1310 · Product of Array Except Self
// Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

// Input: [1,2,3,4]
// Output: [24,12,8,6]
// Explanation:
//  2*3*4=24
//  1*3*4=12
//  1*2*4=8
//  1*2*3=6

// Challenge
//   Could you solve it with constant space complexity? (Note: The output array does not count as extra space for the purpose of space complexity analysis.)
/**
 * @param nums: an array of integers
 * @return: the product of all the elements of nums except nums[i].
 */
vector<int> productExceptSelf(vector<int> &nums) {
    if (nums.empty()) return {};

    int n = (int) nums.size();
    vector<int> res(n, 1);

    int product = 1;

    // constant space solution
    for (int i = 1; i < n; ++i) {
        product *= nums[i-1];
        res[i] *= product;
    }

    product = 1;
    for (int i = n - 2; i >= 0; --i) {
        product *= nums[i+1];
        res[i] *= product;
    }

    return res;
}

// Two Pointer Related

// #406 · Minimum Size Subarray Sum
// Given an array of n positive integers and a positive integer s, find the minimal length of a subarray of which the sum ≥ s. If there isn't one, return -1 instead.
//   Input: [2,3,1,2,4,3], s = 7
//   Output: 2
//   Explanation: The subarray [4,3] has the minimal length under the problem constraint.
/**
* @param nums: an array of integers
* @param s: An integer
* @return: an integer representing the minimum size of subarray
*/
int minimumSize_two_pointer(vector<int> &nums, int s) {

    if (nums.empty()) return -1;

    int n = (int) nums.size(), right = 0, sumOfSubarray = 0, res = 0x3f3f3f3f;
    for (int left = 0; left < n; ++left) {
        while (right < n && sumOfSubarray < s) {
            sumOfSubarray += nums[right];
            ++right;
        }

        if (sumOfSubarray >= s) {
            res = std::min(res, right - left);
        }

        sumOfSubarray -= nums[left];
    }

    if (res == 0x3f3f3f3f) {
        return -1;
    }

    return res;

}


// #1375 · Substring With At Least K Distinct Characters
// Given a string S with only lowercase characters.
// Return the number of substrings that contains at least k distinct characters.
//   10 <= length(S) <= 1,000,000
//   1 <= k <= 26, 1 <= k <= 26

// Input: S = "abcabcabcabc", k = 3
// Output: 55
// Explanation: Any substring whose length is not smaller than 3 contains a, b, c.
//     For example, there are 10 substrings whose length are 3, "abc", "bca", "cab" ... "abc"
//     There are 9 substrings whose length are 4, "abca", "bcab", "cabc" ... "cabc"
//     ...
//     There is 1 substring whose length is 12, "abcabcabcabc"
//     So the answer is 1 + 2 + ... + 10 = 55.

/**
 * @param s: a string
 * @param k: an integer
 * @return: the number of substrings there are that contain at least k distinct characters
 */
long long kDistinctCharacters(string &s, int k) {
    if (s.empty()) return 0;

    int n = (int) s.size(), diff = 0, right = 0;
    long long res = 0;
    std::unordered_map<char, int> char2num;

    for (int left = 0; left < n; ++left) {

        while (right < n && diff < k) {
            if (!char2num.count(s[right])) // if not exist, init 0
                char2num[s[right]] = 0;

            ++char2num[s[right]];
            if (char2num[s[right]] == 1) // char s[right] is new here
                ++diff;

            ++right;
        }
        if (diff == k)
            res += n - (right -1); // length from (right-1) to n  -- [right-1, n]

        if ((char2num[s[left]]--) == 1) // remove the most left char before the next iteration
            --diff;
    }

    return res;
}


// #32 · Minimum Window Substring
// Given two strings source and target. Return the minimum substring of source which contains each char of target.
//    If there is no answer, return "".
//    You are guaranteed that the answer is unique.
//    target may contain duplicate char, while the answer need to contain at least the same number of that char.

// Example
// Input:
//   source = "abc"
//   target = "ac"
// Output:
//   "abc"
// Explanation:
//   "abc" is the minimum substring of source string which contains each char of target "ac".
string minWindow(string &source , string &target) {
    if(source.empty() || target.empty()) return "";

    std::unordered_map<char, int> targetChar2Num, sourceChar2Num;
    int n = (int) source.size(), m = (int) target.size(), j = 0, resLen = 0x3f3f3f3f, resStart = 0;;
    // std::pair<int, int> resPair(0, 0);
    int count = 0;

    for (int i = 0; i < m; ++i) {
        if (!targetChar2Num.count(target[i]))
            targetChar2Num[target[i]] = 0;

        ++targetChar2Num[target[i]];
    }

    for (int i = 0; i < n; ++i) {

        while(j < n && count < m) {

            if (!sourceChar2Num.count(source[j]))
                sourceChar2Num[source[j]] = 0;

            ++sourceChar2Num[source[j]];

            if(targetChar2Num.count(source[j]) && sourceChar2Num[source[j]] <= targetChar2Num[source[j]])
                ++count;

            ++j;
        }

        if (count == m && (j-i) < resLen) {
            resLen = j - i;
            resStart = i;
            // resPair.first = i;
            // resPair.second = j;
        }

        --sourceChar2Num[source[i]];
        if (targetChar2Num.count(source[i]) && sourceChar2Num[source[i]] < targetChar2Num[source[i]])
            --count;
    }

    if (resLen == 0x3f3f3f3f)
        return "";

    // return source.substr(resPair.first, resPair.second - resPair.first);
    return source.substr(resStart, resLen);
}


// #1219 · Heaters (Google - hard)
// Input: [1,2,3,4],[1,4]
// Output: 1
// Explanation: The two heater was placed in the position 1 and 4. We need to use radius 1 standard, then all the houses can be warmed.
/**
 * @param houses: positions of houses
 * @param heaters: positions of heaters
 * @return: the minimum radius standard of heaters
 */

// Solution I: binary search + double pointers
bool findRadius_check(int rds, const vector<int> &houses, const vector<int> &heaters);
int findRadius(vector<int> &houses, vector<int> &heaters) {
    // if (houses.empty() || heaters.empty()) return -1;
    auto cmp = [](const int a, const int b) -> bool{return a < b;};
    std::sort(houses.begin(), houses.end(), cmp);
    std::sort(heaters.begin(), heaters.end(), cmp);

    int n = (int)houses.size(), m = (int)heaters.size(), left = 0, right = std::max(houses[n-1], heaters[m-1]), mid = 0;

    while (left + 1 < right) {
        mid = left + (right - left) / 2;

        if (findRadius_check(mid, houses, heaters)) {
            right = mid;
        } else {
            left = mid;
        }

    }

    if (findRadius_check(left, houses, heaters))
        return left;

    // if (findRadius_check(right, houses, heaters))
        // return right;

    return right;
}
bool findRadius_check(int rds, const vector<int> &houses, const vector<int> &heaters) {
    int i = 0, j = 0, n = (int)houses.size(), m = (int)heaters.size();

    while (i < n && j < m) {
        // if (abs(houses[i] - heaters[j]) <= rds) {
        if ( heaters[j] - rds <= houses[i] && houses[i] <= heaters[j] + rds) {
            ++i;
        } else {
            ++j;
        }
    }

    if (j == m)
        return false;

    return i == n;
}

// Solution II: double pointers without binary search
//  Notes: based on heaters, finding the correct position for each house
//  注解：以heaters为轴，利用同向双指针往上顺次填houses，期间打擂台求max半径，即为最小值
int findRadius_II(vector<int> &houses, vector<int> &heaters) {

    auto cmp = [](const int a, const int b) -> bool{return a < b;};
    std::sort(houses.begin(), houses.end(), cmp);
    std::sort(heaters.begin(), heaters.end(), cmp);


    int n = (int)houses.size(), m = (int)heaters.size(), i = 0, j = 0;
    int res = INT_MIN, currRds = 0, nextRds = 0;

    while (i < n && j < m) {
        currRds = std::abs(heaters[j] - houses[i]);
        nextRds = INT_MAX;
        if (j < m - 1)
            nextRds = std::abs(heaters[j+1] - houses[i]);

        if (currRds < nextRds) { // using <= will not pass test cases
            res = std::max(res, currRds);
            // res = res > prevRds ? res : prevRds;
            ++i;
        } else {
            ++j;
        }
    }

    return res;
}


// #1850 · Pick Apples

// Input:
//   A = [6, 1, 4, 6, 3, 2, 7, 4]
//   K = 3
//   L = 2
// Output:
//   24
// Explanation:
//   because Alice can choose tree 3 to 5 and collect 4 + 6 + 3 = 13 apples, and Bob can choose trees 7 to 8 and collect 7 + 4 = 11 apples.Thus, they will collect 13 + 11 = 24.

/**
 * @param A: a list of integer
 * @param K: a integer
 * @param L: a integer
 * @return: return the maximum number of apples that they can collect. (-1 when there is no answer)
 */

// 因为互相不干涉，所以考虑隔板法，分割成左右两部分
// max(max(左窗口sum) + 当前右窗口sum) 即为结果
// 需要考虑左边K，右边L和左边L，右边K
int PickApples(vector<int> &A, int K, int L) {
    int n = (int) A.size(), leftMax = 0, res = -1;
    if (A.empty() || K + L > n) return -1;

    vector<int> prefix = getPrefixSum(A);

    // K left, L right
    for (int i = K; i <= n - L; ++i) {
        // #i is included in the L, not in K
        //  K range: [i-K, i)
        //  L range: [i, i+L)
        leftMax = std::max(leftMax, prefix[i]-prefix[i-K]);
        res = std::max(res, leftMax + prefix[i+L]-prefix[i]);
    }

    // L left, K right
    leftMax = 0;
    for (int i = L; i <= n - K; ++i) {
        leftMax = std::max(leftMax, prefix[i]-prefix[i-L]);
        res = std::max(res, leftMax + prefix[i+K]-prefix[i]);
    }

    return res;
}


// #1849 · Grumpy Bookstore Owner
//Input:
//  [1,0,1,2,1,1,7,5]
//  [0,1,0,1,0,1,0,1]
//  3
//Output:
//  16
//Explanation:
//  The bookstore owner keeps themselves not grumpy for the last 3 days.
//  The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.
/**
 * @param customers: the number of customers
 * @param grumpy: the owner's temper every day
 * @param X: X days
 * @return: calc the max satisfied customers
 */
int maxSatisfied(vector<int> &customers, vector<int> &grumpy, int X) {

    int n = (int) customers.size(), res = 0;
    std::vector<int> prefix(n+1, 0), prefixTemper(n+1, 0);

    for (int i = 0; i < n; ++i) {
        prefix[i+1] = prefix[i] + customers[i];

        int tmp = (grumpy[i] == 0) ? customers[i] : 0;
        prefixTemper[i+1] = prefixTemper[i] + tmp;
    }

    // if (n == X) {
    //     return prefix[n] - prefix[0];
    // }

    for (int i = 0; i <= n - X; ++i) {
        res = std::max(res, prefixTemper[i] - prefixTemper[0] + prefix[i+X] - prefix[i] + prefixTemper[n] - prefixTemper[i+X]);
    }

    return res;
}

// #404 · Subarray Sum II
// Given an positive integer array A and an interval.
// Return the number of subarrays whose sum is in the range of given interval.

// Input: A = [1, 2, 3, 4], start = 1, end = 3
// Output: 4
// Explanation: All possible subarrays: [1](sum = 1), [1, 2](sum = 3), [2](sum = 2), [3](sum = 3).

// Input: A = [1, 2, 3, 4], start = 1, end = 100
// Output: 10
// Explanation: Any subarray is ok.

/**
 * @param A: An integer array
 * @param start: An integer
 * @param end: An integer
 * @return: the number of possible answer
 */
int subarraySumII(vector<int> &A, int start, int end) {

    if (start > end || A.empty()) return 0;

    int n = (int) A.size(), j = 0, k = 0;
    vector<int> prefix (n+1, 0);
    int res = 0;

    for (int i = 0; i < n; ++i) {
        prefix[i+1] = prefix[i] + A[i];
    }

    // i为左端点, j和k为右端点
    // 也可以考虑j和k为左端点，i为右端点 i from 1 to n:
    //     while(j < i && prefix[i] - prefix[j] > end) ++j;
    //     while(k < i && prefix[i] - prefix[k] >= start) ++k;
    for (int i = 0; i < n; ++i) {

        while (j < n && prefix[j+1] - prefix[i] < start)
            ++j;

        while (k < n && prefix[k+1] - prefix[i] <= end)
            ++k;

        // 必须保证i在j,k的左边才算有效
        if (i <= j)
            res += (k - 1) - j + 1;
    }

    return res;
}

// #1879 · Two Sum VII
// Given an array of integers that is already sorted in ascending absolute order, find two numbers so that the sum of them equals a specific number.
// The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Note: the subscript of the array starts with 0
// You are not allowed to sort this array.

// Input:
//   [0,-1,2,-3,4]
//   1
// Output:
//   [[1,2],[3,4]]
// Explanation:
//   nums[1] + nums[2] = -1 + 2 = 1, nums[3] + nums[4] = -3 + 4 = 1
//   You can return [[3,4],[1,2]], the system will automatically help you sort it to [[1,2],[3,4]]. But [[2,1],[3,4]] is invaild.
/**
 * @param nums: the input array
 * @param target: the target number
 * @return: return the target pair
 */
int getNextLeft(vector<int> &nums, int left);
int getNextRight(vector<int> &nums, int right);
vector<vector<int>> twoSumVII(vector<int> &nums, int target) {
    int left = 0, right = 0;
    vector<vector<int>> res;

    for (int i = 1; i < nums.size(); ++i) {
        right = nums[i] > nums[right] ? i : right;
        left = nums[i] < nums[left] ? i : left;
    }

    while (nums[left] < nums[right]) {
        if (nums[left] + nums[right] > target) {
            right = getNextRight(nums, right);
            if (right == -1)
                break;
        } else if (nums[left] + nums[right] < target) {
            left = getNextLeft(nums, left);
            if (left == -1)
                break;
        } else { // nums[left] + nums[right] == target
            vector<int> temp {left, right};
            if (left > right)
                swap(temp[0], temp[1]);
            res.push_back(temp);

            // left 或 right 要前進，否則會無窮迴圈
            left = getNextLeft(nums, left);
            if (left == -1)
                break;
        }
    }
    return res;
}
int getNextLeft(vector<int> &nums, int left) {
    if (nums[left] < 0) {
        for (int i = left - 1; i >= 0; i--) {
            if (nums[i] < 0) {
                return i;
            }
        }
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] >= 0) {
                return i;
            }
        }
        return -1;
    }
    for (int i = left + 1; i < nums.size(); i++) {
        if (nums[i] >= 0) {
            return i;
        }
    }
    return -1;
}
int getNextRight(vector<int> &nums, int right) {
    if (nums[right] > 0) {
        for (int i = right - 1; i >= 0; i--) {
            if (nums[i] > 0) {
                return i;
            }
        }
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] <= 0) {
                return i;
            }
        }
        return -1;
    }
    for (int i = right + 1; i < nums.size(); i++) {
        if (nums[i] <= 0) {
            return i;
        }
    }
    return -1;
}




#endif //ALGORITHMPRACTICE_ARRAY_RELATED_H