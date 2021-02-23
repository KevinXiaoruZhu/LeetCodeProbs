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

using namespace std;
const int INT_MAX = INT32_MAX;

class SolutionArray {
public:

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
        int mid1 = ( (i + k / 2 - 1) < nums1.size() ? nums1[i + k / 2 - 1] : INT32_MAX);
        int mid2 = ( (j + k / 2 - 1) < nums2.size() ? nums2[j + k / 2 - 1] : INT32_MAX);
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
        int diff = INT32_MAX, closest = INT32_MAX;
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


};

#endif //ALGORITHMPRACTICE_ARRAY_RELATED_H
