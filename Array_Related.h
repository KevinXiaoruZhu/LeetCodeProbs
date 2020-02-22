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

class Solution {
public:
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


    // #3
    /*
    Given a string, find the length of the longest substring without repeating characters.
        Example 1:

        Input: "abcabcbb"
        Output: 3
        Explanation: The answer is "abc", with the length of 3.
        Example 2:

        Input: "bbbbb"
        Output: 1
        Explanation: The answer is "b", with the length of 1.
        Example 3:

        Input: "pwwkew"
        Output: 3
        Explanation: The answer is "wke", with the length of 3.

    Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
    */
    static int lengthOfLongestSubstring(string s) {
        unordered_set<char> st;
        int len = 0, left = 0, right = 0;
        while(right < s.size()){
            if(st.find(s[right]) == st.end()){ // 'st.end()' -->> means do not find the target element in the set
                st.insert(s[right++]);
                len = max(len, (int)st.size());
            }else{
                st.erase(s[left++]);
            }
        }
        return len;
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

    // #5
    /*
    Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

    Example 1:
    Input: "babad"
    Output: "bab"
    Note: "aba" is also a valid answer.

    Example 2:
    Input: "cbbd"
    Output: "bb"
    */
    /* Hint: Dynamic Programming
     * DP[i, j] = {
     *  true, (if i == j)
     *  s[i] == s[j], (if j = i + 1)
     *  s[i] == s[j] && DP[i + 1][j - 1], (if j > i + 1)
     * }
     * 注意dp bottom-up 的顺序
     * */
    static string longestPalindrome(string s) {
        if (s.empty()) return "";
        int n = s.size(), dp[n][n], left = 0, len = 1;
        for (int i = 0; i < n; ++i) {
            dp[i][i] = 1;
            for (int j = 0; j < i; ++j) {
                dp[j][i] = (s[i] == s[j] && (i - j < 2 || dp[j + 1][i - 1]));
                if (dp[j][i] && len < i - j + 1) {
                    len = i - j + 1;
                    left = j;
                }
            }
        }
        return s.substr(left, len);
    }

    // #6
    /*
    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows
    like this: (you may want to display this pattern in a fixed font for better legibility)

    P   A   H   N
    A P L S I I G
    Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"

    Write the code that will take a string and make this conversion given a number of rows:

    string convert(string s, int numRows);
    Example 1:
    Input: s = "PAYPALISHIRING", numRows = 3
    Output: "PAHNAPLSIIGYIR"

    Example 2:
    Input: s = "PAYPALISHIRING", numRows = 4
    Output: "PINALSIGYAHRPI"

    Explanation:
    P     I    N
    A   L S  I G
    Y A   H R
    P     I
    */
    static string convert(string s, int numRows) {
        if (numRows <= 1) return s;
        string ret_str;
        int period = 2 * numRows - 2, n = s.size();
        for (int i = 0; i < numRows; ++i) {
            for (int j = i; j < n; j += period) {
                ret_str += s[j];
                int tmp_pos = j + (numRows - 1 - i) * 2 ;
                if (i != 0 && i != numRows - 1 && tmp_pos < n)
                    ret_str += s[tmp_pos];
            }
        }
        return ret_str;
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

    // #8
    // String to Integer (atoi)
    static int myAtoi(string str) {
        if(str.empty()) return 0;
        int idx = 0, n = str.size();
        long long ret_num = 0;
        bool is_negative = false;
        while(idx < n && str[idx] != ' ') ++idx;
        if(idx < n && (str[idx] == '-' || str[idx] == '+'))
            str[idx++] == '-' ? is_negative = true : is_negative = false;
        while(idx < n && str[idx] >= '0' && str[idx] <= '9')
            ret_num = ret_num * 10 + (str[idx++] - '0');
        if(is_negative) ret_num *= -1;
        if(ret_num >= INT32_MAX) return INT32_MAX;
        if(ret_num <= INT32_MIN) return INT32_MIN;
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

    // #10 -hard
    /*
    Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
    '.' Matches any single character.
    '*' Matches zero or more of the preceding element.
    The matching should cover the entire input string (not partial).

    Note:
    s could be empty and contains only lowercase letters a-z.
    p could be empty and contains only lowercase letters a-z, and characters like . or *.

    Example 1:
    Input:
    s = "aa"
    p = "a"
    Output: false
    Explanation: "a" does not match the entire string "aa".

    Example 2:
    Input:
    s = "aa"
    p = "a*"
    Output: true
    Explanation: '*' means zero or more of the preceding element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".

    Example 3:
    Input:
    s = "ab"
    p = ".*"
    Output: true
    Explanation: ".*" means "zero or more (*) of any character (.)".

    Example 4:
    Input:
    s = "aab"
    p = "c*a*b"
    Output: true
    Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore, it matches "aab".

    Example 5:
    Input:
    s = "mississippi"
    p = "mis*is*p*."
    Output: false
    */
    static bool isMatch(string s, string p) {
        if (p.empty()) return s.empty(); // size of string p is 0
        if (p.size() > 1 && p[1] == '*'){ // '*' would not starts without preceding characters
            return isMatch(s, p.substr(2)) || // p[0] repeat 0 times (skip over the '*')
                   ( !s.empty() && (p[0] == s[0] || p[0] == '.') && isMatch(s.substr(1), p) );
        }else{ // I. p[0] and p[1] are 'a-z' or '.'  II. size of string p is 1
            return !s.empty() && (s[0] == p[0] || p[0] == '.') && isMatch(s.substr(1), p.substr(1));
        }
    }

    // #11
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

    // #17
    //
    static vector<string> letterCombinations(string digits) {
        vector<string> vec{"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        vector<string> ret_vec{""};
        for (char digit : digits) {
            vector<string> tmp;
            string str = vec[digit - '0' - 2];
            for (char ch : str) {
                for (const string& s : ret_vec)
                    tmp.push_back(s + ch);
            }
            ret_vec = tmp;
        }
        return ret_vec;
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

    // #23
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
        while(pos!=nums.end()){
            if(*pos == curr){
                pos = nums.erase(pos);
            }else{
                curr = *pos; ++pos;
            }
        }
        return (int)nums.size();
    }

    // #28
    // Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
    static int strStr(const string& haystack, const string& needle) {
        if(needle.empty()) return 0;
        int nd_len = (int)needle.size(), hs_len = (int)haystack.size();
        if(nd_len > hs_len) return -1;
        for(int i = 0; i <= hs_len - nd_len; ++i){
            string tmp_str = haystack.substr(i, nd_len);
            if(tmp_str == needle) return i;
        }
        return -1;
    }

    // #30 ********
    // Substring with Concatenation of All Words
    //  Input:
    //   s = "barfoothefoobarman",
    //   words = ["foo","bar"]
    //  Output: [0,9]
    static vector<int> findSubstring(string s, vector<string>& words) {
        if (s.empty() || words.empty()) return {};
        std::vector<int> ret_vec;
        int len = (int)words[0].size(), vec_size = (int)words.size();
        std::unordered_map<string, int> word_count;
        // must be ++word_count[word] to prevent the failure case of:
        // ["word","good","best","good"]
        for(const string& word : words) ++word_count[word];

        for(int i = 0; i <= (int)s.size() - len * vec_size; ++i ){
            bool flag = true;
            std::unordered_map<string, int> tmp_map;
            for(int j = 0; j < vec_size; ++j){
                string tmp_str = s.substr(i + j * len, len);
                if(!word_count.count(tmp_str)) {flag = false; break;}
                ++tmp_map[tmp_str];
                if(tmp_map[tmp_str] > word_count[tmp_str]) {flag = false; break;}
            }
            if(flag) ret_vec.emplace_back(i);
        }
        return ret_vec;
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
    // Permutations
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


    // #33
    // Binary search in totated array
    static int search(vector<int>& nums, int target) {
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
        int fisrt_target_idx = first_greater_equal(nums, target);
        if(nums[fisrt_target_idx] != target || fisrt_target_idx == (int)nums.size()){
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


};

#endif //ALGORITHMPRACTICE_ARRAY_RELATED_H
