//
// Created by Xiaoru_Zhu on 2020/2/25.
//

#ifndef ALGORITHMPRACTICE_STRING_RELATED_H
#define ALGORITHMPRACTICE_STRING_RELATED_H

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
P         I
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

// #49
static vector<vector<string>> groupAnagrams(vector<string>& strs) {
    std::unordered_map<string, std::vector<string>> dict;
    for(const auto & str : strs){
        string key = str;
        std::sort(key.begin(), key.end());
        dict[key].push_back(str);
    }
    std::vector<std::vector<string>> ret_vec;
    ret_vec.reserve(dict.size());
    for(const auto & pair : dict) ret_vec.push_back(pair.second);
    return ret_vec;
}


#endif //ALGORITHMPRACTICE_STRING_RELATED_H
