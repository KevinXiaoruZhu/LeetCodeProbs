//
// Created by Kevin Xiaoru Zhu on 2/25/2020.
//

#ifndef LEETCODEPROBS_DYNAMIC_PROGRAMMING_H
#define LEETCODEPROBS_DYNAMIC_PROGRAMMING_H
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
class SolutionDP{
public:

    // #10 -hard
    // String Matching
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
    // dp[i][j] indicates whether substring s[0, i) matches substring p[0, j)
    // 1. P[i][j] = P[i - 1][j - 1], if p[j - 1] != '*' && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
    // 2. P[i][j] = P[i][j - 2], if p[j - 1] == '*' and the pattern repeats for 0 times;
    // 3. P[i][j] = P[i - 1][j] && (s[i - 1] == p[j - 2] || p[j - 2] == '.'), if p[j - 1] == '*' and the pattern repeats for at least 1 times.
    static bool isMatch(string s, string p) {
        int m = (int)s.size(), n = (int)p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (j > 1 && p[j - 1] == '*') {
                    dp[i][j] = dp[i][j - 2] || (i > 0 && (s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1][j]);
                } else {
                    dp[i][j] = i > 0 && dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
                }
            }
        }
        return dp[m][n];
    }


    // #44 Wildcard Matching -hard
    // '?' Matches any single character.
    // '*' Matches any sequence of characters (including the empty sequence).
    static bool isMatchWild(string s, string p) {
        int m = (int)s.size(), n = (int)p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for (int i = 1; i <= n; ++i) {
            if (p[i - 1] == '*') dp[0][i] = dp[0][i - 1];
        }
        for (int i = 1; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p[j - 1] == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                } else {
                    dp[i][j] = (s[i - 1] == p[j - 1] || p[j - 1] == '?') && dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }
};


#endif //LEETCODEPROBS_DYNAMIC_PROGRAMMING_H
