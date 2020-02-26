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


    // #44 Wildcard Matching -hard
    // '?' Matches any single character.
    // '*' Matches any sequence of characters (including the empty sequence).

};


#endif //LEETCODEPROBS_DYNAMIC_PROGRAMMING_H
