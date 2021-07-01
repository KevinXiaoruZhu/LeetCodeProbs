//
// Created by Kevin Xiaoru Zhu on 3/4/2020.
//

#ifndef LEETCODEPROBS_TRAVERSAL_H
#define LEETCODEPROBS_TRAVERSAL_H
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


// #51 N-Queens
static bool verifyNQ(vector<string>& queen, int row, int clm);
static void dfsNQ(vector<vector<string>>& ret, vector<string>& queen, int curr_row);
static vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> ret;
    vector<string> queen(n, string(n, '.'));
    dfsNQ(ret, queen, 0);
    return ret;
}
static void dfsNQ(vector<vector<string>>& ret, vector<string>& queen, int curr_row){
    if(curr_row == (int)queen.size()) {
        ret.emplace_back(queen);
        return;
    }
    for(int i = 0; i < (int)queen.size(); ++i){
        if(verifyNQ(queen, curr_row, i)){
            queen[curr_row][i] = 'Q';
            dfsNQ(ret, queen, curr_row + 1);
            queen[curr_row][i] = '.';
        }
    }
}
static bool verifyNQ(vector<string>& queen, int row, int clm){
    for(int i = 0; i < row; ++i) if(queen[i][clm] == 'Q') return false;
    for(int i = row - 1, j = clm - 1; i >= 0 && j >= 0; --i, --j) {
        if(queen[i][j] == 'Q') return false;
    }
    for(int i = row - 1, j = clm + 1; i >= 0 && j < (int)queen.size(); --i, ++j ){
        if(queen[i][j] == 'Q') return false;
    }
    return true;
}


// # 78 subsets
// input [1,2,3]
// output [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
static void dfsSubsets(vector<vector<int>>& ret, vector<int>& out, int pos, vector<int>& nums);
static vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> ret;
    vector<int> out;
    std::sort(nums.begin(), nums.end());

    dfsSubsets(ret, out, 0, nums);

    return ret;
}
static void dfsSubsets(vector<vector<int>>& ret, vector<int>& out, int pos, vector<int>& nums) {
    ret.push_back(out);
    for(int i = pos; i < nums.size(); ++i) {
        out.push_back(nums[i]);
        dfsSubsets(ret, out, i + 1, nums);
        out.pop_back();
    }
}

// #79 Word Search
// The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring.
// The same letter cell may not be used more than once.
static bool dfs_exist(const vector<vector<char>>& board, const string& word, int idx, int i, int j, vector<vector<bool>>& visited);
static bool exist(vector<vector<char>>& board, string word) {
    if (board.empty() || board[0].empty()) return false;
    int m = (int)board.size(), n = (int)board[0].size();

    vector<vector<bool>> visited(m, vector<bool>(n));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            if (dfs_exist(board, word, 0, i, j, visited)) return true;

    return false;
}
static bool dfs_exist(const vector<vector<char>>& board, const string& word, int idx, int i, int j, vector<vector<bool>>& visited) {
    if (idx == (int)word.size()) return true;

    int m = (int)board.size(), n = (int)board[0].size();
    if (i < 0 || j < 0 || i >= m || j >= n || visited[i][j] || board[i][j] != word[idx]) return false;

    visited[i][j] = true;
    bool rst = dfs_exist(board, word, idx + 1, i - 1, j, visited) ||
               dfs_exist(board, word, idx + 1, i, j - 1, visited) ||
               dfs_exist(board, word, idx + 1, i + 1, j, visited) ||
               dfs_exist(board, word, idx + 1, i, j + 1, visited);
    visited[i][j] = false;

    return rst;
}

// #90
static void subsetsWithDupDFS(const vector<int>& nums, int pos, vector<int>& out, vector<vector<int>>& ret);
static vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    if (nums.empty()) return {};
    vector<vector<int>> ret;
    vector<int> out;
    std::sort(nums.begin(), nums.end());
    subsetsWithDupDFS(nums, 0, out, ret);
    return ret;
}
static void subsetsWithDupDFS(const vector<int>& nums, int pos, vector<int>& out, vector<vector<int>>& ret){
    ret.push_back(out);
    // for (auto it : out) std::cout << it << "\t"; std::cout << "\n";
    for (int i = pos; i < (int)nums.size(); ++i) {
        out.push_back(nums[i]);
        subsetsWithDupDFS(nums, i + 1, out, ret);
        out.pop_back();
        while (i + 1 < (int)nums.size() && nums[i] == nums[i + 1]) ++i;
    }
}

// #126 Word Ladder II - hard
// Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:
// Only one letter can be changed at a time
// Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
// Note:
//  Return an empty list if there is no such transformation sequence.
//  All words have the same length.
//  All words contain only lowercase alphabetic characters.
//  You may assume no duplicates in the word list.
//  You may assume beginWord and endWord are non-empty and are not the same.
static vector<vector<string>> findLaddersII(string beginWord, string endWord, vector<string>& wordList) {
    // std::map<string, int> map1{{"key1", 3}, std::pair<string, int>("key2", 1)};
    // if(std::find(wordList.begin(), wordList.end(), endWord) == wordList.end() || wordList.empty()) return {};
    int word_len = (int)beginWord.size(), list_len = (int)wordList.size();
    std::unordered_set<string> dict(wordList.begin(), wordList.end());
    if (!dict.count(endWord)) return {};
    dict.erase(beginWord); dict.erase(endWord);
}

#endif //LEETCODEPROBS_TRAVERSAL_H
