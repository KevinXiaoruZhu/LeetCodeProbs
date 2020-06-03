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

class SolutionTraversal {
public:

    // #51 N-Queens
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
    
};
#endif //LEETCODEPROBS_TRAVERSAL_H
