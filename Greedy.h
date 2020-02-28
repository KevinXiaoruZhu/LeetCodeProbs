//
// Created by Kevin Xiaoru Zhu on 2/27/2020.
//

#ifndef LEETCODEPROBS_GREEDY_H
#define LEETCODEPROBS_GREEDY_H
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
class SolutionGreedy{
public:

    // #44 Jump II - hard
    static int jump(const vector<int>& nums) {
        if(nums.size() < 2) return 0;
        int max_ = 0, pre = 0, last_pre = 0, jump_num = 0;
        while(max_ < nums.size() - 1){
            ++jump_num;
            last_pre = pre;
            pre = max_;
            for(int i = last_pre; i <= pre; ++i){
                max_ = std::max(max_, i + nums[i]);
            }
            if(max_ == pre) return -1; // optional
        }
        return jump_num;
    }


    // # 55 JUMP I
    // DP Alg
    static bool canJumpDP(const vector<int> & nums){
        if(nums.size() < 2) return true;
        bool dp[(int)nums.size()];
        for(int i = 0; i < (int)nums.size(); ++i) dp[i] = false;
        dp[0] = true;
        for(int i = 1; i < (int)nums.size(); ++i){
            for(int j = i - 1; j >= 0; --j)
                if(dp[j] && nums[j] >= i - j) {
                    dp[i] = true; break;
                }
        }
        return dp[(int)nums.size() - 1];
    }
    // Greedy Alg
    static bool canJumpGreedy(const vector<int> & nums){
        int n = (int)nums.size(), reach = 0;
        for (int i = 0; i < n; ++i) {
            if (i > reach || reach >= n - 1) break;
            reach = max(reach, i + nums[i]);
        }
        return reach >= n - 1;
    }


};
#endif //LEETCODEPROBS_GREEDY_H
