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
    // #44 Jump II
    static int jump(vector<int>& nums) {
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
};
#endif //LEETCODEPROBS_GREEDY_H
