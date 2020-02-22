#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
// #include "Array_Related.h"
using namespace std;

typedef struct {
    char ch1;
} SS0;
typedef struct {
    char ch1;
    char ch2;
    char ch3;
} SS1;
typedef struct {
    u_int16_t ui;
    char ch1;
} SS2;
typedef struct {
    u_int16_t ui;
    char ch1;
    char ch2;
    char ch3;
} SS3;
typedef struct {
    u_int16_t ui;
    unsigned u;
} SS4;
void resolveVarcharKey(const void *key, int &len, std::string &str) {
    len = *(int *)key;
    char tmp_chars[len + 1];
    std::memcpy(tmp_chars, &((char *)key)[sizeof(int)], len);
    tmp_chars[len] = '\0';
    std::string tmp_str(tmp_chars);
    str = tmp_str; // copy constructor
}
int main_test_cs222() {

    char combo[9] = {};
    int len = 5;
    memcpy(combo, (char *)&len, sizeof(int));
    combo[4] = 'k';
    combo[5] = 'e';
    combo[6] = 'v';
    combo[7] = 'i';
    combo[8] = 'n';

    string strd = "database";
    string strk = "kevin";
    const char *ch = strd.c_str();
    char ch111[5] = {'k', 'k', 'k', 'e', '\0'};
    char ch112[5] = {'k', 'e', 'k', 'k', '\0'};
    string ssss = ch111;

    char * tmp_ch;
    vector<char*> vec_ch;
    for(int i = 0; i < 3; ++i){
        tmp_ch = new char[5];
        if(i == 0)
            memcpy(tmp_ch, ch111, 5);
        else
            memcpy(tmp_ch, ch112, 5);
        vec_ch.emplace_back(tmp_ch);
    }

    cout << vec_ch[0][1] << endl;
    cout << vec_ch[1][1] << endl;
    return 0;
}
/*
 * ^^
 * ^^
 * ^^
 * ^^
 * CS222 test above
 */

int main(){
    int i = 0;
    for(; i < 10; i++);
    printf("%f\n", (float)1/2);
//    unordered_set<char> st;
//    st.insert('a');
//    st.insert('d');
//    st.insert('a');
//    unordered_set<char>::iterator it;
//    for(it = st.begin(); it != st.end(); ++it){
//        cout << "loop" << endl;
//    }
    string ss = "ssdfweee";
    bool b = true;
    // cout << b << endl;
    // cout << (bool)(st.find('x') == st.end()) << endl;

    cout << "------" << endl;

    string tmp = ss.substr(8);
    string tmp1 = tmp.substr(0);
    // cout << tmp1 << endl;

//    string words[5] = {"asd", "edd", "ee1", "dlf", "ssq"};
//    std::unordered_map<string, int> word_count = {{"Kevin", 1}};
//    for(const string& word : words) {
//        ++word_count[word];
//        //cout << word_count[word] << endl;
//    }
}
