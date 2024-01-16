//
// Created by user_9k7t0TZ11 on 2024/1/12.
//

#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H


#include <cstdint> // uint64_t
#include <vector>

using namespace std;

void msg(const string& text, const string& title);

vector<int> shape2stride(const vector<int>& shape);
vector<int> indices2shape(vector<vector<int>> ranges);
int shape2size(vector<int> shape);
int* ranges2indices(vector<int> ishape, vector<vector<int>> ranges);
vector<vector<int>> parse_indices(vector<string> str_indices, const vector<int>& shape);

int fast_indices2address(const int* indices, const int* strides, int ndim);
void fast_address2indices( int address, int* indices, const int* shape, const int* strides, int ndim);

vector<vector<int>> cartesian_product(const vector<vector<int>>& vectors);

vector<int> permute_shape(const vector<int>& ishape, const vector<int>& dims);
int* permute_indices(const vector<int>& ishape, const vector<int>& dims);

#endif //TENSOR_UTILS_H
