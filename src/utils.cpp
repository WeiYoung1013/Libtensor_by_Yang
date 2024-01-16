//
// Created by user_9k7t0TZ11 on 2024/1/12.
//

#include <iterator>
#include <fstream>  // for the linux stuff
#include <iostream>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <new>      // included for std::bad_alloc
#include <string>
#include <limits>
#include <algorithm>
#include <sstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <stdexcept>
#include <cmath>
#include <vector>
#include <iomanip>
#include <limits>

#include "utils.h"

void msg(const string& text, const string& title){
    string s(text);
    if(!title.empty()){
        s += " (" + title + ")";
    }
    std::cerr << "==================================================================\n";
    std::cerr << "⚠️  " << s << " ⚠️"<<std::endl;
    std::cerr << "==================================================================\n\n";

    throw std::runtime_error("RuntimeError: " + title);
}

vector<int> shape2stride(const vector<int>& shape){
    vector<int> stride = {1};

    for(int i=shape.size()-1; i>0; i--){
        int s = shape[i];
        int s2 = stride[0];
        stride.insert(stride.begin(), s*s2);
    }

    return stride;
}

vector<int> indices2shape(vector<vector<int>> ranges){
    vector<int> shape;
    for(auto & range : ranges){
        shape.push_back(range[1]-range[0]+1);
    }
    return shape;
}

int shape2size(vector<int> shape){
    int size = 1;
    for(int i=0; i<shape.size(); i++){
        size *= shape[i];
    }
    return size;
}

int* ranges2indices(vector<int> ishape, vector<vector<int>> ranges){
    // Returns an array with the linear positions of the ranges to perform fast translations
    // [0:2, 5] {H=10, W=7}=> ([0,1], [5]) => (0*7+5),(1*7)+5,...

    // Compute output dimensions
    vector<int> istride = shape2stride(ishape);

    vector<int> oshape = indices2shape(ranges);
    vector<int> ostride = shape2stride(oshape);
    int osize = shape2size(oshape);
    int* addresses = new int[osize];  // Because the batch is 1 (default), then it's resized

    // For each output address (0,1,2,3,...n), compute its indices
    // Then add the minimum of each range, and compute the raw address
    for(int i=0; i<osize; i++) {

        // Extract indices
        int A_pos = 0;
        for(int d=0; d<ranges.size(); d++){
            // Compute output indices at dimension d
            int B_idx = (i/ostride[d]) % oshape[d];  // (52 / 32) % 32=> [1, 20]

            // Compute input indices at dimension d
            int A_idx = B_idx + ranges[d][0];  // B_index + A_start => [0, 0, 0] + [0, 5, 5]
            A_pos += A_idx * istride[d];
        }

        // Save address translation
        addresses[i] = A_pos;
    }

    return addresses;  // Be careful! It's easy to forget about this pointer and have a memory leak
}


vector<vector<int>> parse_indices(vector<string> str_indices, const vector<int>& shape){
    string delimiter(":");
    vector<vector<int>> ranges;

    // Shapes must match
    if(str_indices.size() != shape.size()){
        int diff = shape.size() - str_indices.size();
        if(diff>=0){
            for(int i=0; i<diff; i++){
                str_indices.emplace_back(":");
            }
        }else{
            msg( "The number of dimensions of the indices cannot be greater than the shape of the tensor to match", "utils::parse_indices");
        }
    }

    // Parse string indices
    for(int i=0; i<str_indices.size(); i++){
        int min, max;

        // Remove whitespaces
        string str = str_indices[i];
        std::string::iterator end_pos = std::remove(str.begin(), str.end(), ' ');
        str.erase(end_pos, str.end());

        // Find delimiters
        int pos = str.find(delimiter);
        if(pos != string::npos){ // Found
            if(str==delimiter){  // ":"
                min = 0;
                max = shape[i];
            }else{
                if (pos==0){ // ":5"
                    min = 0;
                    max = std::stoi(str.substr(pos+delimiter.length(), string::npos));  // Numpy style
                }else if(pos==str.length()-1){  // "5:"
                    min = std::stoi(str.substr(0, str.length()-delimiter.length()));
                    max = shape[i];
                }else{  // "5:10"
                    min = std::stoi(str.substr(0, pos - 0));  // (start_pos, len= end_pos-start_pos)
                    max = std::stoi(str.substr(pos+delimiter.length(), string::npos));  // Numpy style
                }
            }

            max -= 1;  // last index is not included
        }else{  // Not found => "5"
            min = std::stoi(str);
            max = min;
        }
        // Negative indices // len + (-x)
        if(min<0) { min = shape[i] + min; }
        if(max<0) { max = shape[i] + max; }

        ranges.push_back({min, max});
    }


    // Second check (negative values, max < min, or max > shape)
    for(int i=0; i<ranges.size(); i++){
        string common_str = "Invalid indices: '" + str_indices[i] + "'. ";
        if(ranges[i][0] < 0 || ranges[i][1] < 0){
            msg( common_str + "Indices must be greater than zero.", "utils::parse_indices");
        }else if(ranges[i][1] < ranges[i][0]){
            msg(common_str + "The last index of the range must be greater or equal than the first.", "utils::parse_indices");
        } else if(ranges[i][1] >= shape[i]){
            msg(common_str + "The last index of the range must fit in its dimension.", "utils::parse_indices");
        }
    }
    return ranges;
}

void fast_address2indices( int address, int* indices, const int* shape, const int* strides, int ndim){
    for(int i=0; i<ndim; i++) {
        indices[i] = address / strides[i] % shape[i];
    }
}

int fast_indices2address(const int* indices, const int* strides, int ndim){
    int address = 0;
    for (int i=0; i< ndim; i++){
        address += indices[i] * strides[i];
    }
    return address;
}


vector<vector<int>> cartesian_product(const vector<vector<int>>& vectors){
    vector<vector<int>> results = {{}};
    for (auto &vec : vectors){ // Vectors: {0, 1}, {5, 6, 7}, {8, 9}
        vector<vector<int>> temp;

        for(auto &res : results){  // Previous solution: {{0}, {1}}
            for(auto &elem : vec){  // Elements 1, 2, 3,...
                vector<int> new_vec = res;
                new_vec.push_back(elem);
                temp.push_back(new_vec);
            }
        }
        results.clear();
        results = temp;
    }
    return results;
}

vector<int> permute_shape(const vector<int>& ishape, const vector<int>& dims){
    vector<int> oshape;
    if(dims.size()!=ishape.size()){
        msg("Dimensions do not match", "utils::permute_indices");
    }else{
        for(auto &d : dims){
            oshape.emplace_back(ishape[d]);
        }
    }

    return oshape;
}

int* permute_indices(const vector<int>& ishape, const vector<int>& dims){
    int* addresses = nullptr;
    vector<int> oshape = permute_shape(ishape, dims);

    // Compute size_
    int isize = shape2size(ishape);
    int osize = shape2size(oshape);

    // Check if the shapes are compatible
    if (ishape.size() != oshape.size() || isize!=osize){
        msg("Incompatible dimensions", "utils::permute_indices");
    }else{
        vector<int> istride = shape2stride(ishape);
        vector<int> ostride = shape2stride(oshape);
        addresses = new int[isize];

        // For each output address (0,1,2,3,...n), compute its indices
        // Then add the minimum of each range, and compute the raw address
        for(int i=0; i<isize; i++) {

            // Extract indices
            int B_pos = 0;
            for(int d=0; d<ishape.size(); d++){
                // Compute output indices at dimension d, but permuted
                int A_idx = (i/istride[dims[d]]) % ishape[dims[d]];  // (52 / 32) % 32=> [1, 20]
                B_pos += A_idx * ostride[d];
            }

            // Save address translation
            addresses[B_pos] = i;
        }
    }

    return addresses;  // Be careful! It's easy to forget about this pointer and have a memory leak
}