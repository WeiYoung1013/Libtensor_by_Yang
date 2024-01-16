//
// Created by user_9k7t0TZ11 on 2024/1/11.
//

#include "cpu.h"

// CPU: Logic functions: Comparisons
bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    bool allclose = true;
    int first_idx = -1;

    for (int i = 0; i < A->size_; ++i){
        // Check if both values are NaN
        if (equal_nan && std::isnan(A->ptr[i]) && std::isnan(B->ptr[i])) {
            continue;
        }

        // Compare values
        bool close = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
        if (!close){
            {
                allclose = false;
                if(first_idx < 0) { first_idx=i; }
            }
        }
    }
    return allclose;
}


void cpu_fill_(Tensor *A, float v){
    for (int i = 0; i < A->size_; ++i){
        A->ptr[i] = v;
    }
}

void cpu_eye(Tensor *A, int offset){
    for(int i=0; i<A->size_; i++){
        if ((i/A->shape[0]+offset) == i%A->shape[1]){ A->ptr[i] = 1.0f; }  // rows+offset == col?
        else { A->ptr[i] = 0.0f; }
    }
}

void cpu_select(Tensor *A, Tensor *B, SelDescriptor *sd){

    for (int i = 0; i < B->size_; i++) {
        B->ptr[i] = A->ptr[sd->cpu_addresses[i]];
    }
}

void cpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative){
    // Walk through all the tensors to concat one axis (once)
    unsigned int offset = 0;
    unsigned int src_stride = 0;
    int steps = A->stride[axis] * A->shape[axis];  // Equivalent to A->stride[axis-1], but without the negative index problem

    // Walk through each tensor
    for (unsigned int i = 0; i < t.size(); i++) {
        offset += src_stride;
        src_stride = t[i]->stride[axis] * t[i]->shape[axis];

        // Copy n bytes from src to dest
        float *dest = A->ptr + offset;
        float *src = t[i]->ptr;

        // Walk tensor i
        for (int j = 0; j < t[i]->size_; j++) {
            unsigned int k = j % src_stride;  // Pos (index) in the stride (src)
            unsigned int stride_idx = j / src_stride;  // Index of the stride (src/dst)
            unsigned int dest_offset = stride_idx * steps;  // Offset in dest

            if(derivative){ src[j] += dest[dest_offset + k]; }
            else{ dest[dest_offset + k] = src[j]; }
        }
    }
}

void cpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){

    for (int i = 0; i < B->size_; i++) {
        A->ptr[sd->cpu_addresses[i]] = B->ptr[i];
    }
}

void cpu_transpose(Tensor * A, Tensor * B) {
    for (int i = 0; i < A->size_; i++){
        B->ptr[i] = A->ptr[i];
    }
}

void cpu_copy(Tensor * A, Tensor * B){
    for (int i = 0; i < A->size_; i++){
        B->ptr[i] = A->ptr[i];
    }
}

void cpu_add(Tensor *A, Tensor *B, float v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] + v;
    }

}

void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
}

void cpu_sub(Tensor *A, Tensor *B, float v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] - v;
    }

}

void cpu_sub(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] - scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] - scB * B->ptr[i];
}

void cpu_mul(Tensor *A, Tensor *B, float v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] * v;
    }

}

void cpu_mul(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] * scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] * scB * B->ptr[i];
}

void cpu_div(Tensor *A, Tensor *B, float v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] / v;
    }

}

void cpu_div(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] / scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] / scB * B->ptr[i];
}


float cpu_sum(Tensor *A) {
    return cpu_sum(A->ptr, A->size_, nullptr);
}


void cpu_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_sum(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

float cpu_sum(float *ptr, int size, int *map) {
    float sum = 0.0f;

    if(map == nullptr){
        for (int i = 0; i < size; ++i) { sum += ptr[i]; }
    }else{
        for (int i = 0; i < size; ++i) { sum += ptr[map[i]]; }
    }

    return sum;
}


float cpu_min(Tensor *A) {
    return cpu_min(A->ptr, A->size_, nullptr);
}


void cpu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_min(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

float cpu_min(float *ptr, int size, int *map) {
    float min_ = FLT_MAX;

    if(map == nullptr){
        for (int i = 0; i < size; ++i) {
            if(ptr[i]  < min_) {
                min_ = ptr[i];
            }
        }
    }else{
        for (int i = 0; i < size; ++i) {
            if(ptr[i]  < min_) {
                min_ = ptr[map[i]];
            }
        }
    }

    return min_;
}


float cpu_max(Tensor *A) {
    return cpu_max(A->ptr, A->size_, nullptr);
}


void cpu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_max(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}

float cpu_max(float *ptr, int size, int *map) {
    float max_ = FLT_MAX;

    if(map == nullptr){
        for (int i = 0; i < size; ++i) {
            if(ptr[i]  > max_) {
                max_ = ptr[i];
            }
        }
    }else{
        for (int i = 0; i < size; ++i) {
            if(ptr[i]  > max_) {
                max_ = ptr[map[i]];
            }
        }
    }

    return max_;
}



void cpu_equal(Tensor *A, Tensor *B, float v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = A->ptr[i] == v;
    }
}

void cpu_equal(Tensor *A, Tensor *B, Tensor *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = A->ptr[i] == B->ptr[i];
    }
}

void cpu_nequal(Tensor *A, Tensor *B, float v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = (A->ptr[i] != v);
    }
}

void cpu_nequal(Tensor *A, Tensor *B, Tensor *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = (A->ptr[i] != B->ptr[i]);
    }
}

void cpu_lequal(Tensor *A, Tensor *B, float v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = (A->ptr[i] < v);
    }
}

void cpu_lequal(Tensor *A, Tensor *B, Tensor *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = (A->ptr[i] < B->ptr[i]);
    }
}

void cpu_gequal(Tensor *A, Tensor *B, float v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = (A->ptr[i] > v);
    }
}

void cpu_gequal(Tensor *A, Tensor *B, Tensor *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = (A->ptr[i] > B->ptr[i]);
    }
}

void cpu_log(Tensor *A, Tensor *B) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::logf(A->ptr[i]);
}

void cpu_log2(Tensor *A, Tensor *B) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::log2f(A->ptr[i]);
}

void cpu_log10(Tensor *A, Tensor *B) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::log10f(A->ptr[i]);
}

void cpu_logn(Tensor *A, Tensor *B, float n) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::logf(A->ptr[i]) / ::logf(n);
}