//
// Created by user_9k7t0TZ11 on 2024/1/11.
//

#ifndef TENSOR_CPU_H
#define TENSOR_CPU_H

#include <cmath>
#include <cfloat>

#include "tensor.h"

#include "tensor_descriptors.h"

// CPU: Logic functions: Comparisons
// 比较
bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan);

void cpu_fill_(Tensor *A, float v);

void cpu_eye(Tensor *A, int offset);

void cpu_select(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative);

void cpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_transpose(Tensor * A, Tensor * B) ;

void cpu_copy(Tensor * A, Tensor * B);

void cpu_add(Tensor *A, Tensor *B, float v);
void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

void cpu_sub(Tensor *A, Tensor *B, float v);
void cpu_sub(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

void cpu_mul(Tensor *A, Tensor *B, float v);
void cpu_mul(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

void cpu_div(Tensor *A, Tensor *B, float v);
void cpu_div(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

void cpu_log(Tensor *A, Tensor *B) ;
void cpu_log2(Tensor *A, Tensor *B) ;
void cpu_log10(Tensor *A, Tensor *B) ;
void cpu_logn(Tensor *A, Tensor *B, float n) ;

float cpu_sum(Tensor *A);
void cpu_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_sum(float *ptr, int size, int *map);

float cpu_max(Tensor *A);
void cpu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_max(float *ptr, int size, int *map);

float cpu_min(Tensor *A);
void cpu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_min(float *ptr, int size, int *map);

void cpu_equal(Tensor *A, Tensor *B, float v);
void cpu_equal(Tensor *A, Tensor *B, Tensor *C);

void cpu_nequal(Tensor *A, Tensor *B, float v);
void cpu_nequal(Tensor *A, Tensor *B, Tensor *C);

void cpu_lequal(Tensor *A, Tensor *B, float v);
void cpu_lequal(Tensor *A, Tensor *B, Tensor *C);

void cpu_gequal(Tensor *A, Tensor *B, float v);
void cpu_gequal(Tensor *A, Tensor *B, Tensor *C);

#endif //TENSOR_CPU_H
