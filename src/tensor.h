//
// Created by user_9k7t0TZ11 on 2024/1/11.
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>

#include "utils.h"
#include "tensor_descriptors.h"

using namespace std;

class Tensor {
public:
    unsigned int ndim;
    unsigned long int size_;

    // 数据指针
    float *ptr = nullptr;

    // 张量的尺寸
    std::vector<int> shape;
    std::vector<int> stride;
    Tensor* grad = nullptr;
    std::vector<Tensor*> grad_history;


    Tensor();
    Tensor(const std::vector<int> &shape);
    Tensor(const std::vector<float>& data, const std::vector<int> &shape);
    ~Tensor();

    vector<int> getShape(void);
    std::string size();
    std::string type();
    void* data_ptr();

    void updateData(void);
    void updateSize(void);
    void updateShape(const std::vector<int> &new_shape);
    void deleteData(void);
    void updateStrides();
    void computeGradient();
    Tensor* getGradient();
    void updateGradient(Tensor* new_grad);
    Tensor* getCurrentGradient();
    Tensor *getGradientHistory(int index);


    void print(int precision = -6, bool raw = false);

    static bool sameSize(Tensor *A, Tensor *B);
    static int sameShape(Tensor *A, Tensor *B);
    static int equivalent(Tensor *A, Tensor *B, float atol, float rtol, bool equal_nan);

    void fill_(float v = 0);
    void fill(Tensor* A, float v = 0);

    static Tensor* empty(const std::vector<int> &shape);
    static Tensor* empty_like(Tensor *A);

    static Tensor* zeros(const std::vector<int> &shape);
    static Tensor* ones(const std::vector<int> &shape);

    static Tensor* full(const std::vector<int> &shape, float value );

    static Tensor* eye(int rows, int offset = 0);
    static Tensor* rand(const std::vector<int> &shape, float v = 1.0);

    Tensor* select(const vector<string>& indices);
    static void select(Tensor *A, Tensor *B, SelDescriptor *sd);

    static Tensor* concat(const vector<Tensor*> A, unsigned int axis = 0, Tensor* output = nullptr);

    static Tensor* tile(Tensor* A, const vector<int>& repeats);

    void set_select(const vector<string>& indices, float value);
    void set_select(const vector<string>& indices, Tensor *A);
    static void set_select(Tensor *A, Tensor *B, SelDescriptor *sd);

    static void transpose(Tensor *A, Tensor *B, vector<int> dims);


    void permute_(const vector<int>& dims);
    Tensor* permute(const vector<int>& dims);
    static Tensor* permute(Tensor* A, const vector<int>& dims);

    static void copy(Tensor *A, Tensor *B);

    void reshape_(const vector<int> &new_shape);
    Tensor* reshape(const vector<int> &new_shape);
    static Tensor* reshape(Tensor *A, const vector<int> &shape);

    Tensor* clone();


    //---------------

    void add_(float v);
    Tensor* add(float v);

    void add_(Tensor* A);
    Tensor* add(Tensor* A);

    static void add(Tensor *A, Tensor *B, float v);
    static Tensor* add(Tensor *A, Tensor *B);
    static void add(Tensor *A, Tensor *B, Tensor *C);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
    static void add(Tensor *A, Tensor *B, Tensor *C, float scale);

    friend Tensor& operator+ (Tensor &A, Tensor &B);
    friend Tensor& operator+ (Tensor &A, float v);
    friend Tensor& operator+ (float v, Tensor &A);

    friend void operator+= (Tensor &A, Tensor &B);
    friend void operator+= (Tensor &A, float v);

    //sub

    void sub_(float v);
    Tensor* sub(float v);

    void sub_(Tensor* A);
    Tensor* sub(Tensor* A);

    static void sub(Tensor *A, Tensor *B, float v);
    static Tensor* sub(Tensor *A, Tensor *B);
    static void sub(Tensor *A, Tensor *B, Tensor *C);
    static void sub(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

    friend Tensor& operator- (Tensor &A, Tensor &B);
    friend Tensor& operator- (Tensor &A, float v);
    friend Tensor& operator- (float v, Tensor &A);

    friend void operator-= (Tensor &A, Tensor &B);
    friend void operator-= (Tensor &A, float v);


    //mul

    void mul_(float v);
    Tensor* mul(float v);

    void mul_(Tensor* A);
    Tensor* mul(Tensor* A);

    static void mul(Tensor *A, Tensor *B, float v);
    static Tensor* mul(Tensor *A, Tensor *B);
    static void mul(Tensor *A, Tensor *B, Tensor *C);
    static void mul(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

    friend Tensor& operator* (Tensor &A, Tensor &B);
    friend Tensor& operator* (Tensor &A, float v);
    friend Tensor& operator* (float v, Tensor &A);

    friend void operator*= (Tensor &A, Tensor &B);
    friend void operator*= (Tensor &A, float v);

    // div
    void div_(float v);
    Tensor* div(float v);

    void div_(Tensor* A);
    Tensor* div(Tensor* A);

    static void div(Tensor *A, Tensor *B, float v);
    static Tensor* div(Tensor *A, Tensor *B);
    static void div(Tensor *A, Tensor *B, Tensor *C);
    static void div(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

    friend Tensor& operator/ (Tensor &A, Tensor &B);
    friend Tensor& operator/ (Tensor &A, float v);
    friend Tensor& operator/ (float v, Tensor &A);

    friend void operator/= (Tensor &A, Tensor &B);
    friend void operator/= (Tensor &A, float v);




    //friend Tensor& operator- (Tensor &A, Tensor &B);
    //friend Tensor& operator* (Tensor &A, Tensor &B);
    //friend Tensor& operator/ (Tensor &A, Tensor &B);

    // log

    void log_();
    Tensor* log();
    static void log(Tensor *A, Tensor *B);
    void log2_();
    Tensor* log2();
    static void log2(Tensor *A, Tensor *B);
    void log10_();
    Tensor* log10();
    static void log10(Tensor *A, Tensor *B);
    void logn_(float n);
    Tensor* logn(float n);
    static void logn(Tensor *A, Tensor *B, float n);


    // sum

    float sum();
    static float sum(Tensor* A);
    Tensor* sum(vector<int> axis, bool keepdims);
    static void sum(Tensor* A, Tensor *B, ReduceDescriptor2 *rd);

    // equal_
    void equal_(float v);
    Tensor* equal(float v);
    static void equal(Tensor *A, Tensor *B, float v);
    Tensor* equal(Tensor *A);
    static void equal(Tensor *A, Tensor *B, Tensor *C);

    // not equal_
    void nequal_(float v);
    Tensor* nequal(float v);
    static void nequal(Tensor *A, Tensor *B, float v);
    Tensor* nequal(Tensor *A);
    static void nequal(Tensor *A, Tensor *B, Tensor *C);

    // less equal_
    void lequal_(float v);
    Tensor* lequal(float v);
    static void lequal(Tensor *A, Tensor *B, float v);
    Tensor* lequal(Tensor *A);
    static void lequal(Tensor *A, Tensor *B, Tensor *C);

    // great equal_
    void gequal_(float v);
    Tensor* gequal(float v);
    static void gequal(Tensor *A, Tensor *B, float v);
    Tensor* gequal(Tensor *A);
    static void gequal(Tensor *A, Tensor *B, Tensor *C);





    void  save(const string& filename);
    Tensor*  load(const string& filename);


    friend std::ostream &operator<<(std::ostream &os, Tensor &t);
    //friend ostream &operator<<(ostream &out, complex &A);

    static Tensor *negate(Tensor *A);
};



#endif //TENSOR_TENSOR_H
