//
// Created by user_9k7t0TZ11 on 2024/1/11.
//

#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H
#include <cmath>
#include <cfloat>

#include "tensor.h"

#include "tensor_descriptors.h"
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
#include <fstream>

#include "utils.h"
#include "tensor_descriptors.h"


using namespace std;
template <typename T>
T cpu_sum(T*ptr, int size, int *map);
template <typename T>
class Tensor {
public:
    unsigned int ndim;
    unsigned long int size_;

    // 数据指针
    T *ptr = nullptr;

    // 张量的尺寸
    std::vector<int> shape;
    std::vector<int> stride;
    template <typename u>
    friend Tensor<u>& operator- (Tensor<u> &A, Tensor<u> &B);
    template <typename u>
    friend Tensor<u>& operator- (Tensor <u>&A, u v);
    template <typename u>
    friend Tensor<u>& operator- (u v, Tensor<u> &A);
    static  T sum(Tensor<T>* A){
        return cpu_sum(A);
    }

    static void sum(Tensor<T>* A, Tensor<T> *B, ReduceDescriptor2 *rd){
        cpu_sum(A, B, rd);

    }
    Tensor<T>*sum(vector<int> axis, bool keepdims){
        // Build descriptor
        auto rd = new ReduceDescriptor2(axis, keepdims );
        rd->build(this->shape);

        // Create output tensor
        Tensor *t = Tensor::empty(rd->oshape );
        Tensor::sum(this, t, rd);

        delete rd;
        return t;
    }
    T sum(){
        return Tensor<T>::sum(this);
    }

    template <typename u>
    friend void operator-= (Tensor<u> &A, Tensor<u> &B);
    template <typename u>
    friend void operator-= (Tensor<u> &A, u v);
    template <typename u>
    friend Tensor<u>& operator+ (Tensor<u> &A, Tensor<u> &B);
    template <typename u>
    friend Tensor<u>& operator+ (Tensor<u> &A, u v);
    template <typename u>
    friend Tensor<u>& operator+ (u v, Tensor<u> &A);
    template <typename u>
    friend void operator+= (Tensor<u> &A, Tensor <u>&B);
    template <typename u>
    friend void operator+= (Tensor<u> &A, u v);
    template <typename u>
    friend Tensor<u>& operator* (Tensor<u> &A, Tensor<u> &B);
    template <typename u>
    friend Tensor<u>& operator* (Tensor<u> &A, u v);
    template <typename u>
    friend Tensor<u>& operator* (u v, Tensor<u> &A);
    template <typename u>
    friend void operator*= (Tensor<u> &A, Tensor <u>&B);
    template <typename u>
    friend void operator*= (Tensor<u> &A, u v);
    template <typename u>
    friend Tensor<u>& operator/ (Tensor <u>&A, Tensor<u> &B);
    template <typename u>
    friend Tensor<u>& operator/ (Tensor<u>&A, u v);
    template <typename u>
    friend Tensor<u>& operator/ (u v, Tensor <u>&A);
    template <typename u>
    friend void operator/= (Tensor<u> &A, Tensor<u> &B);
    template <typename u>
    friend void operator/= (Tensor<u> &A, u v);
    template <typename u>
    friend std::ostream &operator<<(std::ostream &os, Tensor<u> &t);

    //
// Created by user_9k7t0TZ11 on 2024/1/11.
//


// 1.1 无参构造函数
    Tensor() : ndim(0), size_(0) {}
// 1.2 传入 shape 每个维度的大小
    Tensor(const std::vector<int> &shape) {
        // Tensor(shape, nullptr, dev)
        updateShape(shape);
        updateSize();
        updateStrides();
        updateData();
    }

// 1.3 传入 一维度的data做数据 传入
// shape 每个维度的大小

    Tensor(const std::vector<T>& data, const std::vector<int> &shape) {
        updateShape(shape);
        updateSize();
        updateStrides();
        updateData();
        std::copy(data.begin(), data.end(), this->ptr);
    }


// 析构函数
    ~Tensor() {
        this->deleteData();
    }

    vector<int> getShape(void) {
        return vector<int>(this->shape);
    }

// 更新张量 size_ 维度
    void updateSize() {
        this->size_ = 1;

        for(auto &d : this->shape) {
            this->size_ = this->size_ * d;
        }
    }

    void updateData() {
        if(this->ptr == nullptr){
            this->ptr = (T *)malloc(size_ * sizeof(T));
        }
    }

// 更新张量  shape  每个维度的大小
    void updateShape(const std::vector<int> &new_shape){
        // this->shape = vector<int>(new_shape);
        this->shape.clear();
        for (int _ : new_shape) this->shape.push_back(_);
        this->ndim = this->shape.size();
    }

    void updateStrides() {
        this->stride.clear();  // Remove all elements

        unsigned long int new_size = this->size_;
        for(int i=0;i<ndim;i++) {
            new_size /= shape[i];
            this->stride.push_back(new_size);
        }
    }

    void deleteData(){
        if(this->ptr != nullptr){
            free(this->ptr);
            this->ptr = nullptr;
        }
    }

// 判断一样的size大小

static    bool sameSize(Tensor *A, Tensor *B) {
        return A->size_ == B->size_;
    }

// 判断一样的维度结构

   static  int sameShape(Tensor *A, Tensor *B) {
        if (A->ndim != B->ndim) return 0;

        for (int i = 0; i < A->ndim; i++){
            if (A->shape[i] != B->shape[i]) return 0;
        }

        return 1;
    }

//-----------------------

//打印函数

/**
  *  @brief Prints the content of the tensor
  *
  *  @param precision Number of decimals places to use
  *  @param raw  Print the tensor without format
*/

    void print(int precision =-6 , bool raw= false  ) {
        size_t opened = 0;
        size_t closed = 0;

        // Clone to CPU (if needed)
        Tensor *aux = nullptr;
        aux = this;

        // ***** Shitty code to prettify the output *******
        std::stringstream buffer;
        buffer << std::fixed;
        buffer << std::setprecision(precision);

        int lines = 0;
        int max_lines = 100000;
        for (int i = 0; i < aux->size_; ++i) {
            if(i % this->stride[0]==0){lines++;}

            if(raw){
                // Print number
                buffer << aux->ptr[i] << ", ";

            }else{

                // Open brackets
                opened = 0;
                for (int j = 0; j < aux->ndim-1; ++j) {
                    if(i%aux->stride[j]==0){
                        if(!opened && closed==1){ if(ndim==2){ buffer << "\n"; } else { buffer << " "; } }
                        buffer << "[";
                        opened += 1;
                    }
                }

                // Print number
                buffer << aux->ptr[i];

                // Close brackets
                closed = 0;
                for (int j = 0; j < aux->ndim-1; ++j) {
                    if((i+1)%aux->stride[j]==0) {
                        buffer << "]";
                        closed += 1;
                    }
                }

                // Break lines
                if (i+1 < aux->size_){
                    if(!closed){ buffer << " ";}
                    else{
                        if (closed == 2 ) {  buffer << "\n"; }
                        else if (closed == 3) { buffer << "\n\n"; }
                        else if (closed > 3) { buffer << "\n\n\n"; }
                    }
                }

                // Stop
                if(lines >= max_lines){
                    std::cout << "Maximum tensor length exceeded." << std::endl;
                    std::cout << "Printing only first " << max_lines << " rows:" << std::endl;
                    break;
                }

            }

        }

        // Print to buffer
        if(aux->ndim>1){
            std::cout << "[\n" << buffer.str() << "\n]" << std::endl;  // For readability
        }else{
            std::cout << "[" << buffer.str() << "]" << std::endl;
        }

    }


//--------------------------------------------------------------------------------------------------------------

 static   Tensor<T>* empty(const std::vector<int> &shape){
        return new Tensor(shape);
    }

  static  Tensor<T>* empty_like(Tensor *A){
        return Tensor::empty(A->shape);
    }

// 1.3 zeros

    static Tensor<T>* zeros(const std::vector<int> &shape ){
        auto t = new Tensor(shape);
        t->fill_(0.0f);
        return t;
    }

//Tensor* Tensor::zeros_like(Tensor *A){
//    return Tensor::zeros(A->shape );
//}

// 1.3 ones

    static Tensor<T>* ones(const std::vector<int> &shape ){
        auto t = new Tensor(shape);
        t->fill_(1.0f);
        return t;
    }
//
//Tensor* Tensor::ones_like(Tensor *A){
//    return Tensor::ones(A->shape);
//}
//

// 1.3 full

    static Tensor<T>* full(const std::vector<int> &shape, T value ){
        auto t = new Tensor(shape );
        t->fill_(value);
        return t;
    }
//
//Tensor* Tensor::full_like(Tensor *A, float value){
//    return Tensor::full(A->shape, value );
//}
//

    void fill_(T v) {
        Tensor::fill(this, v);
    }
//
//Tensor* Tensor::fill(float v){
//    Tensor* t_new = Tensor::empty_like(this);
//    Tensor::fill(t_new, v);
//    return t_new;
//}

    void fill(Tensor* A, T v){
        cpu_fill_(A, v);
    }

//----------------------------------------------------------------------------------------------------------------------

// 判断是否相等
    int equivalent(Tensor *A, Tensor *B, T atol, T rtol, bool equal_nan) {
        // Equal ndims and shapes
        if (!sameShape(A, B)) {
            return 0;
        }

        return cpu_allclose(A, B, rtol, atol, equal_nan);
    }

//----------------------------------------------------------------------------------------------------------------------

// 1.4
    static Tensor<T>* eye(int rows, int offset=0){
        auto t = new Tensor(std::vector<int>{rows, rows});
        cpu_eye(t, offset);
        return t;
    }

// std::srand(std::time(nullptr));

    static Tensor<T>* rand(const std::vector<int> &shape, T v=1.0){
        auto A = new Tensor(shape);
        for (int i = 0; i < A->size_; ++i) {
            float uniform = static_cast<float>(std::rand()) / RAND_MAX;
            A->ptr[i] = uniform * v;
        }

        return A;
    }

// index Indexing and slicing

    Tensor<T>* select(const vector<string>& indices){
        // Build descriptor
        auto *sd = new SelDescriptor(indices);
        sd->build(this->shape);

        // Initialize tensor
        auto* t = new Tensor(sd->oshape);

        // Perform select
        Tensor::select(this, t, sd);

        delete sd;
        return t;
    }
 static void select(Tensor *A, Tensor* B, SelDescriptor *sd){
        cpu_select(A, B, sd);
    }

//---

// 2.2 拼接

    static Tensor<T>* concat(const vector<Tensor*> A, unsigned int axis=0, Tensor* output= nullptr){
        // Check number of vectors to concat
        if(A.size()<2){
            msg("Concat requires a minimum of two tensors", "Tensor::concat");
        }

        // Temp variables
        vector<int> new_shape = A[0]->shape;
        int new_axis = 0;

        // Walk through each tensor to check for compatibility issues (from 1 to n)
        for(int i=1; i<A.size(); i++){

            // Check dimensions
            if(A[0]->ndim != A[i]->ndim){
                msg("The number of dimensions of all tensors must match (" +
                    to_string(A[0]->ndim) +  "!=" + to_string(A[i]->ndim) + ")", "Tensor::concat");
            }


            // Check that all dimensions match except the one to concat
            for(int j=0; j<A[0]->shape.size(); j++) {

                // Check current dimension
                if (j!=axis && A[0]->shape[j] != A[i]->shape[j]) {
                    msg("The dimensions across of all tensors must match (" +
                        to_string(A[0]->shape[j]) +  "!=" + to_string(A[i]->shape[j]) + ")", "Tensor::concat");
                }
            }

            // Sum dimension
            new_axis += A[i]->shape[axis];
        }

        // Update final shape
        new_shape[axis] +=  new_axis; // new_shape[axis] had the shape of the first tensor

        // Create new tensor
        if(output==nullptr){
            output = new Tensor(new_shape);
        }else{
            // Check dimensions
            if(output->shape!=new_shape){
                msg("The dimension of the output tensor is incorrect", "Tensor::concat");
            }
        }

        cpu_concat(output, A, axis, false);

        return output;
    }

    static Tensor<T>* tile(Tensor* A, const vector<int>& repeats){
        // Check dimensions
        if(A->ndim != repeats.size()){
            msg("The number of dimensions in tensor 'A' must match the size_ of 'repeats'", "Tensor::tile");
        }

        // Dimensions must be positive
        for(int i=0; i<repeats.size(); i++){
            if(repeats[i] < 1){
                msg("All repetitions must be greater or equal than 1", "Tensor::tile");
            }
        }

        // Build descriptor
        auto *td = new TileDescriptor(repeats );
        td->build(A->shape);

        // Initialize tensor
        auto* new_t = new Tensor(td->oshape );
        Tensor a;
        // Perform select
        a.select(A, new_t, td);

        delete td;
        return new_t;
    }


// ---
// 2.3 Mutating operations

    void set_select(const vector<string>& indices, T value){
        auto *sd = new SelDescriptor(indices);
        sd->build(this->shape);

        Tensor* A = Tensor::full(sd->oshape, value);

        // Check if the dimensions of the selection and the tensor are compatibles
        if(sd->oshape==A->shape){
            Tensor::set_select(this, A, sd);
        }else{

            msg("Incompatible dimensions", "Tensor::set_select");
        }

        delete A;
        delete sd;
    }

    void set_select(const vector<string>& indices, Tensor *A){
        auto *sd = new SelDescriptor(indices );
        sd->build(this->shape);

        // Check if the dimensions of the selection and the tensor are compatibles
        if(sd->oshape==A->shape){
            Tensor::set_select(this, A, sd);
        }else{
            //info();
            //A->info();

            msg("Incompatible dimensions", "Tensor::set_select");
        }

        delete sd;
    }

    void set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
        cpu_set_select(A, B, sd);
    }

    void transpose(Tensor *A, Tensor *B, vector<int> dims) {
        // TODO: Deprecated.
        // Transpose

        if (A->size_ != B->size_)
            msg("Tensors with different size_", "Tensor::transpose");

        Tensor *N;
        if (A == B) N = new Tensor(A->getShape());
        else N = B;


        // Copy tensor data
        cpu_transpose(A, N);

        if (A == B) delete N;

    }

    void permute_(const vector<int>& dims){
        Tensor* temp = Tensor::permute(this, dims);

        // Update attributes
        updateShape(temp->shape);
        updateSize();
        updateStrides();
        Tensor::copy(temp, this);  // copy data

        delete temp;
    }

     Tensor<T>* permute(const vector<int>& dims){
        Tensor* t_new = Tensor::permute(this, dims);
        return t_new;
    }

    static  Tensor<T>* permute(Tensor* A, const vector<int>& dims){
        // Build descriptor
        auto *sd = new PermuteDescriptor(dims );
        sd->build(A->shape);

        // Initialize new tensor
        auto *new_t = new Tensor(sd->oshape );

        // Fill new tensor
        Tensor::select(A, new_t, sd);

        delete sd;
        return new_t;
    }

    void copy(Tensor *A, Tensor *B) {
        ///////////////////////////////////////
        /// Copy from A to B
        //////////////////////////////////////

        if (!Tensor::sameSize(A, B)) {
            msg("Tensors with different size_", "Tensor::copy");
        }


        cpu_copy(A, B);

    }

    void reshape_(const vector<int> &new_shape){
        int new_size = 1;  // For checking
        vector<int> final_shape;

        // Compute new shape (infer if necessary)
        for(auto d : new_shape) {
            if(d==-1){  // Infer the remaining dimensions
                d = this->size_ / new_size;
            }
            final_shape.push_back(d);
            new_size *= d;
        }

        // Check if the new size_ is compatible
        if(new_size!=this->size_){
            cout << new_size << "!=" << size_ << endl;
            msg("Not compatible shapes", "Tensor::reshape_");
        }

        // Update attributes
        updateShape(final_shape);
        updateSize();
        updateStrides();
        //updateData(this->ptr, nullptr);  // Due to potential Eigen mapping when CPU and dim=2
    }

    Tensor<T>* reshape(const vector<int> &new_shape){
        Tensor *t_new = Tensor::reshape(this, new_shape);
        return t_new;
    }

    Tensor<T>* reshape(Tensor *A, const vector<int> &shape){
        Tensor *t_new = A->clone();
        t_new->reshape_(shape);
        return t_new;
    }

    Tensor<T>* clone(){
        auto* t_new = new Tensor(this->shape);
        Tensor::copy(this, t_new);
        return t_new;
    }


// add
      void add_(T v){
        Tensor::add(this, this, v);
    }

   Tensor<T>* add(T v){
        Tensor *t = this->clone();
        t->add_(v);
        return t;
    }

   void add_(Tensor* A){
        Tensor::add(this, A, this);
    }

 Tensor<T>* add(Tensor* A){
        Tensor *t = this->clone();
        t->add_(A);
        return t;
    }

    static     void add(Tensor *A, Tensor *B, T v){
        cpu_add(A, B, v);
    }

    static   Tensor<T>*add(Tensor *A, Tensor *B){
        Tensor* C = Tensor::empty(A->getShape() );
        Tensor::add(A, B, C);
        return C;
    }

    static    void add(Tensor *A, Tensor *B, Tensor *C) {
        Tensor::add(1.0, A, 1.0, B, C, 0);
    }

    static  void add(T scA, Tensor *A, T scB, Tensor *B, Tensor *C, int incC) {
        ///////////////////////////////////////
        //// sum C=(sca*A)+(scb*B)
        //// or C+=(sca*A)+(scb*B) if incC is 1
        //// Dimensions and types must be compatible
        ///////////////////////////////////////
        int aux = 0;

        if ((!sameShape(A, B)) || (!sameShape(A, C))) {
            msg("Incompatible dims", "Tensor::add");
        }
        cpu_add(scA, A, scB, B, C, incC);
    }


// ------------------------------
// sub
    void sub_(T v){
        sub(this, this, v);
    }
    Tensor<T>* sub(T v){
        Tensor *t = this->clone();
        t->sub_(v);
        return t;
    }

    void sub_(Tensor* A){
        Tensor::sub(this, A, this);
    }

    Tensor<T>* sub(Tensor* A){
        Tensor *t = this->clone();
        t->sub_(A);
        return t;
    }

  static  void sub(Tensor *A, Tensor *B, T v){
        cpu_sub(A, B, v);
    }

    static  Tensor<T>* sub(Tensor *A, Tensor *B){
        Tensor* C = Tensor::empty(A->getShape() );
        Tensor::sub(A, B, C);
        return C;
    }

    static   void sub(Tensor *A, Tensor *B, Tensor *C) {
        Tensor::sub(1.0, A, 1.0, B, C, 0);
    }

    static  void sub(T scA, Tensor *A, T scB, Tensor *B, Tensor *C, int incC) {
        ///////////////////////////////////////
        //// sum C=(sca*A)+(scb*B)
        //// or C+=(sca*A)+(scb*B) if incC is 1
        //// Dimensions and types must be compatible
        ///////////////////////////////////////
        int aux = 0;

        if ((!sameShape(A, B)) || (!sameShape(A, C))) {
            msg("Incompatible dims", "Tensor::sub");
        }
        cpu_sub(scA, A, scB, B, C, incC);
    }

// - *3


// -----
// mul

    void mul_(T v){
        Tensor::mul(this, this, v);
    }

    Tensor<T>* mul(T v){
        Tensor *t = this->clone();
        t->mul_(v);
        return t;
    }

    void mul_(Tensor* A){
        Tensor::mul(this, A, this);
    }

    Tensor<T>* mul(Tensor* A){
        Tensor *t = this->clone();
        t->mul_(A);
        return t;
    }

    static  void mul(Tensor *A, Tensor *B, T v){
        cpu_mul(A, B, v);
    }

    static  Tensor<T>* mul(Tensor *A, Tensor *B){
        Tensor* C = Tensor::empty(A->getShape() );
        Tensor::mul(A, B, C);
        return C;
    }

    static  void mul(Tensor *A, Tensor *B, Tensor *C) {
        Tensor::mul(1.0, A, 1.0, B, C, 0);
    }

    static void mul(T scA, Tensor *A, T scB, Tensor *B, Tensor *C, int incC) {
        ///////////////////////////////////////
        //// sum C=(sca*A)+(scb*B)
        //// or C+=(sca*A)+(scb*B) if incC is 1
        //// Dimensions and types must be compatible
        ///////////////////////////////////////
        int aux = 0;

        if ((!sameShape(A, B)) || (!sameShape(A, C))) {
            msg("Incompatible dims", "Tensor::mul");
        }
        cpu_mul(scA, A, scB, B, C, incC);
    }


// -----div

    void div_(T v){
        Tensor::div(this, this, v);
    }

    Tensor<T>* div(T v){
        Tensor *t = this->clone();
        t->div_(v);
        return t;
    }

    void div_(Tensor* A){
        Tensor::div(this, A, this);
    }

    Tensor<T>* div(Tensor* A){
        Tensor *t = this->clone();
        t->div_(A);
        return t;
    }

    static   void div(Tensor *A, Tensor *B, T v){
        cpu_div(A, B, v);
    }

    static  Tensor<T>* div(Tensor *A, Tensor *B){
        Tensor* C = Tensor::empty(A->getShape() );
        Tensor::div(A, B, C);
        return C;
    }

    static    void div(Tensor *A, Tensor *B, Tensor *C) {
        Tensor::div(1.0, A, 1.0, B, C, 0);
    }
    static  void div(T scA, Tensor *A, T scB, Tensor *B, Tensor *C, int incC) {
        ///////////////////////////////////////
        //// sum C=(sca*A)+(scb*B)
        //// or C+=(sca*A)+(scb*B) if incC is 1
        //// Dimensions and types must be compatible
        ///////////////////////////////////////
        int aux = 0;

        if ((!sameShape(A, B)) || (!sameShape(A, C))) {
            msg("Incompatible dims", "Tensor::div");
        }
        cpu_div(scA, A, scB, B, C, incC);
    }

// - *3

//

// -----log

    void log_(){
       log(this, this);
    }


    Tensor<T>*log(){
        Tensor *t = this->clone();
        t->log_();
        return t;
    }


    void log(Tensor *A, Tensor *B){
        cpu_log(A, B);
    }


    void log2_(){
        log2(this, this);
    }


    Tensor<T>* log2(){
        Tensor *t = this->clone();
        t->log2_();
        return t;
    }


  static  void log2(Tensor *A, Tensor *B){
        cpu_log2(A, B);
    }


    void log10_(){
    log10(this, this);
    }


    Tensor<T>* log10(){
        Tensor *t = this->clone();
        t->log10_();
        return t;
    }


    static void log10(Tensor *A, Tensor *B){
        cpu_log10(A, B);
    }


    void logn_(float n){
        Tensor::logn(this, this, n);
    }


    Tensor<T>* logn(float n){
        Tensor *t = this->clone();
        t->logn_(n);
        return t;
    }


    static  void logn(Tensor *A, Tensor *B, T n){
        cpu_logn(A, B, n);
    }

// ------------------------------




//


    void equal_(T v){
      equal(this, this, v);
    }

    Tensor<T>* equal(T v){
        Tensor *t = this->clone();
        t->equal_(v);
        return t;
    }

    static  void equal(Tensor *A, Tensor *B, T v){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::equal");
        }
        cpu_equal(A, B, v);
    }

    Tensor<T>* equal(Tensor *A){
        Tensor *t = Tensor::empty_like(this);
        t->equal(this, A, t);
        return t;
    }

    static void equal(Tensor *A, Tensor *B, Tensor *C){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::equal");
        }
        if (!Tensor::sameShape(A, C)){
            msg("Tensors with different shape", "Tensor::equal");
        }
        cpu_equal(A, B, C);
    }

// no equal


    void nequal_(T v){
        nequal(this, this, v);
    }

    Tensor<T>* nequal(T v){
        Tensor *t = this->clone();
        t->nequal_(v);
        return t;
    }

    static  void nequal(Tensor *A, Tensor *B, T v){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::nequal");
        }
        cpu_nequal(A, B, v);
    }
    Tensor<T>* nequal(Tensor *A){
        Tensor *t = Tensor::empty_like(this);
        t->nequal(this, A, t);
        return t;
    }

    static  void nequal(Tensor *A, Tensor *B, Tensor *C){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::nequal");
        }
        if (!Tensor::sameShape(A, C)){
            msg("Tensors with different shape", "Tensor::nequal");
        }
        cpu_nequal(A, B, C);
    }


//


    void lequal_(T v){
       lequal(this, this, v);
    }

    Tensor<T>* lequal(T v){
        Tensor *t = this->clone();
        t->lequal_(v);
        return t;
    }

    static void lequal(Tensor *A, Tensor *B, T v){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::lequal");
        }
        cpu_lequal(A, B, v);
    }

    Tensor<T>* lequal(Tensor *A){
        Tensor *t = Tensor::empty_like(this);
        t->lequal(this, A, t);
        return t;
    }

    static   void lequal(Tensor *A, Tensor *B, Tensor *C){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::lequal");
        }
        if (!Tensor::sameShape(A, C)){
            msg("Tensors with different shape", "Tensor::lequal");
        }
        cpu_lequal(A, B, C);
    }


    void gequal_(T v){
        gequal(this, this, v);
    }

    Tensor<T>* gequal(T v){
        Tensor *t = this->clone();
        t->gequal_(v);
        return t;
    }

 static   void gequal(Tensor *A, Tensor *B, T v){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::gequal");
        }
        cpu_gequal(A, B, v);
    }

    Tensor<T>*gequal(Tensor *A){
        Tensor *t = Tensor::empty_like(this);
        t->gequal(this, A, t);
        return t;
    }

 static   void gequal(Tensor *A, Tensor *B, Tensor *C){
        if (!Tensor::sameShape(A, B)){
            msg("Tensors with different shape", "Tensor::gequal");
        }
        if (!Tensor::sameShape(A, C)){
            msg("Tensors with different shape", "Tensor::gequal");
        }
        cpu_gequal(A, B, C);
    }


//

//bool pathExists(const std::string &s) {
//    struct stat buffer;
//    return (stat (s.c_str(), &buffer) == 0);
//}

    void save(const string& filename) {
        //string folder = filename.substr(0, filename.find_last_of("\\/"));
        //if(folder != filename && !pathExists(folder)){
        //    msg("The file could not be saved. Check if the directory exists or if you have permissions to write in it.", "Tensor::save");
        //}

        std::ofstream ofs(filename, std::ios::out | std::ios::binary);
        //Tensor::savefs(ofs, format);
        ofs.write(reinterpret_cast<const char *>(&this->ndim), sizeof(int));
        // Save dimensions
        ofs.write(reinterpret_cast<const char *>(this->shape.data()), this->shape.size() * sizeof(int));
        // Save content (row-major)
        ofs.write(reinterpret_cast<const char *>(this->ptr), this->size_ * sizeof(float));
        ofs.close();

    }

    Tensor<T>*load(const string& filename) {
        std::ifstream ifs(filename, std::ios::in | std::ios::binary);
        //std::ifstream ifs =
        size_t start_row = 0;
        size_t end_row = -1;

        size_t r_ndim;

        // Load number of dimensions
        ifs.read(reinterpret_cast<char *>(&r_ndim),  sizeof(int));

        // Load dimensions
        vector<int> r_shape(r_ndim);
        ifs.read(reinterpret_cast<char *>(r_shape.data()), r_ndim * sizeof(int));

        // Compute total size_
        size_t r_size = 1;
        for(int i=0; i<r_ndim; i++){ r_size *= r_shape[i]; }

        // Compute stride
        vector<int> tmp_stride = shape2stride(r_shape);

        // Compute offsets and positions to read
        size_t start_offset = start_row * tmp_stride[0];
        size_t n_read;

        if(end_row<0){
            n_read = r_size;
        }else{
            // Compute bytes to read
            size_t n_rows = end_row - start_row;
            n_read = n_rows * tmp_stride[0];

            // Set new shape
            r_shape[0] = n_rows;

            // Set cursor's position
            ifs.seekg(start_offset*sizeof(float), std::ifstream::cur);
        }

        auto *t1 = new Tensor(r_shape);
        ifs.read(reinterpret_cast<char*>(t1->ptr), n_read * sizeof(float));
        // Load content (row-major)
        /*
        auto *r_ptr = new float[r_size];
        ifs.read(reinterpret_cast<char*>(r_ptr), n_read * sizeof(float));

        // Return new tensor
        auto *t1 = new Tensor(r_shape, r_ptr, DEV_CPU);
        */
//    t1->info();
        return t1;
    }





    std::string size() {
        vector<int> v = this->getShape();

        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            ss << v[i];
            if (i != v.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]" << std::endl;
        return std::string(ss.str());
    }


    std::string type() {
        return std::string ("float");
    }

    void* data_ptr() {
        return this->ptr;
    }

    Tensor<T>*einsum (const string& equation, vector<Tensor<T>*>&Ta) {
        const auto arrow_pos = equation.find("->");
        const auto lhs = equation.substr(0, arrow_pos);
        const auto rhs = equation.substr(arrow_pos + 2);
        const auto num = Ta.size();
        std::size_t curr_op = 0;
        std::size_t curr_op1 = 0;
        std::vector<std::vector<int>> op_labels(num);
        std::vector<int> op_labelsR(num);
        Tensor* result;
        size_t count=0;//判断是否出现逗号
        for (auto i = decltype(lhs.length()){0}; i < lhs.length(); ++i) {
            switch (lhs[i]) {
                // ......
                case ',':
                    // 遇到逗号，接下来解析下一个输入张量的字符串
                    ++curr_op;
                    count++;
                    // ......
                    break;
                default:
                    // ......
                    // 把 char 字符转成 int
                    op_labels[curr_op].push_back(lhs[i] - 'a');
            }
        }

        for (auto i =0; i < rhs.length(); ++i) {
            int s=rhs[i] - 'a';
            op_labelsR.push_back(s);
        }

        if(rhs.length()==1){
            if(count==0){//ij->i or ij->j
                int judge=0;int judge1=0;int sss=0;int ss=0;
                for (const auto& inner_vector : op_labels) {//ii->i

                    // 遍历内部向量中的每个元素
                    for (int element : inner_vector) {

                        // 输出每个元素
                        if(element== op_labelsR[1]){
                            judge=1;ss=sss;
                        }
                        else{judge1=1;}//一个符合一个不符合
                        sss++;
                    }
                }
                if(judge==1&&judge1==1){
                    //ss为0 按行 ss为1 按列
                    if(ss==0){result=Ta[0]->sum({1}, false);}
                    else {result=Ta[0]->sum({0}, false);}
                    return result;
                }
                for (const auto& inner_vector : op_labels) {//ii->i
                    // 遍历内部向量中的每个元素
                    for (int element : inner_vector) {
                        // 输出每个元素
                        if(element!= op_labelsR[1]){
                            cout<<"The input is wrong!!!!!"<<endl;
                            return Ta[0];
                        }
                    }
                }
                int size=Ta[0]->shape[1]>Ta[0]->shape[0]?Ta[0]->shape[0]:Ta[0]->shape[1];
                result=Tensor::rand({1, size}, 5.0);
                for (int i = 0; i < size; ++i) {
                    int num1 = i;
                    int num2 = i+1;

                    // 将数字转换为字符串
                    std::string str_num1 = std::to_string(num1);
                    std::string str_num2 = std::to_string(num2);

                    // 拼接字符串
                    std::string final = str_num1 + ":" + str_num2;
                    float s=Ta[0]->select({ str_num1,  str_num1})->ptr[0];
                    result->set_select({"0:1", final}, s);  // 前两行前两列 为 7
                }
                return result;

            }
        }
        else if(rhs.length()==2){
            if(count==0){//the condition of ij->ji
                //   if()
            }

        }


        return result;
    }
};

// + *3
template <typename u>
Tensor<u>& operator+ (Tensor<u> &A, Tensor <u>&B) {
    Tensor<u>* t = Tensor<u>::add(&A, &B);
    return (*t);
}
template <typename u>
Tensor<u>& operator+ (Tensor<u> &A, u v) {
    Tensor<u>* t = A.clone();
    t->add_(v);
    return (*t);
}
template <typename u>
Tensor<u>& operator+ (u v, Tensor<u> &A) {
    return A + v;
}

template <typename u>
// - *3
Tensor<u>& operator* (Tensor<u> &A, Tensor<u> &B) {
    Tensor<u>* t = Tensor<u>::mul(&A, &B);
    return (*t);
}
template <typename u>
Tensor<u>& operator* (Tensor<u> &A, u v) {
    Tensor<u>* t = A.clone();
    t->mul_(v);
    return (*t);
}
template <typename u>
Tensor<u>& operator* (u v, Tensor <u>&A) {
    return A + v;
}

template <typename u>
// *= *2
void operator*= (Tensor<u> &A, Tensor<u> &B) {
    Tensor<u>::mul(1.0f, &A, 1.0f, &B, &A, 0);
}
template <typename u>
void operator*= (Tensor<u> &A, u v) {
    A.mul_(v);
}
// += *2
template <typename u>
void operator+= (Tensor<u> &A, Tensor<u> &B) {
    Tensor<u>::add(1.0f, &A, 1.0f, &B, &A, 0);
}
template <typename u>
void operator+= (Tensor<u> &A, u v) {
    A.add_(v);
}
template <typename u>
Tensor<u>& operator- (Tensor<u> &A, Tensor <u>&B) {
    Tensor<u>* t = Tensor<u>::sub(&A, &B);
    return (*t);
}
template <typename u>
Tensor<u>& operator- (Tensor<u> &A, u v) {
    Tensor<u>* t = A.clone();
    t->sub_(v);
    return (*t);
}
template <typename u>
Tensor<u>& operator- (u v, Tensor<u> &A) {
    return A + v;
}


// -= *2
template <typename u>
void operator-= (Tensor<u> &A, Tensor<u> &B) {
    Tensor<u>::sub(1.0f, &A, 1.0f, &B, &A, 0);
}
template <typename u>
void operator-= (Tensor<u> &A, u v) {
    A.sub_(v);
}
template <typename u>
Tensor<u>& operator/ (Tensor<u> &A, Tensor <u>&B) {
    Tensor<u>* t = Tensor<u>::div(&A, &B);
    return (*t);
}
template <typename u>
Tensor<u>& operator/ (Tensor<u> &A, u v) {
    Tensor<u>* t = A.clone();
    t->div_(v);
    return (*t);
}
template <typename u>
Tensor<u>& operator/ (u v, Tensor<u> &A) {
    return A + v;
}


// -= *2
template <typename u>
void operator/= (Tensor<u> &A, Tensor<u> &B) {
    Tensor<u>::div(1.0f, &A, 1.0f, &B, &A, 0);
}
template <typename u>
void operator/= (Tensor<u> &A, u v) {
    A.div_(v);
}
template <typename u>
std::ostream &operator<<(std::ostream &os, Tensor<u> &t) {
    //ostream &operator<<
    //this->print();
    //return <#initializer#>;

    int precision = 4;
    bool raw = false;

    int opened = 0;
    int closed = 0;

    Tensor<u> *aux = nullptr;
    aux = &t;

    // ***** Shitty code to prettify the output *******
    std::stringstream buffer;
    buffer << std::fixed;
    buffer << std::setprecision(precision);

    int lines = 0;
    int max_lines = 100000;
    for (int i = 0; i < aux->size_; ++i) {
        if(i % (&t)->stride[0]==0){lines++;}

        if(raw){
            // Print number
            buffer << aux->ptr[i] << ", ";

        }else{

            // Open brackets
            opened = 0;
            for (int j = 0; j < aux->ndim-1; ++j) {
                if(i%aux->stride[j]==0){
                    if(!opened && closed==1) {
                        if((aux->ndim) == 2) {
                            buffer << "\n";
                        } else {
                            //buffer << " ";
                            buffer << ", ";
                        }
                    }
                    buffer << "[";
                    opened += 1;
                }
            }

            // Print number
            buffer << aux->ptr[i];

            // Close brackets
            closed = 0;
            for (int j = 0; j < aux->ndim-1; ++j) {
                if((i+1)%aux->stride[j]==0) {
                    buffer << "]";
                    closed += 1;
                }
            }

            // Break lines
            if (i+1 < aux->size_){
                if(!closed){
                    //buffer << " ";
                    buffer << ", ";
                }
                else{
                    if (closed == 2 ) {  buffer << "\n"; }
                    else if (closed == 3) { buffer << "\n\n"; }
                    else if (closed > 3) { buffer << "\n\n\n"; }
                }
            }

            // Stop
            if(lines >= max_lines){
                std::cout << "Maximum tensor length exceeded." << std::endl;
                std::cout << "Printing only first " << max_lines << " rows:" << std::endl;
                break;
            }

        }

    }

    // Print to buffer
//    if(aux->ndim>1){
//        //std::cout << "[\n" << buffer.str() << "\n]" << std::endl;  // For readability
//        os << "[\n" << buffer.str() << "\n]" << std::endl;
//    }else{
//        //std::cout << "[" << buffer.str() << "]" << std::endl;
//        os << "[" << buffer.str() << "]" << std::endl;
//    }

    os << "[" << buffer.str() << "]" << std::endl;

    return os;

}
template <typename u>
void cpu_fill_(Tensor <u>*A, u v){
    for (int i = 0; i < A->size_; ++i){
        A->ptr[i] = v;
    }
}
template <typename T>
void cpu_eye(Tensor<T> *A, int offset){
    for(int i=0; i<A->size_; i++){
        if ((i/A->shape[0]+offset) == i%A->shape[1]){ A->ptr[i] = 1.0f; }  // rows+offset == col?
        else { A->ptr[i] = 0.0f; }
    }
}
template <typename T>
bool cpu_allclose(Tensor<T> *A, Tensor <T>*B, T rtol, T atol, bool equal_nan){
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



template <typename T>
void cpu_select(Tensor <T>*A, Tensor<T> *B, SelDescriptor *sd){

    for (int i = 0; i < B->size_; i++) {
        B->ptr[i] = A->ptr[sd->cpu_addresses[i]];
    }
}
template <typename T>
void cpu_concat(Tensor<T> *A, vector<Tensor<T>*> t, unsigned int axis, bool derivative){
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
template <typename T>
void cpu_set_select(Tensor<T> *A, Tensor <T>*B, SelDescriptor *sd){

    for (int i = 0; i < B->size_; i++) {
        A->ptr[sd->cpu_addresses[i]] = B->ptr[i];
    }
}
template <typename T>
void cpu_transpose(Tensor<T> * A, Tensor<T> * B) {
    for (int i = 0; i < A->size_; i++){
        B->ptr[i] = A->ptr[i];
    }
}
template <typename T>
void cpu_copy(Tensor<T> * A, Tensor <T>* B){
    for (int i = 0; i < A->size_; i++){
        B->ptr[i] = A->ptr[i];
    }
}
template <typename T>
void cpu_add(Tensor <T>*A, Tensor <T>*B, T v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] + v;
    }

}
template <typename T>
void cpu_add(T scA, Tensor <T>*A, T scB, Tensor<T> *B, Tensor <T>*C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
}
template <typename T>
void cpu_sub(Tensor<T> *A, Tensor<T> *B, T v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] - v;
    }

}
template <typename T>
void cpu_sub(T scA, Tensor <T>*A, T scB, Tensor<T> *B, Tensor <T>*C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] - scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] - scB * B->ptr[i];
}
template <typename T>
void cpu_mul(Tensor<T> *A, Tensor<T> *B, T v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] * v;
    }

}
template <typename T>
void cpu_mul(T scA, Tensor<T> *A,T scB, Tensor <T>*B, Tensor<T> *C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] * scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] * scB * B->ptr[i];
}
template <typename T>
void cpu_div(Tensor<T> *A, Tensor<T> *B, T v) {

    for (int i = 0; i < A->size_; ++i) {
        B->ptr[i] = A->ptr[i] / v;
    }

}
template <typename T>
void cpu_div(T scA, Tensor<T> *A,T scB, Tensor<T> *B, Tensor <T>*C, int incC) {

    for (int i = 0; i < A->size_; i++)
        if (incC) C->ptr[i] += scA * A->ptr[i] / scB * B->ptr[i];
        else C->ptr[i] = scA * A->ptr[i] / scB * B->ptr[i];
}

template <typename T>
T cpu_sum(Tensor<T> *A) {
    return cpu_sum(A->ptr, A->size_, nullptr);
}

template <typename T>
void cpu_sum(Tensor <T>*A, Tensor <T>*B, ReduceDescriptor2 *rd){
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_sum(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}
template <typename T>
T cpu_sum(T*ptr, int size, int *map) {
    float sum = 0.0f;

    if(map == nullptr){
        for (int i = 0; i < size; ++i) { sum += ptr[i]; }
    }else{
        for (int i = 0; i < size; ++i) { sum += ptr[map[i]]; }
    }

    return sum;
}






template <typename T>
T cpu_min(Tensor<T> *A) {
    return cpu_min(A->ptr, A->size_, nullptr);
}

template <typename T>
void cpu_min(Tensor <T>*A, Tensor<T> *B, ReduceDescriptor2 *rd){
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_min(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}
template <typename T>
T cpu_min(T *ptr, int size, int *map) {
    T min_ = FLT_MAX;

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

template <typename T>
T cpu_max(Tensor <T>*A) {
    return cpu_max(A->ptr, A->size_, nullptr);
}

template <typename T>
void cpu_max(Tensor <T>*A, Tensor<T> *B, ReduceDescriptor2 *rd){
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_max(A->ptr, rd->index[i].size(), rd->index[i].data());
    }
}
template <typename T>
T cpu_max(T *ptr, int size, int *map) {
    T max_ = FLT_MAX;

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


template <typename T>
void cpu_equal(Tensor<T> *A, Tensor <T>*B, float v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = A->ptr[i] == v;
    }
}
template <typename T>
void cpu_equal(Tensor <T>*A, Tensor <T>*B, Tensor<T> *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = A->ptr[i] == B->ptr[i];
    }
}
template <typename T>
void cpu_nequal(Tensor<T> *A, Tensor <T>*B, T v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = (A->ptr[i] != v);
    }
}
template <typename T>
void cpu_nequal(Tensor<T> *A, Tensor<T>*B, Tensor <T>*C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = (A->ptr[i] != B->ptr[i]);
    }
}
template <typename T>
void cpu_lequal(Tensor <T>*A, Tensor <T>*B, T v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = (A->ptr[i] < v);
    }
}
template <typename T>
void cpu_lequal(Tensor <T>*A, Tensor <T>*B, Tensor<T> *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = (A->ptr[i] < B->ptr[i]);
    }
}
template <typename T>
void cpu_gequal(Tensor<T> *A, Tensor<T> *B, T v){
    for (int i = 0; i < A->size_; ++i){
        B->ptr[i] = (A->ptr[i] > v);
    }
}
template <typename T>
void cpu_gequal(Tensor <T>*A, Tensor <T>*B, Tensor<T> *C){
    for (int i = 0; i < A->size_; ++i){
        C->ptr[i] = (A->ptr[i] > B->ptr[i]);
    }
}
template <typename T>
void cpu_log(Tensor<T> *A, Tensor<T> *B) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::logf(A->ptr[i]);
}
template <typename T>
void cpu_log2(Tensor<T>*A, Tensor <T>*B) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::log2f(A->ptr[i]);
}
template <typename T>
void cpu_log10(Tensor <T>*A, Tensor<T> *B) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::log10f(A->ptr[i]);
}
template <typename T>
void cpu_logn(Tensor <T>*A, Tensor<T> *B, T n) {
    for (int i = 0; i < A->size_; ++i) B->ptr[i] = ::logf(A->ptr[i]) / ::logf(n);
}
#endif //TENSOR_TENSOR_H
