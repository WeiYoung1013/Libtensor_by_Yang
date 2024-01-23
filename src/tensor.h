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
#include <memory>
#include <algorithm>

#include "utils.h"
#include "tensor_descriptors.h"


using namespace std;
enum class OperationType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Log2,
    Log10,
    Logn,
};
std::string operationTypeToString(OperationType op) {
    switch (op) {
        case OperationType::Add: return "Add";
        case OperationType::Subtract: return "Subtract";
        case OperationType::Multiply: return "Multiply";
        case OperationType::Divide: return "Divide";
        case OperationType::Log2: return "Log2";
        case OperationType::Log10: return "Log10";
        case OperationType::Logn: return "Logn";
        default: return "Unknown";
    }
}





template <typename T>
T cpu_sum(T*ptr, int size, int *map);
template <typename T>
class Tensor {
public:
    unsigned int ndim;
    unsigned long int size_;
    // 数据指针
    T *ptr = nullptr;
    std::shared_ptr<unsigned int> ref_count;

    // 张量的尺寸
    std::vector<int> shape;
    std::vector<int> stride;
    template <typename u>
    friend Tensor<u>& operator- (Tensor<u> &A, Tensor<u> &B);
    template <typename u>
    friend Tensor<u>& operator- (Tensor <u>&A, u v);
    template <typename u>
    friend Tensor<u>& operator- (u v, Tensor<u> &A);

    Tensor<T> *grad = nullptr;
    std::vector<Tensor<T>*> grad_history; // 梯度历史记录
    vector<OperationType> op_history; // 操作历史记录
    void computeGradient(); // 计算梯度的方法
    Tensor<T>* getGradient(); // 获取当前梯度的方法
    void backward();

    static T determinantRecursive(const T* matrix, int n, int skipRow) {
        if (n == 1) {
            return matrix[0];
        }

        T det = 0;
        std::vector<T> submatrix((n - 1) * (n - 1));
        for (int x = 0; x < n; x++) {
            int subi = 0;
            for (int i = 0; i < n; i++) {
                if (i == skipRow) continue; // 跳过当前行
                int subj = 0;
                for (int j = 0; j < n; j++) {
                    if (j == x) continue; // 跳过当前列
                    submatrix[subi * (n - 1) + subj] = matrix[i * n + j];
                    subj++;
                }
                subi++;
            }
            det += (x % 2 == 0 ? 1 : -1) * matrix[skipRow * n + x] * determinantRecursive(submatrix.data(), n - 1, 0);
        }
        return det;
    }

    T determinant() const {
        if (shape.size() != 2 || shape[0] != shape[1]) {
            std::cout << "Current tensor is not a square matrix, cannot compute determinant." << std::endl;
            return static_cast<T>(0);
        }

        int n = shape[0];
        return determinantRecursive(ptr, n, 0);
    }



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
    // 1.1 无参构造函数
    // 1.1 无参构造函数
    Tensor() : ndim(0), size_(0), ref_count(new unsigned int(1)) {}

    // 1.2 传入 shape 每个维度的大小
    Tensor(const std::vector<int> &shape)
            : ref_count(new unsigned int(1)) {
        updateShape(shape);
        updateSize();
        updateStrides();
        updateData();
    }

    // 1.3 传入线性维度的data作为数据, shape 用于刻画每个维度的大小
    Tensor(const std::vector<T>& data, const std::vector<int> &shape)
            : ref_count(new unsigned int(1)) {
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
    T max() const {
        if (!ptr || size_ == 0) {
            // 处理空指针或大小为0的情况
            throw std::runtime_error("Tensor is empty.");
        }

        return *std::max_element(ptr, ptr + size_);
    }
    T min() const {
        if (!ptr || size_ == 0) {
            // 处理空指针或大小为0的情况
            throw std::runtime_error("Tensor is empty.");
        }

        return *std::min_element(ptr, ptr + size_);
    }
    Tensor<bool>le(const Tensor<T>& other) const {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Incompatible dimensions for comparison.");
        }

        Tensor<bool> result(this->shape);
        for (int i = 0; i < this->size_; ++i) {
            result.ptr[i] = this->ptr[i] <= other.ptr[i];
        }
        return result;
    }
    Tensor<bool> le(T value) const {
        Tensor<bool> result(this->shape);
        for (int i = 0; i < this->size_; ++i) {
            result.ptr[i] = this->ptr[i] <= value;
        }
        return result;
    }
    Tensor<bool>lt(const Tensor<T>& other) const {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Incompatible dimensions for comparison.");
        }

        Tensor<bool> result(this->shape);
        for (int i = 0; i < this->size_; ++i) {
            result.ptr[i] = this->ptr[i] < other.ptr[i];
        }
        return result;
    }
    Tensor<bool> lt(T value) const {
        Tensor<bool> result(this->shape);
        for (int i = 0; i < this->size_; ++i) {
            result.ptr[i] = this->ptr[i] < value;
        }
        return result;
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
    Tensor<T>* transpose(const vector<int>& dims){
        Tensor* t_new = Tensor::transpose(this, dims);
        return t_new;
    }
    static Tensor<T>*  transpose(Tensor* A, const vector<int>& dims) {
        // Build descriptor
        if(dims[0]!=0&&dims[1]!=1)return nullptr;
        auto *sd = new PermuteDescriptor({1,0} );
        sd->build(A->shape);

        // Initialize new tensor
        auto *new_t = new Tensor(sd->oshape );

        // Fill new tensor
        Tensor::select(A, new_t, sd);

        delete sd;
        return new_t;

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

        // 确保t的梯度已初始化
        if (t->grad == nullptr) {
            t->computeGradient();
        }

        // 如果A没有梯度，则计算A的梯度
        if (A->grad == nullptr) {
            A->computeGradient();
        }

        // 将A的梯度添加到t的梯度上
        Tensor<T>::add(static_cast<T>(1), t->grad, static_cast<T>(1), A->grad, t->grad, 0);

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

        // 更新C的梯度
        if (A->grad != nullptr || B->grad != nullptr) {
            if (C->grad == nullptr) {
                C->grad = Tensor<T>::empty(A->getShape());
            } else {
                C->grad_history.push_back(C->grad);
                C->grad = Tensor<T>::empty(A->getShape());
            }

            if (A->grad != nullptr) {
                Tensor<T>::add(static_cast<T>(1), C->grad, static_cast<T>(1), A->grad, C->grad, 0);
            }

            if (B->grad != nullptr) {
                Tensor<T>::add(static_cast<T>(1), C->grad, static_cast<T>(1), B->grad, C->grad, 0);
            }
        }
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

        // 确保t的梯度已初始化
        if (t->grad == nullptr) {
            t->computeGradient();
        }

        // 如果A没有梯度，则计算A的梯度
        if (A->grad == nullptr) {
            A->computeGradient();
        }

        // 将t的梯度与A的梯度相减
        Tensor<T>::sub(static_cast<T>(1), t->grad, static_cast<T>(1), A->grad, t->grad, 0);

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

        // 更新C的梯度
        if (A->grad != nullptr || B->grad != nullptr) {
            if (C->grad == nullptr) {
                C->grad = Tensor<T>::empty(A->getShape());
            } else {
                C->grad_history.push_back(C->grad);
                C->grad = Tensor<T>::empty(A->getShape());
            }

            if (A->grad != nullptr) {
                Tensor<T>::add(static_cast<T>(1), C->grad, static_cast<T>(1), A->grad, C->grad, 0);
            }

            if (B->grad != nullptr) {
                Tensor<T>::sub(static_cast<T>(1), C->grad, static_cast<T>(1), B->grad, C->grad, 0);
            }
        }

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

    void mul_(T v) {
        for (int i = 0; i < this->size_; ++i) {
            this->ptr[i] *= v;
        }
        if(this->grad== nullptr){
            this->computeGradient();
        }
        for (int i = 0; i < this->size_; ++i) {
            this->grad->ptr[i] *= v;
        }
    }

    Tensor<T>* mul(T v) {
        Tensor<T> *result = this->clone();
        result->mul_(v);
        return result;
    }

    Tensor<T>* mul(Tensor<T>* A) {
        // 确保是二维矩阵且符合矩阵乘法的维度要求
        if (this->shape.size() != 2 || A->shape.size() != 2 || this->shape[1] != A->shape[0]) {
            throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
        }

        int m = this->shape[0];
        int n = A->shape[1];
        int p = this->shape[1];
        Tensor<T>* result = new Tensor<T>({m, n});

        // 执行矩阵乘法
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int k = 0; k < p; ++k) {
                    sum += this->ptr[i * p + k] * A->ptr[k * n + j];
                }
                result->ptr[i * n + j] = sum;
            }
        }

        if(this->grad== nullptr){
            this->computeGradient();
        }
        if(A->grad== nullptr){
            A->computeGradient();
        }

        if (this->grad != nullptr) {
            // 创建一个用于存储梯度更新的新 Tensor
            Tensor<T>* gradUpdate = new Tensor<T>(this->shape);

            // 计算梯度更新：gradUpdate = result * A的转置
            int q = A->shape[1];
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < p; ++j) {
                    T sum = 0;
                    for (int k = 0; k < q; ++k) {
                        sum += result->ptr[i * q + k] * A->ptr[j * q + k]; // 注意：使用 A 的转置
                    }
                    gradUpdate->ptr[i * p + j] = sum;
                }
            }

            // 将梯度更新累加到 this 的梯度上
            for (int i = 0; i < this->size_; ++i) {
                this->grad->ptr[i] += gradUpdate->ptr[i];
            }

            // 清理
            delete gradUpdate;
        }
        result->grad= this->grad;



        return result;
    }




// -----div

    void div_(T v) {
        for (int i = 0; i < this->size_; ++i) {
            this->ptr[i] /= v;
        }
        if(this->grad== nullptr){
            this->computeGradient();
        }
        for (int i = 0; i < this->size_; ++i) {
            this->grad->ptr[i] /= v;
        }
    }

    Tensor<T>* div(T v) {
        Tensor<T> *result = this->clone();
        result->div_(v);
        return result;
    }

    void div_(Tensor<T>* A) {
        // 检查是否可以进行广播
        if (this->shape[1] == A->shape[1] && A->shape[0] == 1) {
            // 广播除法
            for (int i = 0; i < this->shape[0]; ++i) { // 对每一行
                for (int j = 0; j < this->shape[1]; ++j) { // 对每一列
                    this->ptr[i * this->shape[1] + j] /= A->ptr[j];
                }
            }
        } else if (this->shape == A->shape) {
            // 正常逐元素除法
            for (int i = 0; i < this->size_; ++i) {
                this->ptr[i] /= A->ptr[i];
            }
        } else {
            throw std::invalid_argument("Incompatible dimensions for element-wise division.");
        }


        if(this->grad== nullptr){
            this->computeGradient();
        }
        if(A->grad== nullptr){
            A->computeGradient();
        }
        if (this->grad != nullptr) {
            // 创建一个用于存储梯度更新的新 Tensor
            Tensor<T>* gradUpdate = new Tensor<T>(this->shape);

            if (A->shape[0] == 1) { // 广播情况
                for (int i = 0; i < this->shape[0]; ++i) {
                    for (int j = 0; j < this->shape[1]; ++j) {
                        // 使用适当的梯度更新公式
                        gradUpdate->ptr[i * this->shape[1] + j] = this->grad->ptr[i * this->shape[1] + j] / A->ptr[j];
                    }
                }
            } else { // 正常逐元素除法
                for (int i = 0; i < this->size_; ++i) {
                    // 使用适当的梯度更新公式
                    gradUpdate->ptr[i] = this->grad->ptr[i] / A->ptr[i];
                }
            }

            // 将梯度更新累加到 this 的梯度上
            for (int i = 0; i < this->size_; ++i) {
                this->grad->ptr[i] += gradUpdate->ptr[i];
            }

            // 清理
            delete gradUpdate;
        }


    }

    Tensor<T>* div(Tensor<T>* A) {
        Tensor<T> *result = this->clone();
        result->div_(A);
        return result;
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

        if (t->grad == nullptr) {
            t->computeGradient();
        }

        T ln2 = std::log(static_cast<T>(2));
        for (size_t i = 0; i < t->size_; ++i) {
            t->grad->ptr[i] += static_cast<T>(1) / (this->ptr[i] * ln2);
        }

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

        if (t->grad == nullptr) {
            t->computeGradient();
        }

        T ln10 = std::log(static_cast<T>(10));
        for (size_t i = 0; i < t->size_; ++i) {
            t->grad->ptr[i] += static_cast<T>(1) / (this->ptr[i] * ln10);
        }

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


        if (t->grad == nullptr) {
            t->computeGradient();
        }

        T lnn = std::log(static_cast<T>(n));
        for (size_t i = 0; i < t->size_; ++i) {
            t->grad->ptr[i] += static_cast<T>(1) / (this->ptr[i] * lnn);
        }

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

    void save(const std::string& filename) {
        std::ofstream ofs(filename, std::ios::out | std::ios::trunc); // 使用 trunc 标志清空文件
        if (!ofs.is_open()) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        // 写入维度和尺寸信息
        ofs << this->ndim << std::endl;
        for (int dim : this->shape) {
            ofs << dim << " ";
        }
        ofs << std::endl;

        // 写入张量数据
        for (size_t i = 0; i < this->size_; ++i) {
            ofs << this->ptr[i] << " ";
        }
        ofs << std::endl;

        ofs.close();
    }


    static Tensor<float>* load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::in);
        if (!ifs.is_open()) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        size_t r_ndim;
        ifs >> r_ndim;

        std::vector<int> r_shape(r_ndim);
        for (size_t i = 0; i < r_ndim; ++i) {
            ifs >> r_shape[i];
        }

        // 计算总大小
        size_t r_size = 1;
        for (int dim : r_shape) {
            r_size *= dim;
        }

        auto* t1 = new Tensor<float>(r_shape);
        for (size_t i = 0; i < r_size; ++i) {
            ifs >> t1->ptr[i];
        }

        ifs.close();
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
        return &this->ref_count;
    }

    T mean() const {
        T sum = 0;
        for (int i = 0; i < this->size_; ++i) {
            sum += this->ptr[i];
        }
        return sum / this->size_;
    }

    static void reverseString(string &str) {
        int start = 0;
        int end = str.length() - 1;

        while (start < end) {
            // 交换起始位置和结束位置的字符
            char temp = str[start];
            str[start] = str[end];
            str[end] = temp;

            // 更新起始和结束位置
            start++;
            end--;
        }
    }

    static Tensor<T> *einsum(const string &equation, vector<Tensor<T> *> &Ta) {
        const auto arrow_pos = equation.find("->");
        string lhs = equation.substr(0, arrow_pos);
        string rhs = equation.substr(arrow_pos + 2);
        const auto num = Ta.size();
        std::size_t curr_op = 0;
        std::size_t curr_op1 = 0;
        std::vector<std::vector<int>> op_labels(10);
        std::vector<int> op_labelsR(num);
        Tensor <T>*result= nullptr;
        size_t count = 0;//判断是否出现逗号

        for (int i = 0 ;i < lhs.length(); ++i) {
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

        for (auto i = 0; i < rhs.length(); ++i) {
            int s = rhs[i] - 'a';
            op_labelsR.push_back(s);
        }
        if(lhs.length()==7){
            if(rhs.length()==3&&count==1&&Ta.size()==2){
                vector<Tensor<T>*>res;
                size_t re=0;
                for (int kk = 0; kk < Ta[0]->shape[0]; ++kk) {
                    Tensor<T>* tcv2 = Tensor<T>::ones({1,Ta[0]->shape[1], Ta[1]->shape[2]});
                    for (int i = 0; i <Ta[0]->shape[1] ; ++i) {
                        for (int j = 0; j < Ta[1]->shape[2]; ++j) {
                            re=0;
                            for (int k = 0; k < Ta[0]->shape[2]; ++k) {
                                re+=(Ta[0]->ptr[k+Ta[0]->shape[2]*i+kk*Ta[0]->shape[1]*Ta[0]->shape[2]])*
                                    Ta[1]->ptr[k*Ta[1]->shape[2]+j+kk*Ta[1]->shape[1]*Ta[1]->shape[2]];
                            }
                            int n = i;int m=j;
                            std::string str_num = std::to_string(n);
                            std::string str_num1 = std::to_string(m);
                            tcv2->set_select({"0",str_num, str_num1}, re);
                        }
                    }
                    res.push_back(tcv2);
                }
                result=Tensor<T>::concat(res);
                return result;

            }
            else{cout<<"Wrong input!!!!!"<<endl;
                return Ta[0];
            }
        }
        if(lhs[0]=='.'){
            if(Ta.size()>1){
                cout<<"Wrong input!!"<<endl;
                return  Ta[0];
            }
            else{
                vector<int>s=Ta[0]->shape;
                size_t size=s.size();
                int lst=s[size-1];//last
                int lsts=s[size-2];//last second
                vector<int>fi;
                for (int i = 0; i < size-2; ++i) {
                    fi.push_back(i);
                }
                fi.push_back(size-1);
                fi.push_back(size-2);
                result=Tensor<T>::permute(Ta[0],fi);
                return  result;
            }

        }
        if(lhs.length()==2&&rhs.length()==0&&count==0){
            Tensor<T>* t8 = Tensor<T>::ones({1, 1});
            size_t size=1;
            vector<int>s=Ta[0]->shape;
            T res=0;
            for (int i = 0; i < s.size(); ++i) {
                size=size*s[i];
            }
            for (int i = 0; i < size; ++i) {
                res+=Ta[0]->ptr[i];
            }
            t8->set_select({"0", "0"}, res);
            return  t8;

        }
        if(lhs.length()>=3&&count==1&&rhs.length()==0){
            Tensor<T>* t8qcs = Tensor<T>::ones({1, 1});
            if(op_labels[0]==op_labels[1]){
                size_t size=1;
                vector<int>s=Ta[0]->shape;
                T res=0;
                for (int i = 0; i < s.size(); ++i) {
                    size=size*s[i];
                }

                //  cout<<t8s3g->ptr[0]<<endl;
                for (int i = 0; i < size; ++i) {
                    res+=Ta[0]->ptr[i]*Ta[1]->ptr[i];
                }
                t8qcs->set_select({"0", "0"}, res);
                return  t8qcs;


            }
        }

        if (rhs.length() == 1) {
            if (count == 0) {//ij->i or ij->j
                int judge = 0;
                int judge1 = 0;
                int sss = 0;
                int ss = 0;
                for (const auto &inner_vector: op_labels) {//ii->i

                    // 遍历内部向量中的每个元素
                    for (int element: inner_vector) {

                        // 输出每个元素
                        if (element == op_labelsR[1]) {
                            judge = 1;
                            ss = sss;
                        } else { judge1 = 1; }//一个符合一个不符合
                        sss++;
                    }
                }
                if (judge == 1 && judge1 == 1) {
                    //ss为0 按行 ss为1 按列
                    if (ss == 0) { result = Ta[0]->sum({1}, false); }
                    else { result = Ta[0]->sum({0}, false); }
                    return result;
                }
                for (const auto &inner_vector: op_labels) {//ii->i
                    // 遍历内部向量中的每个元素
                    for (int element: inner_vector) {
                        // 输出每个元素
                        if (element != op_labelsR[1]) {
                            cout << "The input is wrong!!!!!" << endl;
                            return Ta[0];
                        }
                    }
                }
                int size = Ta[0]->shape[1] > Ta[0]->shape[0] ? Ta[0]->shape[0] : Ta[0]->shape[1];
                result = Tensor::rand({1, size}, 5.0);
                for (int i = 0; i < size; ++i) {
                    int num1 = i;
                    int num2 = i + 1;

                    // 将数字转换为字符串
                    std::string str_num1 = std::to_string(num1);
                    std::string str_num2 = std::to_string(num2);

                    // 拼接字符串
                    std::string final = str_num1 + ":" + str_num2;
                    float s = Ta[0]->select({str_num1, str_num1})->ptr[0];
                    result->set_select({"0:1", final}, s);  // 前两行前两列 为 7
                }
                return result;

            }
            else if(count==1&&lhs.length()==4){
                if(lhs[0]==rhs[0]&&lhs[1]==lhs[3]){
                    Tensor<T>* t8qcs = Tensor<T>::ones({1, Ta[0]->shape[0]});
                    T res=0;
                    for (int i = 0; i < Ta[0]->shape[0]; ++i) {
                        res=0;
                        for (int j = 0; j < Ta[0]->shape[1]; ++j) {
                            res+=Ta[0]->ptr[j+i*Ta[0]->shape[1]]*Ta[1]->ptr[j];
                        }
                        int n = i;
                        std::string str_num = std::to_string(n);
                        t8qcs->set_select({"0", str_num}, res);
                    }
                    return t8qcs;
                }
                else{
                    cout<<"Wrong input!!"<<endl;
                    return Ta[0];
                }


            }

        } else if (rhs.length() == 2) {
            if (count == 0) {//the condition of ij->ji
                //   if()
                reverseString(rhs);
                // std::reverse(rhs.begin(), rhs.end());
                if (lhs == rhs) {
                    result = Tensor<T>::permute(Ta[0], {1, 0});
                    return result;
                }
                else if(count==1&&lhs.length()==3){

                }
            }
            else if(count==1){
                if(lhs.length()==5){
                    if( lhs[0]==rhs[0]&&lhs[1]==lhs[3]&&lhs[4]==rhs[1]){
                        Tensor<T>* t8qcs = Tensor<T>::ones({Ta[0]->shape[0], Ta[1]->shape[1]});
                        T res=0;
                        for (int i = 0; i <Ta[0]->shape[0] ; ++i) {
                            for (int j = 0; j < Ta[1]->shape[1]; ++j) {
                                res=0;
                                for (int k = 0; k < Ta[0]->shape[1]; ++k) {
                                    res+=(Ta[0]->ptr[k+Ta[0]->shape[1]*i])*Ta[1]->ptr[k*Ta[1]->shape[1]+j];
                                }
                                int n = i;int m=j;
                                std::string str_num = std::to_string(n);
                                std::string str_num1 = std::to_string(m);
                                t8qcs->set_select({str_num, str_num1}, res);
                            }
                        }return t8qcs;
                    }
                    else{cout<<"wrong input!"<<endl;
                        return  Ta[0];
                    }
                }
                else if(lhs.length()==3){
                    if(lhs[0]==rhs[0]&&lhs[2]==rhs[1]){
                        T res=0;
                        Tensor<T>* t8qcs = Tensor<T>::ones({Ta[0]->shape[1], Ta[1]->shape[1]});
                        for (int i = 0; i < Ta[0]->shape[1]; ++i) {
                            for (int j = 0; j < Ta[1]->shape[1]; ++j) {
                                res=Ta[0]->ptr[i]*Ta[1]->ptr[j];
                                int n = i;int m=j;
                                std::string str_num = std::to_string(n);
                                std::string str_num1 = std::to_string(m);
                                t8qcs->set_select({str_num, str_num1}, res);
                            }
                        }
                        return t8qcs;
                    }
                }
            }

        }


        cout<<"No matching!!"<<endl;
        return  Ta[0];
    };
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
    Tensor<u>* t = A.mul(&B);
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
    Tensor<u>* t = A.div(&B);
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
        T *dest = A->ptr + offset;
        T *src = t[i]->ptr;

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

template <typename T>
void Tensor<T>::computeGradient() {
    if (this->grad != nullptr) {
        delete this->grad; // 确保在创建新梯度前删除旧梯度
    }
    this->grad = new Tensor<T>(this->shape); // 创建一个与当前张量形状相同的张量作为梯度
    // 初始化梯度为1
    std::fill(this->grad->ptr, this->grad->ptr + this->size_, static_cast<T>(1));
}

template <typename T>
Tensor<T>* Tensor<T>::getGradient() {
    if (this->grad == nullptr) {
        this->computeGradient(); // 如果梯度未计算，先计算梯度
    }
    return this->grad;
}

template <typename T>
void Tensor<T>::backward() {
    std::cout << "Backward pass:" << std::endl;
    for (size_t i = 0; i < grad_history.size(); ++i) {
        std::cout << "-----------------grad_history-----------------" << std::endl;
        grad_history[i]->print(); // 假设 Tensor 有 print 方法打印其内容

        std::cout << "-----------------op_history-----------------" << std::endl;
        if (i < op_history.size()) { // 确保 op_history 中有对应的元素
            std::cout << operationTypeToString(op_history[i]) << std::endl;
        }
    }
}






#endif //TENSOR_TENSOR_H
