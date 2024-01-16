//
// Created by user_9k7t0TZ11 on 2024/1/11.
//

#include <utility>
#include <string>
#include <fstream>

#include "tensor.h"
#include "cpu.h"

// 1.1 无参构造函数
Tensor::Tensor() : ndim(0), size_(0) {}

// 1.2 传入 shape 每个维度的大小
Tensor::Tensor(const std::vector<int> &shape) {
    // Tensor(shape, nullptr, dev)
    updateShape(shape);
    updateSize();
    updateStrides();
    updateData();
}

// 1.3 传入 一维度的data做数据 传入
// shape 每个维度的大小
Tensor::Tensor(const std::vector<float>& data, const std::vector<int> &shape) {
    updateShape(shape);
    updateSize();
    updateStrides();
    updateData();
    std::copy(data.begin(), data.end(), this->ptr);
}


// 析构函数
Tensor::~Tensor() {
    this->deleteData();
}

vector<int> Tensor::getShape(void) {
    return vector<int>(this->shape);
}

// 更新张量 size_ 维度
void Tensor::updateSize() {
    this->size_ = 1;

    for(auto &d : this->shape) {
        this->size_ = this->size_ * d;
    }
}

void Tensor::updateData() {
    if(this->ptr == nullptr){
        this->ptr = (float *)malloc(size_ * sizeof(float));
    }
}

// 更新张量  shape  每个维度的大小
void Tensor::updateShape(const std::vector<int> &new_shape){
    // this->shape = vector<int>(new_shape);
    this->shape.clear();
    for (int _ : new_shape) this->shape.push_back(_);
    this->ndim = this->shape.size();
}

void Tensor::updateStrides() {
    this->stride.clear();  // Remove all elements

    unsigned long int new_size = this->size_;
    for(int i=0;i<ndim;i++) {
        new_size /= shape[i];
        this->stride.push_back(new_size);
    }
}

void Tensor::deleteData(){
    if(this->ptr != nullptr){
        free(this->ptr);
        this->ptr = nullptr;
    }
}

// 判断一样的size大小
bool Tensor::sameSize(Tensor *A, Tensor *B) {
    return A->size_ == B->size_;
}

// 判断一样的维度结构
int Tensor::sameShape(Tensor *A, Tensor *B) {
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
void Tensor::print(int precision  , bool raw  ) {
    int opened = 0;
    int closed = 0;

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

Tensor* Tensor::empty(const std::vector<int> &shape){
    return new Tensor(shape);
}

Tensor* Tensor::empty_like(Tensor *A){
    return Tensor::empty(A->shape);
}

// 1.3 zeros
Tensor* Tensor::zeros(const std::vector<int> &shape ){
    auto t = new Tensor(shape);
    t->fill_(0.0f);
    return t;
}

//Tensor* Tensor::zeros_like(Tensor *A){
//    return Tensor::zeros(A->shape );
//}

// 1.3 ones
Tensor* Tensor::ones(const std::vector<int> &shape ){
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
Tensor* Tensor::full(const std::vector<int> &shape, float value ){
    auto t = new Tensor(shape );
    t->fill_(value);
    return t;
}
//
//Tensor* Tensor::full_like(Tensor *A, float value){
//    return Tensor::full(A->shape, value );
//}
//
void Tensor::fill_(float v) {
    Tensor::fill(this, v);
}
//
//Tensor* Tensor::fill(float v){
//    Tensor* t_new = Tensor::empty_like(this);
//    Tensor::fill(t_new, v);
//    return t_new;
//}

void Tensor::fill(Tensor* A, float v){
    cpu_fill_(A, v);
}

//----------------------------------------------------------------------------------------------------------------------

// 判断是否相等
int Tensor::equivalent(Tensor *A, Tensor *B, float atol, float rtol, bool equal_nan) {
    // Equal ndims and shapes
    if (!sameShape(A, B)) {
        return 0;
    }

    return cpu_allclose(A, B, rtol, atol, equal_nan);
}

//----------------------------------------------------------------------------------------------------------------------

// 1.4
Tensor* Tensor::eye(int rows, int offset){
    auto t = new Tensor(std::vector<int>{rows, rows});
    cpu_eye(t, offset);
    return t;
}

// std::srand(std::time(nullptr));
Tensor* Tensor::rand(const std::vector<int> &shape, float v){
    auto A = new Tensor(shape);
    for (int i = 0; i < A->size_; ++i) {
        float uniform = static_cast<float>(std::rand()) / RAND_MAX;
        A->ptr[i] = uniform * v;
    }

    return A;
}

// index Indexing and slicing

Tensor* Tensor::select(const vector<string>& indices){
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

void Tensor::select(Tensor *A, Tensor* B, SelDescriptor *sd){
    cpu_select(A, B, sd);
}

//---

// 2.2 拼接
Tensor* Tensor::concat(const vector<Tensor*> A, unsigned int axis, Tensor* output){
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

Tensor* Tensor::tile(Tensor* A, const vector<int>& repeats){
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

    // Perform select
    Tensor::select(A, new_t, td);

    delete td;
    return new_t;
}


// ---
// 2.3 Mutating operations

void Tensor::set_select(const vector<string>& indices, float value){
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

void Tensor::set_select(const vector<string>& indices, Tensor *A){
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

void Tensor::set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    cpu_set_select(A, B, sd);
}

void Tensor::transpose(Tensor *A, Tensor *B, vector<int> dims) {
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

void Tensor::permute_(const vector<int>& dims){
    Tensor* temp = Tensor::permute(this, dims);

    // Update attributes
    updateShape(temp->shape);
    updateSize();
    updateStrides();
    Tensor::copy(temp, this);  // copy data

    delete temp;
}

Tensor* Tensor::permute(const vector<int>& dims){
    Tensor* t_new = Tensor::permute(this, dims);
    return t_new;
}

Tensor* Tensor::permute(Tensor* A, const vector<int>& dims){
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

void Tensor::copy(Tensor *A, Tensor *B) {
    ///////////////////////////////////////
    /// Copy from A to B
    //////////////////////////////////////

    if (!Tensor::sameSize(A, B)) {
        msg("Tensors with different size_", "Tensor::copy");
    }


    cpu_copy(A, B);

}

void Tensor::reshape_(const vector<int> &new_shape){
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

Tensor* Tensor::reshape(const vector<int> &new_shape){
    Tensor *t_new = Tensor::reshape(this, new_shape);
    return t_new;
}

Tensor* Tensor::reshape(Tensor *A, const vector<int> &shape){
    Tensor *t_new = A->clone();
    t_new->reshape_(shape);
    return t_new;
}

Tensor* Tensor::clone(){
    auto* t_new = new Tensor(this->shape);
    Tensor::copy(this, t_new);
    return t_new;
}


// add
void Tensor::add_(float v){
    Tensor::add(this, this, v);
}

Tensor* Tensor::add(float v){
    Tensor *t = this->clone();
    t->add_(v);
    return t;
}

void Tensor::add_(Tensor* A){
    Tensor::add(this, A, this);
}

Tensor* Tensor::add(Tensor* A){
    Tensor *t = this->clone();
    t->add_(A);
    return t;
}

void Tensor::add(Tensor *A, Tensor *B, float v){
    cpu_add(A, B, v);
}

Tensor* Tensor::add(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape() );
    Tensor::add(A, B, C);
    return C;
}

void Tensor::add(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::add(1.0, A, 1.0, B, C, 0);
}

void Tensor::add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
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

// + *3
Tensor& operator+ (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::add(&A, &B);
    return (*t);
}

Tensor& operator+ (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->add_(v);
    return (*t);
}

Tensor& operator+ (float v, Tensor &A) {
    return A + v;
}


// += *2
void operator+= (Tensor &A, Tensor &B) {
    Tensor::add(1.0f, &A, 1.0f, &B, &A, 0);
}

void operator+= (Tensor &A, float v) {
    A.add_(v);
}

// ------------------------------
// sub

void Tensor::sub_(float v){
    Tensor::sub(this, this, v);
}

Tensor* Tensor::sub(float v){
    Tensor *t = this->clone();
    t->sub_(v);
    return t;
}

void Tensor::sub_(Tensor* A){
    Tensor::sub(this, A, this);
}

Tensor* Tensor::sub(Tensor* A){
    Tensor *t = this->clone();
    t->sub_(A);
    return t;
}

void Tensor::sub(Tensor *A, Tensor *B, float v){
    cpu_sub(A, B, v);
}

Tensor* Tensor::sub(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape() );
    Tensor::sub(A, B, C);
    return C;
}

void Tensor::sub(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::sub(1.0, A, 1.0, B, C, 0);
}

void Tensor::sub(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
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
Tensor& operator- (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::sub(&A, &B);
    return (*t);
}

Tensor& operator- (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->sub_(v);
    return (*t);
}

Tensor& operator- (float v, Tensor &A) {
    return A + v;
}


// -= *2
void operator-= (Tensor &A, Tensor &B) {
    Tensor::sub(1.0f, &A, 1.0f, &B, &A, 0);
}

void operator-= (Tensor &A, float v) {
    A.sub_(v);
}

// -----
// mul

void Tensor::mul_(float v){
    Tensor::mul(this, this, v);
}

Tensor* Tensor::mul(float v){
    Tensor *t = this->clone();
    t->mul_(v);
    return t;
}

void Tensor::mul_(Tensor* A){
    Tensor::mul(this, A, this);
}

Tensor* Tensor::mul(Tensor* A){
    Tensor *t = this->clone();
    t->mul_(A);
    return t;
}

void Tensor::mul(Tensor *A, Tensor *B, float v){
    cpu_mul(A, B, v);
}

Tensor* Tensor::mul(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape() );
    Tensor::mul(A, B, C);
    return C;
}

void Tensor::mul(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::mul(1.0, A, 1.0, B, C, 0);
}

void Tensor::mul(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
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

// - *3
Tensor& operator* (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::mul(&A, &B);
    return (*t);
}

Tensor& operator* (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->mul_(v);
    return (*t);
}

Tensor& operator* (float v, Tensor &A) {
    return A + v;
}


// *= *2
void operator*= (Tensor &A, Tensor &B) {
    Tensor::mul(1.0f, &A, 1.0f, &B, &A, 0);
}

void operator*= (Tensor &A, float v) {
    A.mul_(v);
}

// -----div

void Tensor::div_(float v){
    Tensor::div(this, this, v);
}

Tensor* Tensor::div(float v){
    Tensor *t = this->clone();
    t->div_(v);
    return t;
}

void Tensor::div_(Tensor* A){
    Tensor::div(this, A, this);
}

Tensor* Tensor::div(Tensor* A){
    Tensor *t = this->clone();
    t->div_(A);
    return t;
}

void Tensor::div(Tensor *A, Tensor *B, float v){
    cpu_div(A, B, v);
}

Tensor* Tensor::div(Tensor *A, Tensor *B){
    Tensor* C = Tensor::empty(A->getShape() );
    Tensor::div(A, B, C);
    return C;
}

void Tensor::div(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::div(1.0, A, 1.0, B, C, 0);
}

void Tensor::div(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
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
Tensor& operator/ (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::div(&A, &B);
    return (*t);
}

Tensor& operator/ (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->div_(v);
    return (*t);
}

Tensor& operator/ (float v, Tensor &A) {
    return A + v;
}


// -= *2
void operator/= (Tensor &A, Tensor &B) {
    Tensor::div(1.0f, &A, 1.0f, &B, &A, 0);
}

void operator/= (Tensor &A, float v) {
    A.div_(v);
}

//

// -----log

void Tensor::log_(){
    Tensor::log(this, this);
}


Tensor* Tensor::log(){
    Tensor *t = this->clone();
    t->log_();
    return t;
}


void Tensor::log(Tensor *A, Tensor *B){
    cpu_log(A, B);
}


void Tensor::log2_(){
    Tensor::log2(this, this);
}


Tensor* Tensor::log2(){
    Tensor *t = this->clone();
    t->log2_();
    return t;
}


void Tensor::log2(Tensor *A, Tensor *B){
    cpu_log2(A, B);
}


void Tensor::log10_(){
    Tensor::log10(this, this);
}


Tensor* Tensor::log10(){
    Tensor *t = this->clone();
    t->log10_();
    return t;
}


void Tensor::log10(Tensor *A, Tensor *B){
    cpu_log10(A, B);
}


void Tensor::logn_(float n){
    Tensor::logn(this, this, n);
}


Tensor* Tensor::logn(float n){
    Tensor *t = this->clone();
    t->logn_(n);
    return t;
}


void Tensor::logn(Tensor *A, Tensor *B, float n){
    cpu_logn(A, B, n);
}

// ------------------------------

float Tensor::sum(){
    return Tensor::sum(this);
}


float Tensor::sum(Tensor* A){
    return cpu_sum(A);
}

Tensor* Tensor::sum(vector<int> axis, bool keepdims){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims );
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape );
    Tensor::sum(this, t, rd);

    delete rd;
    return t;
}

void Tensor::sum(Tensor* A, Tensor *B, ReduceDescriptor2 *rd){
    cpu_sum(A, B, rd);

}


//


void Tensor::equal_(float v){
    Tensor::equal(this, this, v);
}

Tensor* Tensor::equal(float v){
    Tensor *t = this->clone();
    t->equal_(v);
    return t;
}

void Tensor::equal(Tensor *A, Tensor *B, float v){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::equal");
    }
    cpu_equal(A, B, v);
}

Tensor* Tensor::equal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->equal(this, A, t);
    return t;
}

void Tensor::equal(Tensor *A, Tensor *B, Tensor *C){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::equal");
    }
    if (!Tensor::sameShape(A, C)){
        msg("Tensors with different shape", "Tensor::equal");
    }
    cpu_equal(A, B, C);
}

// no equal


void Tensor::nequal_(float v){
    Tensor::nequal(this, this, v);
}

Tensor* Tensor::nequal(float v){
    Tensor *t = this->clone();
    t->nequal_(v);
    return t;
}

void Tensor::nequal(Tensor *A, Tensor *B, float v){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::nequal");
    }
    cpu_nequal(A, B, v);
}

Tensor* Tensor::nequal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->nequal(this, A, t);
    return t;
}

void Tensor::nequal(Tensor *A, Tensor *B, Tensor *C){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::nequal");
    }
    if (!Tensor::sameShape(A, C)){
        msg("Tensors with different shape", "Tensor::nequal");
    }
    cpu_nequal(A, B, C);
}


//


void Tensor::lequal_(float v){
    Tensor::lequal(this, this, v);
}

Tensor* Tensor::lequal(float v){
    Tensor *t = this->clone();
    t->lequal_(v);
    return t;
}

void Tensor::lequal(Tensor *A, Tensor *B, float v){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::lequal");
    }
    cpu_lequal(A, B, v);
}

Tensor* Tensor::lequal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->lequal(this, A, t);
    return t;
}

void Tensor::lequal(Tensor *A, Tensor *B, Tensor *C){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::lequal");
    }
    if (!Tensor::sameShape(A, C)){
        msg("Tensors with different shape", "Tensor::lequal");
    }
    cpu_lequal(A, B, C);
}


void Tensor::gequal_(float v){
    Tensor::gequal(this, this, v);
}

Tensor* Tensor::gequal(float v){
    Tensor *t = this->clone();
    t->gequal_(v);
    return t;
}

void Tensor::gequal(Tensor *A, Tensor *B, float v){
    if (!Tensor::sameShape(A, B)){
        msg("Tensors with different shape", "Tensor::gequal");
    }
    cpu_gequal(A, B, v);
}

Tensor* Tensor::gequal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->gequal(this, A, t);
    return t;
}

void Tensor::gequal(Tensor *A, Tensor *B, Tensor *C){
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

void Tensor::save(const string& filename) {
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

Tensor* Tensor::load(const string& filename) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    //std::ifstream ifs =
    int start_row = 0;
    int end_row = -1;

    int r_ndim;

    // Load number of dimensions
    ifs.read(reinterpret_cast<char *>(&r_ndim),  sizeof(int));

    // Load dimensions
    vector<int> r_shape(r_ndim);
    ifs.read(reinterpret_cast<char *>(r_shape.data()), r_ndim * sizeof(int));

    // Compute total size_
    int r_size = 1;
    for(int i=0; i<r_ndim; i++){ r_size *= r_shape[i]; }

    // Compute stride
    vector<int> tmp_stride = shape2stride(r_shape);

    // Compute offsets and positions to read
    int start_offset = start_row * tmp_stride[0];
    int n_read;

    if(end_row<0){
        n_read = r_size;
    }else{
        // Compute bytes to read
        int n_rows = end_row - start_row;
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



std::ostream &operator<<(std::ostream &os, Tensor &t) {
    //ostream &operator<<
    //this->print();
    //return <#initializer#>;

    int precision = 4;
    bool raw = false;

    int opened = 0;
    int closed = 0;

    Tensor *aux = nullptr;
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

std::string Tensor::size() {
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


std::string Tensor::type() {
    return std::string ("float");
}

void* Tensor::data_ptr() {
    return this->ptr;
}

