#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <complex>


namespace ts {
    template <typename T>
    class Tensor;

    template <typename T>
    Tensor<T> tensor(const std::vector<std::vector<T>>& input);

    template <typename T>
    class Tensor {
    private:
        


    public:
        size_t rows;
        size_t cols;        
        std::vector<std::vector<T>> data;
        Tensor() = default;
        Tensor(const std::vector<std::vector<T>>& input);
        Tensor(size_t rows, size_t cols);

        Tensor(const std::vector<size_t>& size);
        Tensor(const std::vector<size_t>& size, T value);


        std::vector<size_t> size() const;
        std::string type() const;
        const T* data_ptr() const;

        size_t size(size_t dim)const {
            if (dim >= size().size()) {
                throw std::out_of_range("Invalid dimention");

            }
            return size()[dim];
        }
        T& operator()(size_t row, size_t col) {
            if (row >= rows || col >= cols) {
                throw std::out_of_range("Index out of bounds");
            }
            return data[row][col];
        }

        const T& operator()(size_t row, size_t col) const {
            if (row >= rows || col >= cols) {
                throw std::out_of_range("Index out of bounds");
            }
            return data[row][col];
        }

        //1.3
        template <typename U = T>
        static Tensor<U> eye(const std::vector<size_t>& size);

        //1.1
        static Tensor<T> tensor(const std::vector<std::vector<T>>& input);
        //1.2
        static Tensor<T> rand(const std::vector<size_t>& size);
        //2.1
        Tensor<T> operator()(size_t index) const;
        Tensor<T> operator()(size_t dim,const std::vector<size_t>& range) const;
        //2.2
        static Tensor<T> cat(const std::vector<Tensor<T>>& tensors, int dim);
        static Tensor<T> tile(const Tensor<T>& tensor, const std::vector<int>& dims);
        //void Tensor<T>::copyFrom(const Tensor<T>& source, const std::vector<size_t>& startIdx);
        //2.3
        void assign(size_t index, T value);
        void assign(size_t row, const std::vector<size_t>& range, const std::vector<T>& values);
        //2.4
        Tensor<T> transpose(int dim1, int dim2) const;
        Tensor<T> permute(const std::vector<int>& dims) const;
        //2.5
        Tensor<T> add(const Tensor<T>& other) const {
            // Check if the sizes match
            if (size() != other.size()) {
                throw std::invalid_argument("Tensor sizes do not match for addition");
            }

            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] + other(i, j);
                }
            }

            return result;
        }
        //3.1
        Tensor<T> subtract(const Tensor<T>& other) const {
            // Check if the sizes match
            if (size() != other.size()) {
                throw std::invalid_argument("Tensor sizes do not match for subtraction");
            }

            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] - other(i, j);
                }
            }

            return result;
        }
        Tensor<T> multiply(const Tensor<T>& other) const {
            // Check if the sizes match
            if (size() != other.size()) {
                throw std::invalid_argument("Tensor sizes do not match for multiplication");
            }

            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] * other(i, j);
                }
            }

            return result;
        }
        Tensor<T> divide(const Tensor<T>& other) const {
            // Check if the sizes match
            if (size() != other.size()) {
                throw std::invalid_argument("Tensor sizes do not match for division");
            }

            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    // Check for division by zero
                    if (other(i, j) == 0) {
                        throw std::invalid_argument("Division by zero");
                    }

                    result(i, j) = data[i][j] / other(i, j);
                }
            }

            return result;
        }
        Tensor<T> add(T value) const {
            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] + value;
                }
            }

            return result;
        }
        Tensor<T> subtract(T value) const {
            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] - value;
                }
            }

            return result;
        }
        Tensor<T> multiply(T value) const {
            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] * value;
                }
            }

            return result;
        }
        Tensor<T> divide(T value) const {
            // Check for division by zero
            if (value == 0) {
                throw std::invalid_argument("Division by zero");
            }

            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result(i, j) = data[i][j] / value;
                }
            }

            return result;
        }
        Tensor<T> log() const {
            Tensor<T> result(size());
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    // Check for logarithm of non-positive value
                    if (data[i][j] <= 0) {
                        throw std::invalid_argument("Logarithm of non-positive value");
                    }

                    result(i, j) = std::log(data[i][j]);
                }
            }

            return result;
        }
        Tensor<T> operator+(const Tensor<T>& other) const {
            return add(other);
        }
        Tensor<T> operator-(const Tensor<T>& other) const {
            return subtract(other);
        }
        Tensor<T> operator*(const Tensor<T>& other) const {
            return multiply(other);
        }
        Tensor<T> operator/(const Tensor<T>& other) const {
            return divide(other);
        }
        Tensor<T> operator+(T value) const {
            return add(value);
        }
        Tensor<T> operator-(T value) const {
            return subtract(value);
        }
        Tensor<T> operator*(T value) const {
            return multiply(value);
        }
        Tensor<T> operator/(T value) const {
            return divide(value);
        }
        Tensor<T> operator-() const {
            return log();
        }
        Tensor<bool> operator==(const Tensor<T>& other) const;
        Tensor<T> sum(int dim) const;
        Tensor<T> mean(int dim) const;
        Tensor<T> max(int dim) const;
        Tensor<T> min(int dim) const;
        Tensor<bool> eq(const Tensor<T>& other) const;
        Tensor<bool> ne(const Tensor<T>& other) const;
        Tensor<bool> gt(const Tensor<T>& other) const;
        Tensor<bool> ge(const Tensor<T>& other) const;
        Tensor<bool> lt(const Tensor<T>& other) const;
        Tensor<bool> le(const Tensor<T>& other) const;



        Tensor<T> view(const Tensor<T>& tensor, const std::vector<int>& shape);


        template <typename U>
        friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& t);

        template <typename U>
        friend Tensor<U> sum(const Tensor<U>& tensor, int dim);

        const std::vector<T>& operator[](size_t index) const;
    };





} // namespace ts

#endif // TENSOR_H
