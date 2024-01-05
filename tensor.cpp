
#include "tensor.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

using namespace std;

namespace ts {

    template <typename T>
    Tensor<T> Tensor<T>::tensor(const std::vector<std::vector<T>>& initList){
        return Tensor<T>(initList);
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(size_t index) const {
        if (index >= rows) {
            throw std::out_of_range("Index out of bounds");
        }

        Tensor<T> result(1, cols);
        for (size_t j = 0; j < cols; ++j) {
            result(0, j) = data[index][j];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator()(size_t dim, const std::vector<size_t>& range) const {
        if (dim >= rows || range.size() != 2 || range[0] >= cols || range[1] > cols) {
            throw std::out_of_range("Index or slice out of bounds");
        }

        size_t start = range[0];
        size_t end = range[1];

        Tensor<T> result(1, end - start);
        for (size_t j = start; j < end; ++j) {
            result(0, j - start) = data[dim][j];
        }
        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::rand(const std::vector<size_t>& size) {
        std::srand(std::time(nullptr));

        Tensor<T> result(size);
        for (size_t i = 0; i < size[0]; ++i) {
            for (size_t j = 0; j < size[1]; ++j) {
                result(i, j) = static_cast<T>(std::rand()) / RAND_MAX; 
            }
        }
        return result;
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<std::vector<T>>& input) : data(input), rows(input.size()), cols(input[0].size()) {
        data.resize(rows, std::vector<T>(cols, T()));
    }

    template <typename T>
    Tensor<T>::Tensor(size_t rows, size_t cols) : data(rows, std::vector<T>(cols)), rows(rows), cols(cols) {}

    template <typename T>
    std::vector<size_t> Tensor<T>::size() const {
        return { rows, cols };
    }
           

    template <typename T>
    const T* Tensor<T>::data_ptr() const {
        return &data[0][0];
    }

    template <typename T>
    Tensor<T> zeros(const std::vector<size_t>& size) {
        return Tensor<T>(size);
    }

    template <typename T>
    Tensor<T> ones(const std::vector<size_t>& size) {
        return Tensor<T>(size, static_cast<T>(1));
    }

    template <typename T>
    Tensor<T> full(const std::vector<size_t>& size, T value) {
        return Tensor<T>(size, value);
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
        for (size_t i = 0; i < t.size()[0]; ++i) {
            os << "[";
            for (size_t j = 0; j < t.size()[1]; ++j) {
                os << " " << std::fixed << std::setprecision(4) << t.data[i][j];
                if (j < t.size()[1] - 1) {
                    os << ",";
                }
            }
            os << " ]";
            if (i < t.size()[0] - 1) {
                os << std::endl;
            }
        }
        return os;
    }

    template <typename T>
    template <typename U>
    Tensor<U> Tensor<T>::eye(const std::vector<size_t>& size) {
        if (size.size() != 2 || size[0] != size[1]) {
            throw std::invalid_argument("eye function requires a square matrix size");
        }

        Tensor<U> result(size);
        for (size_t i = 0; i < size[0]; ++i) {
            for (size_t j = 0; j < size[1]; ++j) {
                result(i, j) = (i == j) ? static_cast<U>(1) : static_cast<U>(0);
            }
        }
        return result;
    }

   // 2.2
    template <typename T>
    Tensor<T> Tensor<T>::cat(const std::vector<Tensor<T>>& tensors, int dim) {
        if (tensors.empty()) {
            throw std::invalid_argument("Cannot concatenate an empty list of tensors");
        }

        size_t rows = tensors[0].rows;
        size_t cols = tensors[0].cols;

        for (const auto& tensor : tensors) {
            if (tensor.rows != rows || tensor.cols != cols) {
                throw std::invalid_argument("All tensors must have the same dimensions");
            }
        }

        if (dim < 0 || dim >= 2) {
            throw std::invalid_argument("Invalid dimension for concatenation");
        }

        size_t resultRows = (dim == 0) ? tensors.size() * rows : rows;
        size_t resultCols = (dim == 1) ? tensors.size() * cols : cols;

        Tensor<T> result(resultRows, resultCols);

        for (size_t i = 0; i < tensors.size(); ++i) {
            for (size_t j = 0; j < tensors[i].rows; ++j) {
                for (size_t k = 0; k < tensors[i].cols; ++k) {
                    result((dim == 0) ? i * rows + j : j, (dim == 1) ? i * cols + k : k) = tensors[i](j, k);
                }
            }
        }

        return result;
    }
    template <typename T>
    Tensor<T> Tensor<T>::tile(const Tensor<T>& tensor, const std::vector<int>& dims) {
        if (dims.size() != 2) {
            throw std::invalid_argument("Invalid number of dimensions for tiling");
        }

        int tileRows = dims[0];
        int tileCols = dims[1];

        size_t resultRows = tensor.rows * tileRows;
        size_t resultCols = tensor.cols * tileCols;

        Tensor<T> result(resultRows, resultCols);

        for (size_t i = 0; i < resultRows; ++i) {
            for (size_t j = 0; j < resultCols; ++j) {
                result(i, j) = tensor(i % tensor.rows, j % tensor.cols);
            }
        }

        return result;
    }
 
    //2.4
    template <typename T>
    Tensor<T> Tensor<T>::transpose(int dim1, int dim2) const {
        if (dim1 < 0 || dim2 < 0 || dim1 >= cols || dim2 >= cols || dim1 == dim2) {
            throw std::invalid_argument("Invalid dimensions for transpose");
        }

        Tensor<T> transposedTensor(cols, rows);  

        if (rows > 0 && cols > 0 && rows <= data.size() && cols <= data[0].size()) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    transposedTensor(j, i) = data[i][j];
                }
            }
        }
        else {
            throw std::out_of_range("Vector subscript out of range in transpose");
        }

        return transposedTensor;
    }
    template <typename T>
    Tensor<T> Tensor<T>::permute(const std::vector<int>& dims) const {
        if (dims.size() != 2 || dims[0] < 0 || dims[1] < 0 || dims[0] >= rows || dims[1] >= cols || dims[0] == dims[1]) {
            throw std::invalid_argument("Invalid dimensions for permute");
        }

        Tensor<T> permutedTensor(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                permutedTensor(i, j) = data[i][j];
            }
        }

        return permutedTensor;
    }

    //3.2
    template <typename T>
    Tensor<T> Tensor<T>::sum(int dim) const {
        if (dim < 0 || dim >= data[0].size() || data.empty() || data[0].empty()) {
            throw std::invalid_argument("Invalid dimension for sum");
        }

        size_t rows = data.size();
        size_t cols = data[0].size();

        Tensor<T> result;

        if (dim == 0) {
            for (size_t j = 0; j < cols; j++) {
                T sum = T();
                for (size_t i = 0; i < rows; i++) {
                    sum += data[i][j];
                }
                result.data.push_back({ sum });
            }
        }
        else if (dim == 1) {
            for (size_t i = 0; i < rows; i++) {
                T sum = T();
                for (size_t j = 0; j < cols; j++) {
                    sum += data[i][j];
                }
                result.data.push_back({ sum });
            }
        }
        else {
            throw std::invalid_argument("Invalid dimension for sum");
        }

        cout << "Sum result:" << result << endl;

        return Tensor<T>(result.data);
    }

    template <typename U>
    Tensor<U> sum(const Tensor<U>& tensor, int dim) {
        return tensor.sum(dim);
    }
    template <typename T>
    const std::vector<T>& Tensor<T>::operator[](size_t index) const {
        if (index >= rows) {
            throw std::out_of_range("Index out of bounds");
        }
        return data[index];
    }

    template <typename T>
    std::string Tensor<T>::type() const {
        return typeid(T).name();
    }

    template <typename T>
    Tensor<bool> Tensor<T>::operator==(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] == other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return result;
    }

    template <typename T>
    Tensor<bool> Tensor<T>::eq(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] == other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return Tensor<bool>(result.data);
    }


    template <typename T>
    Tensor<bool> Tensor<T>::ne(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] != other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return Tensor<bool>(result.data);
    }

    template <typename T>
    Tensor<bool> Tensor<T>::gt(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] > other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return Tensor<bool>(result.data);
    }

    template <typename T>
    Tensor<bool> Tensor<T>::ge(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] >= other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return Tensor<bool>(result.data);
    }

    template <typename T>
    Tensor<bool> Tensor<T>::lt(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] < other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return Tensor<bool>(result.data);
    }

    template <typename T>
    Tensor<bool> Tensor<T>::le(const Tensor<T>& other) const {
        if (data.size() != other.data.size() || data[0].size() != other.data[0].size()) {
            throw std::invalid_argument("Cannot compare tensors with different dimensions");
        }

        Tensor<bool> result;

        for (size_t i = 0; i < data.size(); i++) {
            std::vector<bool> row;
            for (size_t j = 0; j < data[0].size(); j++) {
                row.push_back(data[i][j] <= other.data[i][j]);
            }
            result.data.push_back(row);
        }

        return Tensor<bool>(result.data);
    }


    template <typename T>
    Tensor<T> view(const Tensor<T>& tensor, const std::vector<int>& shape) {
        size_t totalSize = 1;
        for (int dim : shape) {
            totalSize *= dim;
        }

        if (totalSize != tensor.data.size() * tensor.data[0].size()) {
            throw std::invalid_argument("Invalid shape for view operation");
        }

        Tensor<T> result({});

        result.data.resize(shape[0], std::vector<T>(shape[1]));

        size_t idx = 0;
        for (size_t i = 0; i < tensor.data.size(); ++i) {
            for (size_t j = 0; j < tensor.data[0].size(); ++j) {
                result.data[idx / shape[1]][idx % shape[1]] = tensor.data[i][j];
                ++idx;
            }
        }

        return Tensor<T>(result.data);
    }


    template <typename T>
    void Tensor<T>::assign(size_t index, T value) {
        if (index >= rows * cols) {
            throw std::out_of_range("Index out of bounds");
        }

        size_t row = index / cols;
        size_t col = index % cols;

        data[row][col] = value;
    }



    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& size)
        : data(size[0], std::vector<T>(size[1], T(0))), rows(size[0]), cols(size[1]) {}

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& size, T value)
        : data(size[0], std::vector<T>(size[1], value)), rows(size[0]), cols(size[1]) {}

    template <typename T>
    void Tensor<T>::assign(size_t row, const std::vector<size_t>& range, const std::vector<T>& values) {
        if (row >= rows || range.size() != 2 || range[0] >= cols || range[1] > cols || values.size() != range[1] - range[0]) {
            throw std::out_of_range("Index or slice out of bounds");
        }

        size_t start = range[0];
        size_t end = range[1];

        size_t valueIndex = 0;
        for (size_t j = start; j < end; ++j) {
            data[row][j] = values[valueIndex];
            ++valueIndex;
        }
    }




 

 
} // namespace ts

template class ts::Tensor<int>;
template class ts::Tensor<float>;
template class ts::Tensor < double > ;

int main() {
    std::cout << "----------1.1------------" << std::endl;
    ts::Tensor<float> floatTensor = ts::zeros<float>({ 2, 3 });
    ts::Tensor<int> intTensor = ts::ones<int>({ 2, 3 });
    ts::Tensor<double> doubleTensor = ts::full<double>({ 2, 3 }, 0.6);
    
    std::cout << "Float Tensor:" << std::endl << floatTensor << std::endl;
    std::cout << "Int Tensor:" << std::endl << intTensor << std::endl;
    std::cout << "Double Tensor:" << std::endl << doubleTensor << std::endl;
    std::cout << "----------1.2------------" << std::endl;
    ts::Tensor<double> t4 = ts::Tensor<double>::rand({ 2,3 });
    cout << t4 << endl;

    cout << "----------1.3------------" << endl;
    ts::Tensor<double> t1 = ts::zeros<double>({ 2, 3 });
    ts::Tensor<double> t2 = ts::ones<double>({ 2, 3 });
    ts::Tensor<double> t3 = ts::full({ 2, 3 }, 0.6);
    
    
    std::cout << t1 << std::endl << t2 << std::endl << t3 << std::endl;

    cout << "----------1.5------------" << endl;
    ts::Tensor<double> identityMatrix = ts::Tensor<double>::eye({ 3, 3 });
    std::cout << "Identity Matrix:" << std::endl << identityMatrix << std::endl;

    ts::Tensor<double> t = ts::Tensor<double>::tensor({ {0.1,1.2},{2.2,3.1}, {4.8,5.2} });
    cout << t << endl;

    ts::Tensor<double> t5 = ts::Tensor<double>::tensor({
        {0.1, 1.2, 3.4, 5.6, 7.8},
        {2.2, 3.1, 4.5, 6.7, 8.9},
        {4.9, 5.2, 6.3, 7.4, 8.5}
        });
    cout << "----------index------------" << endl;
    cout << t5(1) << std::endl;
    cout << "----------slice------------" << endl;
    cout << t5(2, { 2, 4 }) << std::endl;

    ts::Tensor<double> t6 = ts::Tensor<double>::tensor({
        {0.1, 1.2},
        {2.2, 3.1},
        {4.9, 5.2}
        });

    ts::Tensor<double> t7 = ts::Tensor<double>::tensor({
        {0.2, 1.3},
        {2.3, 3.2},
        {4.8, 5.1}
        });
    cout << "----------cat0------------" << endl;
    ts::Tensor<double> t8 = ts::Tensor<double>::cat({ t6, t7 }, 0);
    cout << t8 << std::endl;
    cout << "----------cat1------------" << endl;
    ts::Tensor<double> t40 = ts::Tensor<double>::cat({ t6, t7 }, 1);
    cout << t40 << std::endl;
    cout << "----------tile------------" << endl;
    ts::Tensor<double> t9 = ts::Tensor<double>::tile(t6, { 2, 3 });
    cout << t9 << std::endl;
    cout << "----------Mutating------------" << endl;
    ts::Tensor<double> t10 = ts::Tensor<double>::tensor({ {0.1, 1.2, 3.4, 5.6, 7.8}, {2.2, 3.1, 4.5, 6.7, 8.9}, {4.9, 5.2, 6.3, 7.4, 8.5} });
    t10.assign(1, { 0, 5 }, { 1.0, 1.0, 1.0, 1.0, 1.0 });
    std::cout << t10 << std::endl;
    cout << "----------2.4------------" << endl;
    ts::Tensor<double> t11 = ts::Tensor<double>::tensor({ {0.1, 1.2, 3.4, 5.6, 7.8}, {2.2, 3.1, 4.5, 6.7, 8.9}, {4.9, 5.2, 6.3, 7.4, 8.5} });
    std::cout << t11.transpose(0, 1) << std::endl << t11.permute({ 1, 0 }) << std::endl;


    cout << "----------3.1------------" << endl;
    ts::Tensor<double> t12 = ts::Tensor<double>::tensor({{0.1, 1.2, 3.4, 5.6, 7.8}, {2.2, 3.1, 4.5, 6.7, 8.9}, {4.9, 5.2, 6.3, 7.4, 8.5}});


    ts::Tensor<double> t13 = ts::Tensor<double>::tensor({ {0.1, 1.2}, {2.2, 3.1}, {4.9, 5.2} });
    ts::Tensor<double> t14 = ts::Tensor<double>::tensor({ {0.2, 1.3}, {2.3, 3.2}, {4.8, 5.1} });

    std::cout <<"plus"<< t13 + t14 << std::endl << "mul"<<t13*t14 << endl<<"sub"<<t13-t14<<endl<<"div"<<t13/t14<< endl;
    
    ts::Tensor<double> t20 = ts::Tensor<double>::tensor({ {0.1, 1.2}, {2.2, 3.1}, {4.9, 5.2} });

    cout << "----------3.2------------" << endl;

    cout << t20.sum(1) << endl;
    cout << ts::sum(t20,0) << endl;

    ts::Tensor<double> t15 = ts::Tensor<double>::tensor({ {0.1, 1.2}, {2.2, 3.1}, {4.9, 5.2} });
    ts::Tensor<double> t16 = ts::Tensor<double>::tensor({ {0.2, 1.3}, {2.2, 3.2}, {4.8, 5.1} });

    cout << "----------3.3------------" << endl;
    std::cout << "Inequality Comparison:\n" << t15.ne(t16) << std::endl;
    std::cout << "Greater Than Comparison:\n" << t15.gt(t16) << std::endl;
    std::cout << "Greater Than or Equal To Comparison:\n" << t15.ge(t16) << std::endl;
    std::cout << "Less Than Comparison:\n" << t15.lt(t16) << std::endl;
    std::cout << "Less Than or Equal To Comparison:\n" << t15.le(t16) << std::endl;

    std::cout << "----------3.4------------" << endl;
    ts::Tensor<double> t17 = ts::Tensor<double>::tensor({ {1,2,3} });
    ts::Tensor<double> t18 = ts::Tensor<double>::tensor({ {4,5,6} });


 
}
