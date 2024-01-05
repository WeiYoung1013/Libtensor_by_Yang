cmake_minimum_required(VERSION 3.27)
project(Libtensor_by_Yang)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Libtensor_by_Yang
        Libtensor_by_Yang-main/tensor.h
        Libtensor_by_Yang-main/tensor.cpp)
