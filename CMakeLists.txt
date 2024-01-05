cmake_minimum_required(VERSION 3.27)
project(Libtensor_by_Yang_main)

set(CMAKE_CXX_STANDARD 14)

include_directories(Libtensor_by_Yang-main)

add_executable(Libtensor_by_Yang_main
        Libtensor_by_Yang-main/tensor.cpp
        Libtensor_by_Yang-main/tensor.h)
