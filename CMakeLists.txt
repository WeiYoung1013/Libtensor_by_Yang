cmake_minimum_required(VERSION 3.27)
project(tensor)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

<<<<<<< HEAD
#add_executable(tensor tensor.cpp)
add_executable(main
        src/tensor.cpp
        src/tensor.h
        src/cpu.cpp
        src/cpu.h
        src/tensor_descriptors.h
        src/utils.h
        src/utils.cpp
        src/descriptor_select.cpp
        src/descriptor_tensor.cpp
        src/descriptor_tile.cpp
        src/descriptor_permute.cpp
        src/descriptor_reduce.cpp
        main.cpp)
=======
add_executable(Libtensor_by_Yang
        Libtensor_by_Yang-main/tensor.h
        Libtensor_by_Yang-main/tensor.cpp)
>>>>>>> 3bf839296b76384b80d1f8f12285efc8382b7100
