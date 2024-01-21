cmake_minimum_required(VERSION 3.27)
project(tensor)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

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

 FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()
