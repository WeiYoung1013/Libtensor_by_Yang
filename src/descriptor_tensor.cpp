//
// Created by user_9k7t0TZ11 on 2024/1/12.
//

#include "tensor_descriptors.h"

TensorDescriptor::TensorDescriptor(){
    // Set device
    //this->device = dev;  // Currently ignored

    // Initialize addresses
    cpu_addresses = nullptr;
    gpu_addresses = nullptr;
}

TensorDescriptor::~TensorDescriptor() {
    this->free_memory();
}

void TensorDescriptor::free_memory() {
    if (this->cpu_addresses != nullptr) {
        delete[] this->cpu_addresses;
    }

}