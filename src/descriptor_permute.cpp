//
// Created by user_9k7t0TZ11 on 2024/1/13.
//

#include "tensor_descriptors.h"
#include "utils.h"

PermuteDescriptor::PermuteDescriptor(const vector<int>& dims ) : SelDescriptor( ) {
    this->dims = vector<int>(dims);
}


void PermuteDescriptor::build(vector<int> ishape){
    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = permute_shape(ishape, this->dims);

    // Build indices
    this->build_indices();
}

void PermuteDescriptor::resize(int b){
//    // Update shapes
//    this->ishape[0] = b;
//    this->oshape[0] = b;

    // Build indices
    this->build_indices();
}

void PermuteDescriptor::build_indices(){
    // Delete previous allocations
    this->free_memory();

    // Compute index translation (output=>input)
    this->cpu_addresses = permute_indices(this->ishape, this->dims);
}
