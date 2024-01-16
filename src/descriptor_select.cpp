//
// Created by user_9k7t0TZ11 on 2024/1/12.
//

#include "tensor_descriptors.h"
#include "utils.h"

SelDescriptor::SelDescriptor() : TensorDescriptor() {

}

SelDescriptor::SelDescriptor(const vector<string>& indices ) : TensorDescriptor() {
    this->indices = vector<string>(indices);
}

void SelDescriptor::build(vector<int> ishape){
    // Compute ranges
    this->idxs_range = parse_indices(this->indices, ishape);

    // Get input/output shapes
    this->ishape = ishape;
    this->oshape = indices2shape(this->idxs_range);

    // Build indices
    this->build_indices();
}

void SelDescriptor::resize(int b){

    build_indices();
}

void SelDescriptor::build_indices(){
    // Delete previous allocations
    this->free_memory();

    // Compute index translation (output=>input)
    this->cpu_addresses = ranges2indices(this->ishape, this->idxs_range);
}
