//
// Created by user_9k7t0TZ11 on 2024/1/12.
//

#ifndef TENSOR_TENSOR_DESCRIPTORS_H
#define TENSOR_TENSOR_DESCRIPTORS_H


#include <cstdio>
#include <vector>
#include <string>
#include <mutex>

using namespace std;


class TensorDescriptor {
public:
    int device;

    int* cpu_addresses;
    int* gpu_addresses;

    explicit TensorDescriptor();
    ~TensorDescriptor();

    // Don't mark as pure virtual because not all methods use the same parameters
    //virtual void build(){};
    virtual void resize(int b){};
    void free_memory();
};

class SelDescriptor : public TensorDescriptor {

public:
    vector<int> ishape;
    vector<int> oshape;
    vector<vector<int>> idxs_range;


    vector<string> indices;

    explicit SelDescriptor();
    SelDescriptor(const vector<string>& indices);

    virtual void build(vector<int> ishape);
    void resize(int b) override;
    virtual void build_indices();
};

class PermuteDescriptor : public SelDescriptor {
public:
    vector<int> dims;

    PermuteDescriptor(const vector<int>& dims);

    void build(vector<int> ishape) override;
    void resize(int b) override;
    void build_indices() override;
};

class GatherDescriptor : public SelDescriptor {
public:
    vector<int> dims;

    GatherDescriptor(const vector<int>& dims, int dev);

    void build(vector<int> ishape) override;
    void resize(int b) override;
    void build_indices() override;
};

class ExpandDescriptor : public SelDescriptor {
public:
    int size;

    ExpandDescriptor(int size, int dev);

    void build(vector<int> ishape) override;
    void resize(int b) override;
    void build_indices() override;
};

class RepeatDescriptor : public SelDescriptor {
public:
    vector<unsigned int> vrepeats;
    unsigned int axis;

    RepeatDescriptor(vector<unsigned int> vrepeats, unsigned int axis, int dev);

    void build(vector<int> ishape) override;
    void resize(int b) override;
    void build_indices() override;
};


class TileDescriptor : public SelDescriptor {
public:
    vector<int> vrepeats;
    int elem_repeats = 0;

    TileDescriptor(vector<int> vrepeats);

    void build(vector<int> ishape) override;
    void resize(int b) override;
    void build_indices() override;
};

class ReduceDescriptor2 : public TensorDescriptor {

private:
    void compute_output();
    void build_indices();

public:
    vector<int> axis;
    bool keepdims;
    vector<vector<int>> index;
    vector<int> ishape;
    vector<int> oshape;
    int size_reduction;

    ReduceDescriptor2(const vector<int>& axis, bool keepdims );

    ~ReduceDescriptor2();

    void build(const vector<int>& ishape);
    void resize(int b) override;
    void build_map(bool reverse=false);     // TODO: TEMP! I don't like this approach

};

#endif //TENSOR_TENSOR_DESCRIPTORS_H
