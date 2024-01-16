#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <array>

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include "src/tensor.h"

int main() {
ios::sync_with_stdio(false);
cin.tie(nullptr);cout.tie(nullptr);
#ifdef _WIN32
    //SetConsoleOutputCP(CP_UTF8);
#endif // _WIN32

    //初始化随机数种子
    srand((unsigned)(100));

    // 1. 生成 2 * 2 张量
    std::cout << "-----------------[1.1.1] create 2 * 2-----------------" << std::endl;
    Tensor* t8s3g = new Tensor({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    t8s3g->print();
    delete t8s3g;

    // 2. 生成 2 * 3 * 3 张量
    std::cout << "-----------------[1.1.2] create 2 * 3 * 3-----------------" << std::endl;
    Tensor* t63g0 = new Tensor({-0.5792, -0.1372,  -0.5792, -0.1372, 0.5962,  1.2097,
                               -0.5792, -0.1372,  -0.5792, -0.1372, 0.5962,  1.2097,
                               -0.5792, -0.1372,  -0.5792, -0.1372, 0.5962,  1.2097}, {2, 3, 3} );
    t63g0->print();
    delete t63g0;



    //  3 * 2 * 2 随机 张量
    std::cout << "-----------------[1.2.1] create 3 * 2 * 2 rand-----------------" << std::endl;

    Tensor* tgzxg = Tensor::rand({3, 2, 2}, 5.0);
    tgzxg->print();
    delete tgzxg;



    // 3. 4 * 2 * 2 全0 张量
    std::cout << "-----------------[1.3.1] create 4 * 2 * 2 zero-----------------" << std::endl;
    Tensor* temno = new Tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                               }, {4, 2, 3});
    Tensor* tdzja = Tensor::zeros({4, 2, 3});

    temno->print();
    delete t63g0;

    tdzja->print();
    delete tdzja;

    // [1.3.6] 2, 2, 2 全1 张量
    std::cout << "-----------------[1.3.2] create 2, 2, 2 full 1 ----------------" << std::endl;

    Tensor* tlcj6 = Tensor::ones({2, 2, 2});
    tlcj6->print();
    delete tlcj6;

    // [1.3.6] 2, 2, 2 full5 张量
    std::cout << "-----------------[1.3.6] create 2, 2, 2 full 5 ----------------" << std::endl;

    Tensor* tbwsq = Tensor::full({2, 2, 2}, 5);
    tbwsq->print();
    delete tbwsq;

    //  [1.4.1] create 4 * 4  eye
    std::cout << "-----------------[1.4.1] create 4 * 4  eye-----------------" << std::endl;
    Tensor* tm4pf = Tensor::eye(4);
    tm4pf->print();
    delete tm4pf;

    //  2.1 index and select
    std::cout << "-----------------[2.1] index-----------------" << std::endl;

    Tensor* tw109 = Tensor::rand({4, 6});
    tw109->print();

    Tensor* tw1092 = tw109->select({"0", "2"});  // 取 (0,2) 的数据
    tw1092->print();

    Tensor* tw1093 = tw109->select({"1", ":"});  // 取 第2行 的数据
    tw1093->print();

    Tensor* tw1094 = tw109->select({":", "2"});  // 取 第3列 的数据
    tw1094->print();

    Tensor* tw1095 = tw109->select({"0:2", "0:2"}); // 取前2列的前2行
    tw1095->print();

    delete tw109;
    delete tw1092;
    delete tw1093;
    delete tw1094;
    delete tw1095;

    //  2.2 join and tile
    std::cout << "-----------------[2.2] join and tile-----------------" << std::endl;

    //Tensor* tsxfy = new Tensor({1, 2}, {2, 1});
    Tensor* tsxfy = new Tensor({1, 1, 2, 2, 1, 1, 2, 2}, {4, 2});
    tsxfy->print();

    //
    Tensor* tsxfy1 = Tensor::tile(tsxfy, {2, 2});
    tsxfy1->print();

    Tensor* tsxfy2 = Tensor::full({2, 2, 2}, 2);
    Tensor* tsxfy3 = Tensor::full({2, 2, 2}, 5);

    // 拼接
    Tensor* tsxfy4 = Tensor::concat({tsxfy2, tsxfy3});

    tsxfy4->print();

    delete tsxfy;
    delete tsxfy1;
    delete tsxfy2;
    delete tsxfy3;
    delete tsxfy4;

    // Mutating
    std::cout << "-----------------[2.3] Mutating-----------------" << std::endl;

    Tensor* tlchy = Tensor::ones({4, 4});
    tlchy->set_select({"0", "2"}, 2.0f); // 把(0, 2) 设置 2
    tlchy->print();

    Tensor* tlchy2 = Tensor::ones({4, 4});
    tlchy2->set_select({":", "2"}, 5.0f);  // 第3列设置为 5
    tlchy2->print();

    Tensor* tlchy3 = Tensor::ones({4, 4});
    tlchy3->set_select({"0:1", "0:1"}, 7.0f);  // 前两行前两列 为 7
    tlchy3->print();

    delete tlchy;
    delete tlchy2;
    delete tlchy3;

    // permute
    std::cout << "-----------------[2.4] permute-----------------" << std::endl;
    Tensor* tuzku = Tensor::rand({3, 4}, 1.0);
    tuzku->print();

    // 转置 二维矩阵 3,4 -> 4，3
    Tensor *tuzku2 = Tensor::permute(tuzku, {1, 0});
    tuzku2->print();
    delete tuzku;
    delete tuzku2;

    // view
    std::cout << "-----------------[2.5] view-----------------" << std::endl;

    Tensor* tod8r = Tensor::rand({4, 3, 2}, 1.0);
    tod8r->print( );

    // 维度变换 4*3*2 变换 2*2*2
    tod8r->reshape_({2, 3, 4});
    tod8r->print( );

    delete tod8r;


    // 3.1 add
    std::cout << "-----------------[3.1] add-----------------" << std::endl;

    Tensor* t8qcv = Tensor::ones({3, 4});
    Tensor* t8qcv2 = Tensor::ones({3, 4});
    // 矩阵对应元素相加
    Tensor* t8qcv3 = Tensor::add(t8qcv, t8qcv2);
    Tensor* t8qcv4 = Tensor::add(t8qcv3, t8qcv);
    t8qcv4->print();

    Tensor qcv5 = Tensor({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor qcv6 = Tensor({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor qcv7 = qcv5 + qcv6;
    qcv7.print();

    delete t8qcv;
    delete t8qcv2;
    delete t8qcv3; delete t8qcv4;
    //delete qcv5;delete qcv6;delete qcv7;

    // 3.1 sub
    std::cout << "-----------------[3.1] sub-----------------" << std::endl;

    Tensor* t8qcs = Tensor::ones({3, 4});
    Tensor* t8qcs2 = Tensor::ones({3, 4});
    // 矩阵对应元素相加
    Tensor* t8qcs3 = Tensor::sub(t8qcs, t8qcs2);
    Tensor* t8qcs4 = Tensor::sub(t8qcs3, t8qcs);
    t8qcs4->print();

    Tensor qcv25 = Tensor({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor qcv26 = Tensor({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor qcv27 = qcv25 + qcv26;
    qcv27.print();

    delete t8qcs;
    delete t8qcs2;
    delete t8qcs3;
    delete t8qcs4;

    // 3.1 mul
    std::cout << "-----------------[3.1] mul-----------------" << std::endl;

    Tensor* t8qcsq = Tensor::rand({3, 4});
    Tensor* t8qcsq2 = Tensor::rand({3, 4});
    // 矩阵对应元素相加
    Tensor* t8qcsq3 = Tensor::mul(t8qcsq, t8qcsq2);
    //Tensor* t8qcsq4 = Tensor::mul(t8qcsq3, 6);
    t8qcsq3->print();

    Tensor qcv2d5 = Tensor({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor qcv2d6 = Tensor({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor qcv2d7 = qcv2d5 * qcv2d6;
    qcv2d7.print();

    delete t8qcsq;
    delete t8qcsq2;
    delete t8qcsq3;

    // 3.1 mul
    std::cout << "-----------------[3.1] div-----------------" << std::endl;

    Tensor* t8qcaq = Tensor::rand({3, 4});
    Tensor* t8qcaq2 = Tensor::rand({3, 4});
    // 矩阵对应元素相加
    Tensor* t8qcaq3 = Tensor::div(t8qcaq, t8qcaq2);
    //Tensor* t8qcaq4 = Tensor::div(t8qcaq3, t8qcaq4);
    t8qcaq3->print();

    Tensor qcv2qd5 = Tensor({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor qcv2qd6 = Tensor({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor qcv2qd7 = qcv2qd5 + qcv2qd6;
    qcv2d7.print();

    delete t8qcaq;
    delete t8qcaq2;
    delete t8qcaq3;


    // 3.1.1 log
    std::cout << "-----------------[3.1.1] log-----------------" << std::endl;
    Tensor* tmd45 = Tensor::rand({4, 4}, 5.0f);
    tmd45->print();
    // 元素对2取对数
    Tensor* tmd451 = tmd45->log2();
    tmd451->print();
    delete tmd45;
    delete tmd451;



    // 3.2 sum
    std::cout << "-----------------[3.2] sum-----------------" << std::endl;

    Tensor* ty369 = Tensor::rand({4, 4}, 5.0f);

    ty369->print();

    // 每一行 第一个元素 相加
    Tensor* ty3692 = ty369->sum({0}, false);

    // 每一列 第一个元素 相加
    Tensor* ty3693 = ty369->sum({1}, false);
    ty3692->print();
    ty3693->print();

    float ty3695 = ty369->sum();
    std::cout << ty3695 << std::endl;

    delete ty369;
    delete ty3692;
    delete ty3693; //delete t8qcv4;

    // 3.3 Comparison operations
    std::cout << "-----------------[3.3] Comparison operations-----------------" << std::endl;
    Tensor* t8ac5 = new Tensor( {0, 0.2, 0, 0, 0, 0.2, 0.2, 0}, {2, 2, 2} );
    Tensor* t8ac51 = Tensor::full(t8ac5->shape, 0.2);

    // 满足条件的相对位置为1 不满足为0
    Tensor* t8ac52 = t8ac5->equal(t8ac51);
    t8ac52->print( );

    delete t8ac5;
    delete t8ac51;

    // 3.3 Comparison operations lequal
    std::cout << "-----------------[3.3] Comparison operations lequal-----------------" << std::endl;
    Tensor* t8ac15 = new Tensor( {0, 0.9, 0, 0, 0, 0.9, 0.2, 0}, {2, 2, 2} );
    Tensor* t8ac151 = Tensor::full(t8ac15->shape, 0.2);

    Tensor* t8ac152 = t8ac15->lequal(t8ac151);
    t8ac152->print( );

    delete t8ac15;
    delete t8ac151;

    // 3.3 Comparison operations lequal
    std::cout << "-----------------[3.3] Comparison operations gequal-----------------" << std::endl;
    Tensor* t8ac105 = new Tensor( {0, 0.09, 0, 0, 0, 0.09, 0.2, 0}, {2, 2, 2} );
    Tensor* t8ac1051 = Tensor::full(t8ac105->shape, 0.2);

    Tensor* t8ac1052 = t8ac105->gequal(t8ac1051);
    t8ac1052->print( );

    delete t8ac105;
    delete t8ac1051;


    // 3.2.1 serialization
    std::cout << "-----------------[3.2.1]serialization-----------------" << std::endl;
    // 序列化 tensor
    Tensor* te00i =  Tensor::rand({6, 3, 2}, 5.0f);
    te00i->print();

    te00i->save("tensor.bin");

    Tensor* te02i = te00i->load("tensor.bin");
    te02i->print();

    // cout
    std::cout << "-----------------cout-----------------" << std::endl;
    std::cout << *te02i << std::endl;
    std::cout << "-----------------size-----------------" << std::endl;
    std::cout << te02i->size() << std::endl;

    std::cout << "-----------------data_ptr-----------------" << std::endl;
    std::cout << te02i->data_ptr() << std::endl;


    delete te00i;
    delete te02i;

    while(true) {

    };
    return 0;
}
