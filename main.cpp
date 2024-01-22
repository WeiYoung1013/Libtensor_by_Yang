#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32
#include <omp.h>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include "src/tensor.h"
using namespace chrono;
int main() {

#ifdef _WIN32
    //SetConsoleOutputCP(CP_UTF8);
#endif // _WIN32

    //初始化随机数种子
    srand((unsigned)(100));
    int fi=1;
    // 1. 生成 2 * 2 张量

    std::cout << "-----------------[1.1.1] create 2 * 2-----------------" << std::endl;
    Tensor<float>* t8s3g = new Tensor<float>({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    t8s3g->print();


    delete t8s3g;

    // 2. 生成 2 * 3 * 3 张量
    std::cout << "-----------------[1.1.2] create 2 * 3 * 3-----------------" << std::endl;
    Tensor<float>* t63g0 = new Tensor<float>({-0.5792, -0.1372,  -0.5792, -0.1372, 0.5962,  1.2097,
                                              -0.5792, -0.1372,  -0.5792, -0.1372, 0.5962,  1.2097,
                                              -0.5792, -0.1372,  -0.5792, -0.1372, 0.5962,  1.2097}, {2, 3, 3} );
    t63g0->print();
    delete t63g0;



    //  3 * 2 * 2 随机 张量
    std::cout << "-----------------[1.2.1] create 3 * 2 * 2 rand-----------------" << std::endl;

    Tensor<double>* tgzxg = Tensor<double>::rand({3, 2, 2}, 5.0);
    Tensor<int>* tgzg = Tensor<int>::rand({1, 1, 3}, 5.0);
    tgzxg->print();
    tgzg->print();
    delete tgzxg;



    // 3. 4 * 2 * 2 全0 张量
    std::cout << "-----------------[1.3.1] create 4 * 2 * 2 zero-----------------" << std::endl;
    Tensor<float>* temno = new Tensor<float>({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                             }, {4, 2, 2});
    Tensor<float>* tdzja = Tensor<float>::zeros({4, 2, 2});

    temno->print();
    delete temno;

    tdzja->print();
    delete tdzja;

    // [1.3.6] 2, 2, 2 全1 张量
    std::cout << "-----------------[1.3.2] create 2, 2, 2 full 1 ----------------" << std::endl;

    Tensor<int>* tlcj6 = Tensor<int>::ones({2, 2, 2});
    tlcj6->print();
    delete tlcj6;

    // [1.3.6] 2, 2, 2 full5 张量
    std::cout << "-----------------[1.3.6] create 2, 2, 2 full 5 ----------------" << std::endl;

    Tensor<float>* tbwsq = Tensor<float>::full({2, 2, 2}, 5);
    tbwsq->print();
    delete tbwsq;

    //  [1.4.1] create 4 * 4  eye
    std::cout << "-----------------[1.4.1] create 4 * 4  eye-----------------" << std::endl;
    Tensor<float>* tm4pf = Tensor<float>::eye(4);
    tm4pf->print();
    delete tm4pf;

    //  2.1 index and select
    std::cout << "-----------------[2.1] index-----------------" << std::endl;

    Tensor<double>* tw109 = Tensor<double>::rand({4, 6});
    tw109->print();

    Tensor<double>* tw1092 = tw109->select({"0", "2"});  // 取 (0,2) 的数据
    tw1092->print();
    cout<<tw109->data_ptr()<<endl;
    cout<<tw1092->data_ptr()<<endl;

    Tensor<double>* tw1093 = tw109->select({"1", ":"});  // 取 第2行 的数据
    tw1093->print();

    Tensor<double>* tw1094 = tw109->select({":", "2"});  // 取 第3列 的数据
    tw1094->print();

    Tensor<double>* tw1095 = tw109->select({"0:2", "0:2"}); // 取前2列的前2行
    tw1095->print();

    delete tw109;
    delete tw1092;
    delete tw1093;
    delete tw1094;
    delete tw1095;

    //  2.2 join and tile
    std::cout << "-----------------[2.2] join and tile-----------------" << std::endl;

    //Tensor* tsxfy = new Tensor({1, 2}, {2, 1});
    Tensor<float>* tsxfy = new Tensor<float>({1, 1, 2, 2, 1, 1, 2, 2}, {4, 2});
    tsxfy->print();

    //
    Tensor<float>* tsxfy1 = Tensor<float>::tile(tsxfy, {2, 2});
    tsxfy1->print();

    Tensor<float>* tsxfy2 = Tensor<float>::full({2, 2, 2}, 2);
    Tensor<float>* tsxfy3 = Tensor<float>::full({2, 2, 2}, 5);

    // 拼接
    Tensor<float>* tsxfy4 = Tensor<float>::concat({tsxfy2, tsxfy3});

    tsxfy4->print();

    delete tsxfy;
    delete tsxfy1;
    delete tsxfy2;
    delete tsxfy3;
    delete tsxfy4;

    // Mutating
    std::cout << "-----------------[2.3] Mutating-----------------" << std::endl;

    Tensor<int>* tlchy = Tensor<int>::ones({4, 4});
    tlchy->set_select({"0", "2"}, 2.0f); // 把(0, 2) 设置 2
    tlchy->print();

    Tensor<int>* tlchy2 = Tensor<int>::ones({4, 4});
    tlchy2->set_select({":", "2"}, 5.0f);  // 第3列设置为 5
    tlchy2->print();

    Tensor<int>* tlchy3 = Tensor<int>::ones({4, 4});
    tlchy3->set_select({"0:1", "0:1"}, 7.0f);  // 前两行前两列 为 7
    tlchy3->print();

    delete tlchy;
    delete tlchy2;
    delete tlchy3;


    // permute
    std::cout << "-----------------[2.4] permute-----------------" << std::endl;
    Tensor<int>* tuzku = Tensor<int>::rand({3, 4}, 7.0);
    tuzku->print();

    // 转置 二维矩阵 3,4 -> 4，3
    Tensor<int> *tuzku2 = Tensor<int>::permute(tuzku, {1, 0});
    tuzku2->print();
    Tensor<int> *tuzku3 = Tensor<int>::transpose(tuzku,{0,1});
    tuzku3->print();
    delete tuzku;
    delete tuzku2;


    // view
    std::cout << "-----------------[2.5] view-----------------" << std::endl;

    Tensor<float>* tod8r = Tensor<float>::rand({4, 3, 2}, 1.0);
    tod8r->print( );

    // 维度变换 4*3*2 变换 2*2*2
    tod8r->reshape_({2, 3, 4});
    tod8r->print( );

    delete tod8r;


    // 3.1 add
    std::cout << "-----------------[3.1] add-----------------" << std::endl;

    Tensor<float>* t8qcv = Tensor<float>::ones({3, 4});
    Tensor<float>* t8qcv2 = Tensor<float>::ones({3, 4});
    // 矩阵对应元素相加
    Tensor<float>* t8qcv3 = Tensor<float>::add(t8qcv, t8qcv2);
    Tensor<float>* t8qcv4 = Tensor<float>::add(t8qcv3, t8qcv);
    t8qcv4->print();

    Tensor <float>qcv5 = Tensor<float>({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor<float> qcv6 = Tensor<float>({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor<float> qcv7 = qcv5 + qcv6;
    qcv7.print();

    delete t8qcv;
    delete t8qcv2;
    delete t8qcv3; delete t8qcv4;
    //delete qcv5;delete qcv6;delete qcv7;

    // 3.1 sub
    std::cout << "-----------------[3.1] sub-----------------" << std::endl;

    Tensor<float>* t8qcs = Tensor<float>::ones({3, 4});
    Tensor<float>* t8qcs2 = Tensor<float>::ones({3, 4});
    // 矩阵对应元素相加
    Tensor<float>* t8qcs3 = Tensor<float>::sub(t8qcs, t8qcs2);
    Tensor<float>* t8qcs4 = Tensor<float>::sub(t8qcs3, t8qcs);
    t8qcs4->print();

    Tensor <float>qcv25 = Tensor<float>({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor <float>qcv26 = Tensor<float>({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor<float> qcv27 = qcv25 + qcv26;
    qcv27.print();

    delete t8qcs;
    delete t8qcs2;
    delete t8qcs3;
    delete t8qcs4;

    // 3.1 mul
    std::cout << "-----------------[3.1] mul-----------------" << std::endl;

    Tensor<float>* changshu = Tensor<float>::rand({3,3});
    changshu->print();
    Tensor<float>* changshu2=changshu->mul(2.0f);
    changshu2->print();

    Tensor<float>* t8qcsq = Tensor<float>::rand({3, 4});
    Tensor<float>* t8qcsq2 = Tensor<float>::rand({4, 5});
    // 矩阵对应元素相加
    Tensor<float>* t8qcsq3 = t8qcsq->mul(t8qcsq2);
    t8qcsq3->print();

    Tensor <float>qcv2d5 = Tensor<float>({-0.1372, -0.1372,   0.2097,  1.2097}, {2,2} );
    Tensor<float> qcv2d6 = Tensor<float>({-0.5792, -0.1372,   0.5962,  1.2097}, {2,2} );
    // 运算符重载
    Tensor <float>qcv2d7 = qcv2d5 * qcv2d6;
    qcv2d7.print();

    delete t8qcsq;
    delete t8qcsq2;
    delete t8qcsq3;

    // 3.1 div
    std::cout << "-----------------[3.1] div-----------------" << std::endl;

    Tensor<double>* t8qcaq = Tensor<double>::rand({3, 4});
    Tensor<double>* t8qcaq2 = Tensor<double>::rand({1, 4});
    Tensor<double>* t8qcaq3 = t8qcaq->div(t8qcaq2);
    t8qcaq3->print();

    Tensor <float>qcv2qd5 = Tensor<float>({-0.3711, -1.9353, -0.4605, -0.2917,0.1815, -1.0111,  0.9805, -1.5923, 0.1062,  1.4581,  0.7759, -1.2344,-0.1830, -0.0313,  1.1908, -1.4757}, {4,4} );
    Tensor <float>qcv2qd6 = Tensor<float>({0.8032,  0.2930, -0.8113, -0.2308}, {1,4} );
    // 运算符重载
    Tensor <float>qcv2qd7 = qcv2qd5 / qcv2qd6;
    qcv2qd7.print();

    delete t8qcaq;
    delete t8qcaq2;
    delete t8qcaq3;



    // 3.1.1 log
    std::cout << "-----------------[3.1.1] log-----------------" << std::endl;
    Tensor<double>* tmd45 = Tensor<double>::rand({4, 4}, 5.0f);
    tmd45->print();
    // 元素对2取对数
    Tensor<double>* tmd451 = tmd45->log2();
    tmd451->print();
    delete tmd45;
    delete tmd451;
    Tensor<double>* tmd46 = Tensor<double>::rand({4, 4}, 5.0f);
    tmd46->print();
    Tensor<double>* tmd461 = tmd46->log10();
    tmd461->print();
    delete tmd46;
    delete tmd461;

    Tensor<double>* tmd47 = Tensor<double>::rand({4, 4}, 5.0f);
    tmd47->print();
    Tensor<double>* tmd471 = tmd47->logn(3);
    tmd471->print();
    delete tmd47;
    delete tmd471;




    // 3.2 sum
    std::cout << "-----------------[3.2] sum-----------------" << std::endl;

    Tensor<float>* ty369 = Tensor< float>::rand({4, 4}, 5.0f);

    ty369->print();

    // 每一行 第一个元素 相加
    Tensor< float>* ty3692 = ty369->sum({0}, false);

    // 每一列 第一个元素 相加
    Tensor< float>* ty3693 = ty369->sum({1}, false);
    ty3692->print();
    ty3693->print();

    float ty3695 = ty369->sum();
    std::cout << ty3695 << std::endl;

    delete ty369;
    delete ty3692;
    delete ty3693; //delete t8qcv4;
    std::cout << "-----------------[3.2.1,2,3] mean,max,min-----------------" << std::endl;

    Tensor<float>* ten= Tensor< float>::rand({2, 2}, 5.0f);
    ten->print();
    float meanValue = ten->mean();
    float maxValue = ten->max();
    float minValue = ten->min();
    std::cout << "Mean value: " << meanValue << std::endl;
    std::cout << "Max value: " << maxValue << std::endl;
    std::cout << "Min value: " << minValue << std::endl;
    delete ten; //delete t8qcv4;

    // 3.3 Comparison operations
    std::cout << "-----------------[3.3] Comparison operations-----------------" << std::endl;
    Tensor<double>* t8ac5 = new Tensor<double>( {0, 0.2, 0, 0, 0, 0.2, 0.2, 0}, {2, 2, 2} );
    Tensor<double>* t8ac51 = Tensor<double>::full(t8ac5->shape, 0.2);

    // 满足条件的相对位置为1 不满足为0
    Tensor<double>* t8ac52 = t8ac5->equal(t8ac51);
    t8ac52->print( );

    delete t8ac5;
    delete t8ac51;

    // 3.3 Comparison operations lequal
    std::cout << "-----------------[3.3] Comparison operations lequal-----------------" << std::endl;
    Tensor<float>* t8ac15 = new Tensor<float>( {0, 0.9, 0, 0, 0, 0.9, 0.2, 0}, {2, 2, 2} );
    Tensor<float>* t8ac151 = Tensor<float>::full(t8ac15->shape, 0.2);

    Tensor<float>* t8ac152 = t8ac15->lequal(t8ac151);
    t8ac152->print( );

    delete t8ac15;
    delete t8ac151;

    // 3.3 Comparison operations lequal
    std::cout << "-----------------[3.3] Comparison operations gequal-----------------" << std::endl;
    Tensor<float>* t8ac105 = new Tensor<float>( {0, 0.09, 0, 0, 0, 0.09, 0.2, 0}, {2, 2, 2} );
    Tensor<float>* t8ac1051 = Tensor<float>::full(t8ac105->shape, 0.2);

    Tensor<float>* t8ac1052 = t8ac105->gequal(t8ac1051);
    t8ac1052->print( );

    delete t8ac105;
    delete t8ac1051;
    delete t8ac1052;
    // 3.3 Comparison operations nequal
    std::cout << "-----------------[3.3] Comparison nequal-----------------" << std::endl;
    Tensor<double>* t8ac50 = new Tensor<double>( {0, 0.2, 0, 0, 0, 0.2, 0.2, 0}, {2, 2, 2} );
    Tensor<double>* t8ac501 = Tensor<double>::full(t8ac50->shape, 0.2);

    // 满足条件的相对位置为1 不满足为0
    Tensor<double>* t8ac502 = t8ac50->nequal(t8ac501);
    t8ac502->print( );

    delete t8ac50;
    delete t8ac501;
    delete t8ac502;
    std::cout << "-----------------[3.3] Comparison le and lt Operation -----------------" << std::endl;
    Tensor<double>* tensor1 = new Tensor<double>({0.5, 1.0, 1.5, 2.0}, {2, 2});
    double constant = 1.0;

// 比较 tensor1 中的每个元素是否小于等于 constant
    Tensor<bool> result1 = tensor1->le(constant);
    Tensor<bool> result3 = tensor1->lt(constant);
    result1.print();
    result3.print();

    delete tensor1;
    std::cout << "-----------------[Test le lt Operation with Tensor]-----------------" << std::endl;
    Tensor<int>* tensor2 = new Tensor<int>({1, 2 ,3, 4}, {2, 2});
    Tensor<int>* tensor3 = new Tensor<int>({1, 1, 4, 4}, {2, 2});

// 比较 tensor2 和 tensor3 中相应位置的元素
    Tensor<bool>result2 = tensor2->le(*tensor3);
    Tensor<bool>result4 = tensor2->lt(*tensor3);
    result2.print();
    result4.print();

    delete tensor2;
    delete tensor3;









    //3.4 Einsum operations
    std::cout << "-----------------[3.4] Einsum operations -----------------" << std::endl;
    Tensor<float>* t1 = Tensor<float>::rand({3, 4}, 1.0);
    Tensor<float>* t2 = Tensor<float>::rand({3, 4}, 1.0);
    Tensor<float>* t3 = Tensor<float>::rand({1, 4}, 1.0);
    vector<Tensor<float>*>first;
    vector<Tensor<float>*>second;
    vector<Tensor<float>*>ss;
    vector<Tensor<int>*>ssd;
    first.push_back(t1);
    second.push_back(t1);
    second.push_back(t2);
    t1->print();
    t2->print();
    std::cout << "-----------------[3.4.1] Einsum operations ii->i-----------------" << std::endl;
    Tensor<float>* t5 = Tensor<float>::einsum("ss->s", first); // This computes the diagonal of t1.
    t5->print();
    std::cout << "------- if the input string is wrong, then we just return the original string-------" << std::endl;
    //if the input string is wrong, then we just return the original string
    Tensor<float>* t6 = Tensor<float>::einsum("ss->a", first); // This computes the diagonal of t1.
    t6->print();
    std::cout << "-----------------[3.4.2] Einsum operations ij->ji-----------------" << std::endl;

    Tensor<float>* t9 = Tensor<float>::einsum("gy->yg", first); // 矩阵反转
    t9->print();
    std::cout << "-----------------[3.4.3] Einsum operations ...ij->...ji-----------------" << std::endl;
    Tensor<int>* t222 = Tensor<int>::rand({2,2, 4}, 4.0);
    t222->print();
    vector<Tensor<int>*>sssssd;
    sssssd.push_back(t222);
    Tensor<int>* t1000 = Tensor<int>::einsum("...gj->...jg", sssssd); // 矩阵反转
    t1000->print();
    std::cout << "-----------------[3.4.4] Einsum operations ij->-----------------" << std::endl;
    Tensor<float>* t88 = Tensor<float>::einsum("gy->", first); // 求和
    t88->print();
    std::cout << "-----------------[3.4.5] Einsum operations ij->i-----------------" << std::endl;
    Tensor<float>* t7 = Tensor<float>::einsum("sa->a", first); // 行列求和
    t7->print();
    Tensor<float>* t8 = Tensor<float>::einsum("gy->g", first); // 行列求和
    t8->print();
    std::cout << "-----------------[3.4.6] Einsum operations ik,k->i-----------------" << std::endl;
    Tensor <float> *qcv= new Tensor<float>({0, 1.5, 2.5, 3.5, 4.5, 5.5}, {2, 3} );
    Tensor <float>*qcv2 = new Tensor<float>({0,1.5,2.5}, {1,3} );
    ss.push_back(qcv);
    ss.push_back(qcv2);
    Tensor<float>* t111 = Tensor<float>::einsum("ig,g->i", ss);
    ss.clear();
    t111->print();
    std::cout << "-----------------[3.4.7] Einsum operations ik,kj->ij-----------------" << std::endl;
    Tensor <int> *q1= new Tensor<int>({0, 1, 2, 3, 4, 5}, {2, 3} );
    Tensor <int>*q2 = new Tensor<int>({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}, {3,5} );
    ssd.push_back(q1);
    ssd.push_back(q2);
    Tensor<int>* t = Tensor<int>::einsum("ab,bc->ac", ssd);
    t->print();
    std::cout << "-----------------[3.4.8] Einsum operations i,i->i-----------------" << std::endl;
    Tensor<float>* t12 = Tensor<float>::rand({1, 4}, 1.0);
    t12->print();
    Tensor<float>* t13= Tensor<float>::rand({1, 4}, 1.0);
    t13->print();
    vector<Tensor<float>*>third;
    third.push_back(t12);
    third.push_back(t13);
    Tensor<float>* t11 = Tensor<float>::einsum("i,i->", third); // 点乘
    t11->print();
    std::cout << "-----------------[3.4.9] Einsum operations ij,ij>-----------------" << std::endl;
    Tensor<float>* t10 = Tensor<float>::einsum("ij,ij->", second); // 内积
    t10->print();
    std::cout << "-----------------[3.4.10] Einsum operations i,j->ij>-----------------" << std::endl;
    ssd.clear();
    Tensor <int> *q101= new Tensor<int>({0, 1, 2}, {1, 3} );
    Tensor <int>*q102 = new Tensor<int>({3,4,5,6}, {1,4} );
    ssd.push_back(q101);
    ssd.push_back(q102);
    Tensor<int>* t100 = Tensor<int>::einsum("i,j->ij", ssd); // 内积
    t100->print();
    std::cout << "-----------------[3.4.11] Einsum operations  Batch matrix mul, ijk,ikl->ijl-----------------" << std::endl;
    vector<Tensor<int>*>t34;
    Tensor<int>* t341 = Tensor<int>::rand({2,3,4}, 3.0);
    Tensor<int>* t342 = Tensor<int>::rand({2,4,3}, 3.0);
    t34.push_back(t341);
    t34.push_back(t342);
    t341->print();
    t342->print();
    Tensor<int>* t330=Tensor<int>::einsum("ijk,ikl->ijl",t34);
    t330->print();
    delete t341;
    delete t342;
    delete t330;
    delete t100;
    delete q101;
    delete q102;
    delete t10;
    delete t11;
    delete t5;
    delete t6;
    delete t7;
    delete t8;
    delete t9;
    delete t88;
    // 3.2.1 serialization

    std::cout << "-----------------[3.2.1]serialization-----------------" << std::endl;
    // 序列化 tensor
    Tensor<float>* te00i =  Tensor<float>::rand({6, 3, 2}, 5.0f);
    te00i->print();

    te00i->save("tensor.txt");
    cout<<"finish save"<<endl;

    Tensor<float>* te02i = Tensor<float>::load("tensor.txt");
    cout<<"finish load"<<endl;
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




    std::cout << "-----------------[Test Gradient Operations]-----------------" << std::endl;
    Tensor<float>* tensorA = new Tensor<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});

    // 创建另一个张量用于操作
    Tensor<float>* tensorB = new Tensor<float>({0.5, 1.5, 2.5, 3.5}, {2, 2});

    // 执行加法操作，并打印梯度
    Tensor<float>* tensorAdd = tensorA->add(tensorB);
    tensorAdd->print();
    tensorAdd->getGradient()->print();
    tensorA->grad_history.push_back(tensorAdd->getGradient());
    tensorA->op_history.push_back(OperationType::Add);

    // 执行减法操作，并打印梯度
    Tensor<float>* tensorSub = tensorA->sub(tensorB);
    tensorSub->print();
    tensorSub->getGradient()->print();
    tensorA->grad_history.push_back(tensorSub->getGradient());
    tensorA->op_history.push_back(OperationType::Subtract);


    // 执行乘法操作，并打印梯度
    Tensor<float>* tensorMul = tensorA->mul(tensorB);
    tensorMul->print();
    tensorMul->getGradient()->print();
    tensorA->grad_history.push_back(tensorMul->getGradient());
    tensorA->op_history.push_back(OperationType::Multiply);

    // 执行除法操作，并打印梯度
    Tensor<float>* tensorDiv = tensorA->div(tensorB);
    tensorDiv->print();
    tensorDiv->getGradient()->print();
    tensorA->grad_history.push_back(tensorDiv->getGradient());
    tensorA->op_history.push_back(OperationType::Divide);

    // 执行对数操作，并打印梯度
    Tensor<float>* tensorLog2 = tensorA->log2();
    tensorLog2->print();
    tensorLog2->getGradient()->print();
    tensorA->grad_history.push_back(tensorLog2->getGradient());
    tensorA->op_history.push_back(OperationType::Log2);

    Tensor<float>* tensorLog10 = tensorA->log10();
    tensorLog10->print();
    tensorLog10->getGradient()->print();
    tensorA->grad_history.push_back(tensorLog10->getGradient());
    tensorA->op_history.push_back(OperationType::Log10);

    Tensor<float>* tensorLogn = tensorA->logn(3);
    tensorLogn->print();
    tensorLogn->getGradient()->print();
    tensorA->grad_history.push_back(tensorLogn->getGradient());
    tensorA->op_history.push_back(OperationType::Logn);






    std::cout << "-----------------[Test Gradient-history]-----------------" << std::endl;

    tensorA->backward();

    delete tensorA;
    delete tensorB;
    delete tensorAdd;
    delete tensorSub;
    delete tensorMul;
    delete tensorDiv;
    delete tensorLog2;
    delete tensorLog10;
    delete tensorLogn;


    std::cout << "-----------------[Test Determinant of Square Matrix]-----------------" << std::endl;
    Tensor<float>* tensorSquare = new Tensor<float>({1.0, 2.0, 3.0, 4.0}, {2, 2});
    cout<<tensorSquare->determinant();
    cout<<endl;

    Tensor<float>* tensorSquare2 = new Tensor<float>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0},{3, 3});
    tensorSquare2->print();
    cout<<tensorSquare2->determinant();
    cout<<endl;


    Tensor<float>* tensorNonSquare = new Tensor<float>({0.5, 1.5, 2.5, 3.5, 4.5}, {2, 3});
    cout<<tensorNonSquare->determinant()<<endl;
    cout<<"---------acceleration with openMp--------"<<endl;
    auto start_serial = high_resolution_clock::now();
    Tensor<float> v7 = qcv5 + qcv6;
    Tensor <float>v2d7 = qcv2d5 * qcv2d6;
    Tensor <float>v2qd7 = qcv2qd5 / qcv2qd6;
    for (int i = 0; i < 100000; ++i) {
        v7=v7+qcv5;
        v7=v7+qcv6;
    }
    for (int i = 0; i < 100000; ++i) {
        v2d7=v2d7*qcv2d5*qcv2d6;
    }
    for (int i = 0; i < 100000; ++i) {
        v2d7=v2d7/qcv2d5;
        v2d7=v2d7/qcv2d6;
    }
    Tensor<float> v8 = qcv5 + qcv6;
    for (int i = 0; i < 100000; ++i) {
        v8=v8-qcv5;
        v8=v8-qcv6;
    }
    auto stop_serial = high_resolution_clock::now();
    auto duration_serial = duration_cast<microseconds>(stop_serial - start_serial);

    omp_set_num_threads(4);
    // 使用OpenMP并行化运算
    auto start_parallel = high_resolution_clock::now();
#pragma omp parallel default(none) shared(cout, qcv5, qcv6,qcv2d5,qcv2d6)
    {
#pragma omp sections
        {
#pragma omp section
            { Tensor<float> v88 = qcv5 + qcv6;
                // 并行化加法运算
#pragma omp parallel for default(none) shared(cout, qcv5, qcv6, v88)
                for (int i = 0; i < 100000; ++i) {
                    v88=v88+qcv5;
                    v88=v88+qcv6;
                }

#pragma omp critical
                {

                }
            }

#pragma omp section
            {    Tensor <float>v2d88 = qcv2d5 * qcv2d6;
                // 并行化乘法运算

#pragma omp parallel for default(none) shared(cout, qcv2d5 , qcv2d6, v2d88)
                for (int i = 0; i < 100000; ++i) {
                    v2d88=qcv2d5*v2d88;
                    v2d88= qcv2d6* v2d88;
                }
#pragma omp critical
                {

                }
            }

#pragma omp section
            {
                // 并行化减法运算
                Tensor<float> v887 = qcv5 - qcv6;
#pragma omp parallel for default(none) shared(cout, qcv5 , qcv6, v887)
                for (int i = 0; i < 100000; ++i) {
                    v887=v887-qcv5;
                    v887=v887-qcv6;
                }
#pragma omp critical
                {

                }
            }

#pragma omp section
            {
                // 并行化除法运算
                Tensor <float>v2qd88 = qcv5  / qcv6;
#pragma omp parallel for default(none) shared(cout, qcv5, qcv6, v2qd88)
                for (int i = 0; i < 100000; ++i) {
                    v2qd88=v2qd88/qcv5;
                    v2qd88=v2qd88/qcv6;
                }
#pragma omp critical
                {
                }
            }
        }
    }
    auto stop_parallel = high_resolution_clock::now();
    auto duration_parallel = duration_cast<microseconds>(stop_parallel - start_parallel);
    cout << "raw Duration: " << duration_serial.count() << " microseconds" << endl;
    cout << "openmp Duration: " << duration_parallel.count() << " microseconds" << endl;
    return 0;
}
