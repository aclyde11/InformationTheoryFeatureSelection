#include <iostream>

#include "mutualinformationlib/mi_lib.h"
#include <Eigen/Dense>
#include <ctime>
#include <random>
#include "mutualinformationlib/Feature.h"
#include <chrono>
using Eigen::VectorXi;



VectorXi randombits(std::mt19937 &generator, int size, double p=0.5) {
    VectorXi vec(size);
    std::bernoulli_distribution distribution(p);
    for (int i = 0; i < size; i++) {
        vec(i) = (int) (distribution(generator));
    }
    return vec;
}

VectorXi random_classes(std::mt19937 &generator,  int size, int classes=37) {
    VectorXi vec(size);
    std::uniform_int_distribution distribution(0, classes);
    for (int i = 0; i < size; i++) {
        vec(i) = (int) (distribution(generator));
    }
    return vec;
}

int main(int argc, char **argv) {
    std::random_device rd;
    std::mt19937 generator(rd());
    int n = std::stoi(argv[1]);
    int m = 11000;

    std::vector<VectorXi> features(n);
    for (int i = 0; i < n; i++)
        features[i] = randombits(generator, m);
    VectorXi c = random_classes(generator, m);

    std::vector<double> v(n * n);
    auto t1 = std::chrono::high_resolution_clock::now();

    mi::JMIM(features, c, std::stoi(argv[2]));


//#pragma omp parallel for simd reduction(max: x)
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            tmp = mi::compute_joint_mutual_information(features[i], features[j], c);
//            if (tmp > x)
//                x = tmp;
//        }
//    }


    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration d = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << d.count() << std::endl;

    return 0;
}