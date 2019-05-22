#include <iostream>

#include "mutualinformationlib/mi_lib.h"
#include <Eigen/Dense>
#include <ctime>
#include <random>
#include "mutualinformationlib/Feature.h"
#include <chrono>
#include <vector>
#include <fstream>
using Eigen::VectorXi;



VectorXi randombits(std::mt19937 &generator, int size, double p=0.5) {
    VectorXi vec(size);
    std::bernoulli_distribution distribution(p);
    for (int i = 0; i < size; i++) {
        vec(i) = (int) (distribution(generator));
    }
    return vec;
}

VectorXi random_classes(std::mt19937 &generator,  int size, int classes=10) {
    VectorXi vec(size);
    std::uniform_int_distribution<int> distribution(0, classes);
    for (int i = 0; i < size; i++) {
        vec(i) = (int) (distribution(generator));
    }
    return vec;
}

// read CSV of ints.



std::vector<VectorXi> load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;


    VectorXi y;
    std::vector<int> values;
    uint rows = 0;
    uint cols = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stoi(cell));
            cols++;
        }
        ++rows;
    }
    cols = cols / rows;
    std::vector<VectorXi> features(cols);
    for (int i = 0; i < cols; i++) {
        features[i].resize(rows);
    }

        int f;
    for (int i = 0; i < values.size(); i++) {
        f = i % cols;
        features[f](i / cols) =  values[i];
    }
    return features;
}

int main(int argc, char **argv) {
    std::random_device rd;
    std::mt19937 generator(rd());

    std::vector<VectorXi> features = load_csv(argv[1]);
    VectorXi y = features[std::stoi(argv[2])];
    features.erase(features.begin() + std::stoi(argv[2]));



    auto feature_mask = mi::JMIM(features, y, std::stoi(argv[3]));


//#pragma omp parallel for simd reduction(max: x)
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            tmp = mi::compute_joint_mutual_information(features[i], features[j], c);
//            if (tmp > x)
//                x = tmp;
//        }
//    }

    std::vector<int> set_indicies;
    for (int i = 0; i < feature_mask.size(); i++) {
        if (feature_mask[i]) {
            set_indicies.push_back(i);
            std::cout << i << std::endl;
        }
    }

    std::cout << set_indicies.size() << std::endl;


    return 0;
}