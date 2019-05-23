#include <iostream>

#include <Eigen/Dense>
#include <ctime>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>

#include "mutualinformationlib/mi_lib.h"

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

std::vector<VectorXd> load_csv_double (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;


    VectorXi y;
    std::vector<double> values;
    uint rows = 0;
    uint cols = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
            cols++;
        }
        ++rows;
    }
    cols = cols / rows;
    std::vector<VectorXd> features(cols);
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

    std::vector<VectorXd> features = load_csv_double(argv[1]);
    VectorXi y = features[std::stoi(argv[2])].cast<int>();
    features.erase(features.begin() + std::stoi(argv[2]));

    std::cout << "doign disc" << std::endl;
    std::vector<VectorXi> disc_features(features.size());
    std::vector<int> vec_y(y.data(), y.data() + y.rows() * y.cols());

#pragma omp parallel for
    for (int i = 0; i < features.size(); i++) {
        VectorXd tmp = features[i];
        std::vector<double> vec(tmp.data(), tmp.data() + tmp.rows() * tmp.cols());
        auto cut_points = mi::MDLPDiscretize(vec, vec_y);
        disc_features[i] = mi::bin(tmp, cut_points);
    }
    std::cout << "done" << std::endl;

    auto p = mi::JMIM(disc_features, y, std::stoi(argv[3]));
    std::vector<bool> feature_mask = p.first;
    std::vector<int> ordering = p.second;

//#pragma omp parallel for simd reduction(max: x)
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            tmp = mi::compute_joint_mutual_information(features[i], features[j], c);
//            if (tmp > x)
//                x = tmp;
//        }
//    }

    for (int i = 0; i < ordering.size(); i++) {
        std::cout << i << ": " << ordering[i] << std::endl;
    }


    return 0;
}