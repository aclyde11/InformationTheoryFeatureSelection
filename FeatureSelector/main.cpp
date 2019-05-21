#include <iostream>

#include "mutualinformationlib/mi_lib.h"
#include <Eigen/Dense>

using Eigen::Vector4i;


int main() {
    Vector4i a, b;
    a << 0, 1, 1, 1;
    b << 0, 1, 1, 1;

    mi::JointProbabilityState js(a, b);
    std::cout << mi::compute_mutual_information(js) << std::endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}