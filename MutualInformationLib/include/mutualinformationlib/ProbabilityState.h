//
// Created by Austin Clyde on 2019-05-21.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_PROBABILITYSTATE_H
#define INFORMATIONTHEORYFEATURESELECTION_PROBABILITYSTATE_H
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;


namespace mi {
    class ProbabilityState {
    public:
        VectorXd prob;

        ProbabilityState() {

        }

        explicit ProbabilityState(VectorXi const& vec)  {
            auto vecLen = (double) vec.size();
            auto states =  vec.maxCoeff() + 1;
            prob.resize(states);
            prob.setZero();

            for (int i = 0; i < vecLen; i++) {
                prob(vec(i)) += 1;
            }

            prob = prob / vecLen;
        }

        int numStates() const {
            return prob.size();
        }
    };
}
#endif //INFORMATIONTHEORYFEATURESELECTION_PROBABILITYSTATE_H
