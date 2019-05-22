//
// Created by Austin Clyde on 2019-05-21.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_JOINTPROBABILITYSTATE_H
#define INFORMATIONTHEORYFEATURESELECTION_JOINTPROBABILITYSTATE_H

#include <vector>
#include "Feature.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;


namespace mi {

    class JointProbabilityState {
    public:
        MatrixXd jointProbabilityVector;
        VectorXd firstProbabilityVector, secondProbabilityVector;


        JointProbabilityState(VectorXi const &first, VectorXi const &second) {
            auto vecLen = (double) first.size();
            auto firstStates = getStateCount(first);
            auto secondStates = getStateCount(second);
            assert (first.size() == second.size());

            jointProbabilityVector.resize(firstStates, secondStates);
            firstProbabilityVector.resize(firstStates);
            secondProbabilityVector.resize(secondStates);

            firstProbabilityVector.setZero();
            secondProbabilityVector.setZero();
            jointProbabilityVector.setZero();

            // Count up states occuring in each vector
            for (int i = 0; i < vecLen; i++) {
                firstProbabilityVector(first(i)) += 1;
                secondProbabilityVector(second(i)) += 1;
                jointProbabilityVector(first(i), second[i]) += 1;
            }

            firstProbabilityVector = firstProbabilityVector / vecLen;
            secondProbabilityVector = secondProbabilityVector / vecLen;
            jointProbabilityVector = jointProbabilityVector / vecLen;

        }

        JointProbabilityState(Feature::Feature const &first, Feature::Feature const &second) {
            auto vecLen = (double) first.size();
            auto firstStates = first.probs.numStates();
            auto secondStates = second.probs.numStates();
            firstProbabilityVector = first.probs.prob;
            secondProbabilityVector = second.probs.prob;

            jointProbabilityVector.resize(firstStates, secondStates);
            jointProbabilityVector.setZero();

            // Count up states occuring in each vector
            for (int i = 0; i < vecLen; i++) {
                jointProbabilityVector(first.probs.prob(i), second.probs.prob[i]) += 1;
            }

            jointProbabilityVector = jointProbabilityVector / vecLen;

        }

        int numFirstStates() const {
            return firstProbabilityVector.size();
        }

        int numSecondStates() const {
            return secondProbabilityVector.size();
        }

    private:
        int getStateCount(VectorXi const &vec) const {
            int states = vec.maxCoeff() + 1;
            return states;
        }
    };


}


#endif //INFORMATIONTHEORYFEATURESELECTION_JOINTPROBABILITYSTATE_H
