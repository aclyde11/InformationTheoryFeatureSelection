//
// Created by Austin Clyde on 2019-05-21.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_JOINTPROBABILITYSTATE_H
#define INFORMATIONTHEORYFEATURESELECTION_JOINTPROBABILITYSTATE_H

#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXi;
using Eigen::VectorXd;


namespace mi {
    class ProbabilityState {
        VectorXd prob;
    };

    class JointProbabilityState {
    public:
        MatrixXd jointProbabilityVector;
        VectorXd firstProbabilityVector, secondProbabilityVector;


        JointProbabilityState(VectorXi const &first, VectorXi const &second) {
            auto vecLen = (double) first.size();
            auto firstStates = getStateCount(first);
            auto secondStates = getStateCount(second);
            assert (first.size() == second.size());

            VectorXd firstStateCounts(firstStates);
            VectorXd secondStateCounts(secondStates);
            MatrixXd jointStateCounts(firstStates, secondStates);

            firstStateCounts.setZero();
            secondStateCounts.setZero();
            jointStateCounts.setZero();

            jointProbabilityVector.resize(firstStates, secondStates);
            firstProbabilityVector.resize(firstStates);
            secondProbabilityVector.resize(secondStates);

            firstProbabilityVector.setZero();
            secondProbabilityVector.setZero();
            jointProbabilityVector.setZero();


            // Count up states occuring in each vector
            for (int i = 0; i < vecLen; i++) {
                firstStateCounts(first(i)) += 1;
                secondStateCounts(second(i)) += 1;
                jointStateCounts(first(i), second[i]) += 1;
            }

            for (int i = 0; i < firstStates; i++) {
                firstProbabilityVector(i) = firstStateCounts(i) / vecLen;
            }

            for (int i = 0; i < secondStates; i++) {
                secondProbabilityVector(i) = secondStateCounts(i) / vecLen;
            }

            for (int i = 0; i < firstStates; i++) {
                for (int j = 0; j < secondStates; j++) {
                    jointProbabilityVector(i, j) = jointStateCounts(i, j) / vecLen;
                }
            }
        }

        int numFirstStates() {
            return firstProbabilityVector.size();
        }

        int numSecondStates() {
            return secondProbabilityVector.size();
        }

    private:
        int getStateCount(VectorXi const &vec) {
            int states = vec.maxCoeff() + 1;
            assert(states == 2);
            return states;
        }
    };


}


#endif //INFORMATIONTHEORYFEATURESELECTION_JOINTPROBABILITYSTATE_H
