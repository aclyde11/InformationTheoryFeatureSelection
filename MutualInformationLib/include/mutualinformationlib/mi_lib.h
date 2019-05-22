//
// Created by Austin Clyde on 2019-05-21.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_MI_LIB_H
#define INFORMATIONTHEORYFEATURESELECTION_MI_LIB_H

#include <tgmath.h>
#include <Eigen/Dense>
#include "JointProbabilityState.h"
#include "Feature.h"
#include "ProbabilityState.h"
using Eigen::VectorXi;

namespace mi {

    namespace utils {
        VectorXi mergeArrays(VectorXi const& a, VectorXi const& b) {
            int astates = a.maxCoeff() + 1;
            int bstates = b.maxCoeff() + 1;
            VectorXi newStates(astates * bstates);
            VectorXi output(a.size());
            output.setZero();
            newStates.setZero();

            int curIndex;
            int stateCount = 1;
            for (int i = 0; i < a.size(); i++) {
                curIndex = a[i] + (b[i] * astates);
                if (newStates[curIndex] == 0) {
                    newStates[curIndex] = stateCount;
                    stateCount++;
                }
                output[i] = newStates[curIndex];
            }

            return output;
        }
    }



    double compute_mutual_information(JointProbabilityState &state) {
        double mutual_information = 0;
        for (int x = 0; x < state.numFirstStates(); x++) {
            for (int y = 0; y < state.numSecondStates(); y++) {
                if (state.jointProbabilityVector(x, y) > 0 && state.firstProbabilityVector(x) > 0 and
                    state.secondProbabilityVector(y) > 0) {
                    mutual_information += state.jointProbabilityVector(x, y) *
                                          (state.jointProbabilityVector(x, y) / (state.firstProbabilityVector(x) *
                                                                                 state.secondProbabilityVector(y)));
                }
            }
        }

        return mutual_information;
    }

    double compute_mutual_information(VectorXi const& X, VectorXi const& Y) {
        double mutual_information = 0;
        JointProbabilityState state(X, Y);
        for (int x = 0; x < state.numFirstStates(); x++) {
            for (int y = 0; y < state.numSecondStates(); y++) {
                if (state.jointProbabilityVector(x, y) > 0 && state.firstProbabilityVector(x) > 0 and
                    state.secondProbabilityVector(y) > 0) {
                    mutual_information += state.jointProbabilityVector(x, y) *
                                          (state.jointProbabilityVector(x, y) / (state.firstProbabilityVector(x) *
                                                                                 state.secondProbabilityVector(y)));
                }
            }
        }

        return mutual_information;
    }

    double compute_entropy(ProbabilityState &state) {
        double entropy = 0;
        for (int i = 0; i < state.numStates(); i++) {
            if (state.prob(i) > 0) {
                entropy -= state.prob(i) * log(state.prob(i));
            }
        }
        return entropy;
    }

    double compute_conditional_entropy(JointProbabilityState &state) {
        double entropy = 0;

        for (int x = 0; x < state.numFirstStates(); x++) {
            for (int y = 0; y < state.numSecondStates(); y++) {

                if (state.jointProbabilityVector(x, y) > 0 and state.secondProbabilityVector(y) > 0)
                    entropy -= state.jointProbabilityVector(x, y) *
                               log(state.jointProbabilityVector(x, y) / state.secondProbabilityVector(y));
            }
        }
        return entropy;
    }


    double compute_conditional_mutual_information(VectorXi const& X, VectorXi const& Y, VectorXi const& Z) {
        VectorXi merged_arr = utils::mergeArrays(Y, Z);
        JointProbabilityState cond1(X, Z);
        JointProbabilityState cond2(X, merged_arr);
        return compute_conditional_entropy(cond1) - compute_conditional_entropy(cond2);
    }

    double compute_conditional_mutual_information(Feature::Feature const& X , Feature::Feature const& Y, Feature::Feature const& Z) {
        VectorXi merged_arr = utils::mergeArrays(Y.vec, Z.vec);
        JointProbabilityState cond1(X, Z);
        JointProbabilityState cond2(X.vec, merged_arr);
        return compute_conditional_entropy(cond1) - compute_conditional_entropy(cond2);
    }

    double compute_joint_mutual_information(VectorXi const& X1, VectorXi const& X2, VectorXi const& Y) {
        return compute_conditional_mutual_information(X1, Y, X2) + compute_mutual_information(X2, Y);
    }

    double compute_joint_mutual_information(Feature::Feature const& X1, Feature::Feature const& X2, Feature::Feature const& Y) {
        return compute_conditional_mutual_information(X1, Y, X2) + compute_mutual_information(X2.vec, Y.vec);
    }

}


#endif //INFORMATIONTHEORYFEATURESELECTION_MI_LIB_H
