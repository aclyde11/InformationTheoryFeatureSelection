//
// Created by Austin Clyde on 2019-05-21.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_MI_LIB_H
#define INFORMATIONTHEORYFEATURESELECTION_MI_LIB_H

#include <tgmath.h>
#include "JointProbabilityState.h"


namespace mi {
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
}


#endif //INFORMATIONTHEORYFEATURESELECTION_MI_LIB_H
