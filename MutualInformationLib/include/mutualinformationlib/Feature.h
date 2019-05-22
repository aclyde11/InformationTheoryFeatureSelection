//
// Created by Austin Clyde on 2019-05-21.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_FEATURE_H
#define INFORMATIONTHEORYFEATURESELECTION_FEATURE_H

#include <Eigen/Dense>
#include "ProbabilityState.h"


namespace Feature {
    class Feature {
    public:
        std::string name;
        int index;
        mi::ProbabilityState probs;
        Eigen::VectorXi vec;


        Feature()  {

        }
      explicit Feature(Eigen::VectorXi const& vec_, int index_=0) : probs(vec) {
          index = index_;
          vec = (vec_);

      }

      int size() const {
          return vec.size();
      }
    };

};

#endif //INFORMATIONTHEORYFEATURESELECTION_FEATURE_H
