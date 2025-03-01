/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    InvariantKalmanFilter.h
 * @brief   Class to perform Left Invariant Kalman Filtering using nonlinear factor graphs
 * @author  Matthew Kielo
 * @author  Scott Baker
 */

// \callgraph
#pragma once

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace gtsam {

/**
 * This is a Left-Invariant Kalman Filter class implemented using GTSAM's factor graphs.
 * The key difference from the standard EKF is that it operates on Lie groups and uses
 * a left-invariant error definition: eta = x^{-1} * x̂
 *
 * The filter maintains the mean state as a Lie group element and transforms
 * covariances using the adjoint map.
 *
 * The class provides a "predict" and "update" function to perform these steps independently.
 */

template <class VALUE>
class InvariantKalmanFilter {
  // Check that VALUE type is a testable Manifold
  GTSAM_CONCEPT_ASSERT(IsTestable<VALUE>);
  GTSAM_CONCEPT_ASSERT(IsLieGroup<VALUE>);

 public:
  typedef std::shared_ptr<InvariantKalmanFilter<VALUE> > shared_ptr;
  typedef VALUE T;

 protected:
  T x_;                                     // linearization point
  JacobianFactor::shared_ptr priorFactor_;  // density on the left-invariant error

  static T solve_(const GaussianFactorGraph& linearFactorGraph, const Values& linearizationPoints,
                  Key x, JacobianFactor::shared_ptr* newPrior);

 public:
  /// @name Standard Constructors
  /// @{

  InvariantKalmanFilter(Key key_initial, T x_initial, noiseModel::Gaussian::shared_ptr P_initial);

  /// @}
  /// @name Testable
  /// @{

  /// print
  void print(const std::string& s = "") const {
    std::cout << s << "\n";
    x_.print(s + "x");
    priorFactor_->print(s + "density");
  }

  /// @}
  /// @name Interface
  /// @{

  /**
   * Predict step using left-invariant error
   * The motion model should use left-invariant observations
   */
  T predict(const BetweenFactor<T>& motionFactor);

  /**
   * Update step using left-invariant error
   * The measurement model should use  left-invariant observations
   */
  
  T update(const PriorFactor<T>& measurementFactor);

  /// Return current predictive (if called after predict)/posterior (if called after update)
  const JacobianFactor::shared_ptr Density() const {
    return priorFactor_;
  }

  /// @}
};

}  // namespace

#include <gtsam/nonlinear/InvariantKalmanFilter-inl.h>
