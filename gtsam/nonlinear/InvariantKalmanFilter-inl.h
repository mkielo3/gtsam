/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    InvariantKalmanFilter-inl.h
 * @brief   Class to perform generic Kalman Filtering using nonlinear factor graphs
 * @author  Matthew Kielo
 * @author  Scott Baker
 */

#pragma once

#include <gtsam/nonlinear/InvariantKalmanFilter.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>

#include <cassert>

namespace gtsam {

  /* ************************************************************************* */
  template<class VALUE>
  typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::solve_(
      const GaussianFactorGraph& linearFactorGraph,
      const Values& linearizationPoint, Key lastKey,
      JacobianFactor::shared_ptr* newPrior)
  {
    // Compute the marginal on the last key
    // Solve the linear factor graph, converting it into a linear Bayes Network
    // P(x0,x1) = P(x0|x1)*P(x1)
    const Ordering lastKeyAsOrdering{lastKey};
    const GaussianConditional::shared_ptr marginal =
      linearFactorGraph.marginalMultifrontalBayesNet(lastKeyAsOrdering)->front();

    // Extract the current estimate of x1,P1
    VectorValues result = marginal->solve(VectorValues());
    const T& current = linearizationPoint.at<T>(lastKey);

    // For right-invariant EKF, we use right multiplication for update
    // x = x̂ * exp(xi), where xi is the correction in the Lie algebra
    T x = current.expmap(result[lastKey]);

    // Create a new prior factor for the next iteration
    // The covariance needs to be transformed using the adjoint map
    assert(marginal->nrFrontals() == 1);
    assert(marginal->nrParents() == 0);

    // Transform the information matrix using the adjoint
    Matrix A = marginal->getA(marginal->begin());
    Matrix adjoint = x.AdjointMap();
    Matrix transformed_A = A * adjoint;

    *newPrior = std::make_shared<JacobianFactor>(
      marginal->keys().front(),
      transformed_A,
      marginal->getb() - transformed_A * result[lastKey],
      marginal->get_model());

    return x;
  }

  /* ************************************************************************* */
  template <class VALUE>
  InvariantKalmanFilter<VALUE>::InvariantKalmanFilter(
      Key key_initial, T x_initial, noiseModel::Gaussian::shared_ptr P_initial)
      : x_(x_initial)
  {
    // Create a Jacobian Prior Factor
    // For right-invariant EKF, the error is eta = x̂ * x^{-1}
    int n = traits<T>::GetDimension(x_initial);
    priorFactor_ = std::make_shared<JacobianFactor>(
        key_initial, 
        P_initial->R(),  // Square root information matrix
        Vector::Zero(n), // b vector is zero since x_initial is the mean
        noiseModel::Unit::Create(n));
  }
  
  /* ************************************************************************* */
  template<class VALUE>
  typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::predict(
      const NoiseModelFactor& motionFactor) {
    const auto keys = motionFactor.keys();

    // Create a Gaussian Factor Graph
    GaussianFactorGraph linearFactorGraph;

    // Add in previous posterior as prior
    linearFactorGraph.push_back(priorFactor_);

    // Linearize motion model using right-invariant error
    Values linearizationPoint;
    linearizationPoint.insert(keys[0], x_);
	// For prediction, use same state as initial guess
    linearizationPoint.insert(keys[1], x_);
    linearFactorGraph.push_back(motionFactor.linearize(linearizationPoint));

    // Solve the factor graph and update the current state estimate
    x_ = solve_(linearFactorGraph, linearizationPoint, keys[1], &priorFactor_);

    return x_;
  }

  /* ************************************************************************* */
  template<class VALUE>
  typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::update(
      const NoiseModelFactor& measurementFactor) {
    const auto keys = measurementFactor.keys();

    // Create a Gaussian Factor Graph
    GaussianFactorGraph linearFactorGraph;

    // Add in the prior on the first state
    linearFactorGraph.push_back(priorFactor_);

    // Linearize measurement factor using right-invariant error
    Values linearizationPoint;
    linearizationPoint.insert(keys[0], x_);
    linearFactorGraph.push_back(measurementFactor.linearize(linearizationPoint));

    // Solve the factor graph and update the current state estimate
    x_ = solve_(linearFactorGraph, linearizationPoint, keys[0], &priorFactor_);

    return x_;
  }

} // namespace gtsam
