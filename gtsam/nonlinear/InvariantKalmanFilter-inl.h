#pragma once

#include <gtsam/nonlinear/InvariantKalmanFilter.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/base/Lie.h>
#include <memory>

namespace gtsam {

/* ************************************************************************* */
template<class VALUE>
typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::solve_(
    const GaussianFactorGraph& linearFactorGraph,
    const Values& linearizationPoint, Key lastKey,
    JacobianFactor::shared_ptr* newPrior) {
  // Compute the marginal on the last key
  const Ordering lastKeyAsOrdering{lastKey};
  const GaussianConditional::shared_ptr marginal =
      linearFactorGraph.marginalMultifrontalBayesNet(lastKeyAsOrdering)->front();

  // Extract the current estimate in the body frame
  VectorValues result = marginal->solve(VectorValues());
  const T& current = linearizationPoint.at<T>(lastKey);
  
  // For left-invariant: update using Expmap in body frame
  T x = current.compose(traits<T>::Expmap(result[lastKey]));

  // Create new prior factor for next iteration
  assert(marginal->nrFrontals() == 1);
  assert(marginal->nrParents() == 0);
  *newPrior = std::make_shared<JacobianFactor>(
      marginal->keys().front(),
      marginal->getA(marginal->begin()),
      marginal->getb() - marginal->getA(marginal->begin()) * result[lastKey],
      marginal->get_model());

  return x;
}

/* ************************************************************************* */
template<class VALUE>
InvariantKalmanFilter<VALUE>::InvariantKalmanFilter(
    Key key_initial, T x_initial, noiseModel::Gaussian::shared_ptr P_initial)
    : x_(x_initial) {
  int n = traits<T>::GetDimension(x_initial);
  priorFactor_ = typename JacobianFactor::shared_ptr(
      new JacobianFactor(key_initial, P_initial->R(), Vector::Zero(n),
                         noiseModel::Unit::Create(n)));
}

/* ************************************************************************* */
template<class VALUE>
typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::predict(
    const BetweenFactor<T>& motionFactor) {
  const auto keys = motionFactor.keys();
  
  // Create factor graph and add prior
  GaussianFactorGraph linearFactorGraph;
  linearFactorGraph.push_back(priorFactor_);

  // Setup initial linearization point
  Values linearizationPoint;
  linearizationPoint.insert(keys[0], x_);
  T predicted = x_.compose(motionFactor.measured());
  linearizationPoint.insert(keys[1], predicted);

  // Compute original error and transform it using adjoint
  T measured = motionFactor.measured();
  T actual = x_.between(predicted);
  Vector originalError = T::Logmap(measured.between(actual));
  Vector invariantError = predicted.AdjointMap() * originalError;  // Changed Adjoint to AdjointMap

  // Update linearization point with invariant error
//   linearizationPoint.update(keys[1], T::Expmap(invariantError) * predicted);
  linearizationPoint.update(keys[1], predicted.compose(T::Expmap(invariantError)));

  // Linearize with transformed linearization point
  GaussianFactor::shared_ptr gaussianFactor = motionFactor.linearize(linearizationPoint);
  linearFactorGraph.push_back(gaussianFactor);

  // Solve and update state
  x_ = solve_(linearFactorGraph, linearizationPoint, keys[1], &priorFactor_);

  return x_;
}

/* ************************************************************************* */

template<class VALUE>
typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::update(
    const PriorFactor<T>& measurementFactor) {
  // Create factor graph and add prior
  GaussianFactorGraph linearFactorGraph;
  linearFactorGraph.push_back(priorFactor_);

  // Get keys and initial linearization point
  const KeyVector keys = measurementFactor.keys();
  Values linearizationPoint;
  linearizationPoint.insert(keys[0], x_);

  // Compute original error and transform it using adjoint
  T measured = measurementFactor.prior();
  T actual = x_;
  Vector originalError = T::Logmap(measured.between(actual));
  Vector invariantError = x_.AdjointMap() * originalError;

  // Update linearization point with invariant error
//   linearizationPoint.update(keys[0], T::Expmap(invariantError) * x_);
  linearizationPoint.update(keys[0], x_.compose(T::Expmap(invariantError)));

  // Linearize with transformed linearization point
  GaussianFactor::shared_ptr gaussianFactor = 
      measurementFactor.linearize(linearizationPoint);
  linearFactorGraph.push_back(gaussianFactor);

  // Solve and update state
  x_ = solve_(linearFactorGraph, linearizationPoint, keys[0], &priorFactor_);

  return x_;
}

}