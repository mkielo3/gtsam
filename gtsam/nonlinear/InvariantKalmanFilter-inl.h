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
  const Ordering lastKeyAsOrdering{lastKey};
  const GaussianConditional::shared_ptr marginal =
      linearFactorGraph.marginalMultifrontalBayesNet(lastKeyAsOrdering)->front();

  VectorValues result = marginal->solve(VectorValues());
  const T& current = linearizationPoint.at<T>(lastKey);
  T x = current.compose(traits<T>::Expmap(result[lastKey]));

  assert(marginal->nrFrontals() == 1);
  assert(marginal->nrParents() == 0);
  *newPrior = JacobianFactor::shared_ptr(
      new JacobianFactor(marginal->keys().front(),
                         marginal->getA(marginal->begin()),
                         marginal->getb() - marginal->getA(marginal->begin()) * result[lastKey],
                         marginal->get_model()));

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
// Updated: Accepts a BetweenFactor<T> which provides measured() and noiseModel()
template<class VALUE>
typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::predict(
    const BetweenFactor<T>& motionFactor) {
  const auto keys = motionFactor.keys();

  // Create factor graph and add prior
  GaussianFactorGraph linearFactorGraph;
  linearFactorGraph.push_back(priorFactor_);

  // Setup linearization points
  Values linearizationPoint;
  linearizationPoint.insert(keys[0], x_);
  linearizationPoint.insert(keys[1], x_);

  // Compute invariant error for motion model
  const T& x0 = linearizationPoint.at<T>(keys[0]);
  const T& x1 = linearizationPoint.at<T>(keys[1]);
  T relative = x0.between(x1);
  T measured = motionFactor.measured();  // Valid for BetweenFactor<T>
  Vector originalError = T::Logmap(measured.between(relative));
  Vector invariantError = x0.AdjointMap() * originalError;

  // Update linearization point with invariant error
  linearizationPoint.update(keys[1], T::Expmap(invariantError) * x0);

  // Add motion factor and solve
  linearFactorGraph.push_back(motionFactor.linearize(linearizationPoint));

  GaussianBayesNet::shared_ptr bayesNet = linearFactorGraph.eliminateSequential(Ordering(keys));
  VectorValues result = bayesNet->optimize();

  // Compute predicted state
  T x_predict = linearizationPoint.at<T>(keys[1]).compose(traits<T>::Expmap(result[keys[1]]));

  // Transform covariance using Adjoint
  Matrix Ad = x_predict.AdjointMap();
  Matrix P = priorFactor_->get_model()->covariance();
  
  // Cast noise model to a Gaussian noise model to access covariance()
  auto gaussianNoise = std::dynamic_pointer_cast<noiseModel::Gaussian>(motionFactor.noiseModel());
  Matrix Q = gaussianNoise->covariance();
  
  Matrix P_pred = Ad * P * Ad.transpose() + Q;
  auto P_updated = noiseModel::Diagonal::Variances(P_pred.diagonal());

  // Create new prior factor for next step
  priorFactor_ = JacobianFactor::shared_ptr(
      new JacobianFactor(keys[1],
                         P_updated->R(),
                         bayesNet->back()->d() - bayesNet->back()->R() * result[keys[1]],
                         P_updated));

  x_ = x_predict;
  return x_;
}

/* ************************************************************************* */
// Updated: Accepts a PriorFactor<T> which provides prior() instead of measured()
template<class VALUE>
typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::update(
    const PriorFactor<T>& measurementFactor) {
  const auto keys = measurementFactor.keys();

  GaussianFactorGraph linearFactorGraph;
  linearFactorGraph.push_back(priorFactor_);

  Values linearizationPoint;
  linearizationPoint.insert(keys[0], x_);

  // Use measurementFactor.prior() instead of measured()
  Vector originalError = T::Logmap(measurementFactor.prior().between(linearizationPoint.at<T>(keys[0])));
  Vector invariantError = linearizationPoint.at<T>(keys[0]).AdjointMap() * originalError;
  linearizationPoint.update(keys[0], T::Expmap(invariantError) * linearizationPoint.at<T>(keys[0]));

  linearFactorGraph.push_back(measurementFactor.linearize(linearizationPoint));
  x_ = solve_(linearFactorGraph, linearizationPoint, keys[0], &priorFactor_);
  return x_;
}

} // namespace gtsam
