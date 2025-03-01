/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    InvariantKalmanFilter-inl.h
 * @brief   Class to perform Left-Invariant Extended Kalman Filtering using nonlinear factor graphs
 * @author  Matthew Kielo
 * @author  Scott Baker
 */

 #pragma once

 #include <gtsam/nonlinear/InvariantKalmanFilter.h>
 #include <gtsam/nonlinear/NonlinearFactor.h>
 #include <gtsam/linear/GaussianBayesNet.h>
 #include <gtsam/linear/GaussianFactorGraph.h>
 #include <gtsam/slam/BetweenFactor.h>
 
 namespace gtsam {
 
   /* ************************************************************************* */
   template<class VALUE>
   typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::solve_(
	   const GaussianFactorGraph& linearFactorGraph,
	   const Values& linearizationPoint, Key lastKey,
	   JacobianFactor::shared_ptr* newPrior)
   {
	 // Compute marginal density using factor graph (this is computing Kalman gain implicitly)
	 const Ordering lastKeyAsOrdering{lastKey};
	 const GaussianConditional::shared_ptr marginal =
	   linearFactorGraph.marginalMultifrontalBayesNet(lastKeyAsOrdering)->front();
 
	 // Solve for the optimal correction in the Lie algebra
	 VectorValues result = marginal->solve(VectorValues());
	 const T& current = linearizationPoint.at<T>(lastKey);
	 
	 // Apply the correction using left-invariant update
	 T x = traits<T>::Between(current, traits<T>::Expmap(result[lastKey]));
 
	 // Create a new prior factor for the next iteration
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
   template <class VALUE>
   InvariantKalmanFilter<VALUE>::InvariantKalmanFilter(
	   Key key_initial, T x_initial, noiseModel::Gaussian::shared_ptr P_initial)
	   : x_(x_initial)
   {
	 // Create a Jacobian Prior Factor
	 int n = traits<T>::GetDimension(x_initial);
	 priorFactor_ = std::make_shared<JacobianFactor>(
		 key_initial, 
		 P_initial->R(),
		 Vector::Zero(n),
		 noiseModel::Unit::Create(n));
   }
   
   /* ************************************************************************* */
   template<class VALUE>
   typename InvariantKalmanFilter<VALUE>::T InvariantKalmanFilter<VALUE>::predict(
	   const NoiseModelFactor& motionFactor) {
	 const auto keys = motionFactor.keys();
   
	 // Create a Gaussian Factor Graph
	 GaussianFactorGraph linearFactorGraph;
	 linearFactorGraph.push_back(priorFactor_);
   
	 // Linearize motion model and add it to the Kalman Filter graph
	 Values linearizationPoint;
	 linearizationPoint.insert(keys[0], x_);
	 linearizationPoint.insert(keys[1], x_);  // Simplified - same as EKF
	 linearFactorGraph.push_back(motionFactor.linearize(linearizationPoint));
   
	 // Solve and update state
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

	// Linearize measurement factor and add it to the Kalman Filter graph
	Values linearizationPoint;
	linearizationPoint.insert(keys[0], x_);
	linearFactorGraph.push_back(measurementFactor.linearize(linearizationPoint));

	// Solve the factor graph and update the current state estimate
	// and the prior factor for the next iteration
	x_ = solve_(linearFactorGraph, linearizationPoint, keys[0], &priorFactor_);
	return x_;
	} 
 } // namespace gtsam