#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/base/Vector.h>

using namespace std;
using namespace gtsam;
using symbol_shorthand::X;

int main() {
  GaussianFactorGraph::shared_ptr linearFactorGraph(new GaussianFactorGraph);
  Ordering::shared_ptr ordering(new Ordering);
  Values linearizationPoints;

  // Initial state (Pose2 at origin)
  Pose2 x_initial(0, 0, 0);
  SharedDiagonal P_initial = noiseModel::Isotropic::Sigma(3, 0.1);
  ordering->push_back(X(0));

  // Add prior factor
  linearizationPoints.insert(X(0), x_initial);
  linearFactorGraph->add(X(0), 
    P_initial->R(), 
    Vector::Zero(3),
    noiseModel::Unit::Create(3));

  linearizationPoints.insert(X(1), Pose2(1.0, 0.0, 0.0));  
  // Time step 1: Predict
  ordering->push_back(X(1));

  Pose2 motionModel(1.0, 0.0, 0.0);
  SharedDiagonal Q = noiseModel::Isotropic::Sigma(3, 0.1);
  
  // Standard BetweenFactor
  BetweenFactor<Pose2> factor(X(0), X(1), motionModel, Q);

  // Compute original error
  Pose2 measured = motionModel;
  Pose2 actual = linearizationPoints.at<Pose2>(X(0)).between(linearizationPoints.at<Pose2>(X(1)));
  Vector3 originalError = Pose2::Logmap(measured.between(actual));

  // Modify the error to remove state dependence
  Vector3 invariantError = linearizationPoints.at<Pose2>(X(0)).Adjoint(originalError);

  // Modify linearization point so that the standard factor behaves like an invariant factor
  linearizationPoints.update(X(1), Pose2::Expmap(invariantError) * linearizationPoints.at<Pose2>(X(0)));

  // Add factor after modification
  linearFactorGraph->push_back(factor.linearize(linearizationPoints));

  // Solve to get prediction
  GaussianBayesNet::shared_ptr bayesNet = linearFactorGraph->eliminateSequential(*ordering);
  VectorValues result = bayesNet->optimize();
  Pose2 x1_predict = linearizationPoints.at<Pose2>(X(1)).compose(Pose2(result[X(1)]));
  traits<Pose2>::Print(x1_predict, "X1 Predict");

  linearizationPoints.update(X(1), x1_predict);

  // Correct Noise Propagation Using Adjoint
  Matrix3 Ad_X1 = x1_predict.AdjointMap();
  Matrix3 P_pred = Ad_X1 * P_initial->covariance() * Ad_X1.transpose() + Q->covariance();
  SharedDiagonal P_updated = noiseModel::Diagonal::Variances(P_pred.diagonal());

  // Reset factor graph and ordering properly
  linearFactorGraph = GaussianFactorGraph::shared_ptr(new GaussianFactorGraph);
  ordering = Ordering::shared_ptr(new Ordering);  // Reset ordering to match new graph

  // Only push_back X(1) once in the ordering
  linearFactorGraph->add(X(1), 
	  P_updated->R(),
	  bayesNet->back()->d() - bayesNet->back()->R() * result[X(1)],
	  P_updated);
  ordering->push_back(X(1));  // Re-add after inserting factor

  Pose2 z1(1.0, 0.0, 0.0);
  SharedDiagonal R1 = noiseModel::Isotropic::Sigma(3, 0.25);
  PriorFactor<Pose2> factor4(X(1), z1, R1);
  linearFactorGraph->push_back(factor4.linearize(linearizationPoints));

  // Solve to get updated state
  GaussianBayesNet::shared_ptr updatedBayesNet = linearFactorGraph->eliminateSequential(*ordering);
  VectorValues updatedResult = updatedBayesNet->optimize();
  Pose2 x1_update = linearizationPoints.at<Pose2>(X(1)).compose(Pose2(updatedResult[X(1)]));
  traits<Pose2>::Print(x1_update, "X1 Update");

  linearizationPoints.update(X(1), x1_update);

  return 0;
}
