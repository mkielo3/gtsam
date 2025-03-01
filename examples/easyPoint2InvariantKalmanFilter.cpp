#include <gtsam/nonlinear/InvariantKalmanFilter.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Pose2.h>

using namespace std;
using namespace gtsam;

int main() {
  // Create the filter initialization point (Pose2 is 3-dimensional: x, y, theta)
  Pose2 x_initial(0.0, 0.0, 0.0);
  // Initial uncertainty in body frame: 3D vector
  SharedDiagonal P_initial = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.1));

  Symbol x0('x',0);
  InvariantKalmanFilter<Pose2> ikf(x0, x_initial, P_initial);

  // Process noise in body frame: 3D vector noise model
  SharedDiagonal Q = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.1));

  // Predict step: Moving forward 1m with no rotation
  Symbol x1('x',1);
  // Here, difference is defined as a Pose2 (not Point2)
  Pose2 difference(1.0, 0.0, 0.0);
  // Use BetweenFactor<Pose2> (not BetweenFactor<Point2>)
  BetweenFactor<Pose2> factor1(x0, x1, difference, Q);
  Pose2 x1_predict = ikf.predict(factor1);
  traits<Pose2>::Print(x1_predict, "X1 Predict");

  // Measurement noise: again, a 3D vector noise model
  SharedDiagonal R = noiseModel::Diagonal::Sigmas(Vector3(0.25, 0.25, 0.25), true);

  // Update step with measurement: measurement should also be a Pose2.
  Pose2 z1(1.0, 0.0, 0.0);
  // Use PriorFactor<Pose2> instead of any other factor type.
  PriorFactor<Pose2> factor2(x1, z1, R);
  Pose2 x1_update = ikf.update(factor2);
  traits<Pose2>::Print(x1_update, "X1 Update");

  return 0;
}
