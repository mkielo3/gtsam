/**
 * @file easyPose2InvariantKalmanFilter.cpp
 * Simple Invariant Kalman filter on a moving 2D pose using factor graphs
 */

 #include <gtsam/nonlinear/InvariantKalmanFilter.h>
 #include <gtsam/inference/Symbol.h>
 #include <gtsam/nonlinear/PriorFactor.h>
 #include <gtsam/slam/BetweenFactor.h>
 #include <gtsam/geometry/Pose2.h>
 
 using namespace std;
 using namespace gtsam;
 
 int main() {
   // Create the Kalman Filter initialization point
   // Pose2(x, y, theta)
   Pose2 x_initial(0.0, 0.0, 0.0);
   // Noise model for x, y, theta
   SharedDiagonal P_initial = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.1));
 
   // Create Key for initial pose
   Symbol x0('x',0);
 
   // Create an InvariantKalmanFilter object
   InvariantKalmanFilter<Pose2> ikf(x0, x_initial, P_initial);
 
   // Process noise for motion
   SharedDiagonal Q = noiseModel::Diagonal::Sigmas(Vector3(0.1, 0.1, 0.1), true);
 
   // Predict step: Moving forward 1m with no rotation
   Symbol x1('x',1);
   // Create motion: forward 1m with no rotation
   Pose2 difference(1.0, 0.0, 0.0);
   BetweenFactor<Pose2> factor1(x0, x1, difference, Q);
 
   // Predict the new value
   Pose2 x1_predict = ikf.predict(factor1);
   traits<Pose2>::Print(x1_predict, "X1 Predict");
 
   // Measurement noise
   SharedDiagonal R = noiseModel::Diagonal::Sigmas(Vector3(0.25, 0.25, 0.25), true);
 
   // Update step with measurement
   Pose2 z1(1.0, 0.0, 0.0);  // Measured pose
   PriorFactor<Pose2> factor2(x1, z1, R);
   Pose2 x1_update = ikf.update(factor2);
   traits<Pose2>::Print(x1_update, "X1 Update");
 
   // Second prediction
   Symbol x2('x',2);
   difference = Pose2(1.0, 0.0, 0.0);  // Move forward 1m again
   BetweenFactor<Pose2> factor3(x1, x2, difference, Q);
   Pose2 x2_predict = ikf.predict(factor3);
   traits<Pose2>::Print(x2_predict, "X2 Predict");
   
   // Second update
   Pose2 z2(2.0, 0.0, 0.0);
   PriorFactor<Pose2> factor4(x2, z2, R);
   Pose2 x2_update = ikf.update(factor4);
   traits<Pose2>::Print(x2_update, "X2 Update");
 
   // Third prediction
   Symbol x3('x',3);
   difference = Pose2(1.0, 0.0, 0.0);
   BetweenFactor<Pose2> factor5(x2, x3, difference, Q);
   Pose2 x3_predict = ikf.predict(factor5);
   traits<Pose2>::Print(x3_predict, "X3 Predict");
 
   // Third update
   Pose2 z3(3.0, 0.0, 0.0);
   PriorFactor<Pose2> factor6(x3, z3, R);
   Pose2 x3_update = ikf.update(factor6);
   traits<Pose2>::Print(x3_update, "X3 Update");
 
   return 0;
 }