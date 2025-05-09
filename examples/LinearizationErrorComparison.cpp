/**
 * @file LinearizationErrorComparison.cpp
 * @brief Comparison of linearization error between GTSAM's CombinedImuFactor and GalileanImuFactor,
 *        reproducing the experiment from "Equivariant IMU Preintegration with Biases: a Galilean Group Approach"
 */

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/CombinedImuFactor.h>   // Standard GTSAM IMU factor
#include <gtsam/navigation/GalileanImuFactor.h> // Our new Galilean IMU factor
#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/numericalDerivative.h>

#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <string>

using namespace gtsam;
using namespace std;

using symbol_shorthand::X; // Pose3
using symbol_shorthand::V; // Velocity
using symbol_shorthand::B; // IMU Bias

// Constants for the simulation
constexpr double kGravity = 9.81;       // m/s^2
constexpr double kDt = 0.005;           // 200 Hz
constexpr size_t kNumSteps = 6000;      // 30 seconds
constexpr size_t kNumMonteCarloRuns = 100; // Number of Monte Carlo runs

// Noise parameters based on the paper
struct NoiseParameters {
    // Noise densities (continuous time)
    double gyroNoiseDensity = 7e-2;     // rad/s / sqrt(Hz)
    double accelNoiseDensity = 1.9e-1;  // m/s^2 / sqrt(Hz)
    double gyroBiasRwDensity = 1.5e-4;  // rad/s^2 / sqrt(Hz)
    double accelBiasRwDensity = 1.2e-2; // m/s^3 / sqrt(Hz)

    // Standard deviations for noise addition (scaled by lambda)
    double gyroNoise;   // rad/s
    double accelNoise;  // m/s^2
    double gyroBiasRW;  // rad/s over interval dt
    double accelBiasRW; // m/s^2 over interval dt

    // Lambda scaling factor
    double lambda_;
    double dt_;

    // Constructor to compute stddevs
    NoiseParameters(double lambda = 1.0, double dt = kDt)
        : lambda_(lambda), dt_(dt)
    {
        // Convert continuous-time densities to discrete-time stddevs
        gyroNoise = gyroNoiseDensity * lambda / sqrt(dt);
        accelNoise = accelNoiseDensity * lambda / sqrt(dt);
        gyroBiasRW = gyroBiasRwDensity * lambda * sqrt(dt);
        accelBiasRW = accelBiasRwDensity * lambda * sqrt(dt);
    }

    // Functions to get variances for GTSAM preintegration params
    Vector3 gyroVariance() const { return Vector3::Constant(pow(gyroNoiseDensity * lambda_, 2) / dt_); }
    Vector3 accelVariance() const { return Vector3::Constant(pow(accelNoiseDensity * lambda_, 2) / dt_); }
    Vector3 gyroBiasRwVariance() const { return Vector3::Constant(pow(gyroBiasRwDensity * lambda_, 2) * dt_); }
    Vector3 accelBiasRwVariance() const { return Vector3::Constant(pow(accelBiasRwDensity * lambda_, 2) * dt_); }
};

/**
 * @brief Generates a circular trajectory with a cosine wave on the z-axis
 */
std::vector<NavState> generateTrajectory(
    double radius, double height, double angularVelocity,
    double zFrequency, size_t numSteps, double dt) {

    std::vector<NavState> trajectory;
    trajectory.reserve(numSteps);

    for (size_t i = 0; i < numSteps; ++i) {
        double t = i * dt;

        // Position
        double x = radius * cos(angularVelocity * t);
        double y = radius * sin(angularVelocity * t);
        double z = height * cos(zFrequency * t);

        // Velocity (world frame)
        double vx = -radius * angularVelocity * sin(angularVelocity * t);
        double vy = radius * angularVelocity * cos(angularVelocity * t);
        double vz = -height * zFrequency * sin(zFrequency * t);

        // Rotation (aligning with movement direction)
        double yaw = angularVelocity * t;
        double pitch = 0.0;
        double roll = 0.0;

        Rot3 rotation = Rot3::RzRyRx(roll, pitch, yaw);
        Point3 position(x, y, z);
        Vector3 velocity_world(vx, vy, vz);

        trajectory.push_back(NavState(Pose3(rotation, position), velocity_world));
    }

    return trajectory;
}

/**
 * @brief Generates IMU measurements from a trajectory
 */
std::pair<std::vector<Vector3>, std::vector<Vector3>> generateImuMeasurements(
    const std::vector<NavState>& trajectory,
    const Vector3& gravity_world,
    const imuBias::ConstantBias& biases,
    double dt) {

    std::vector<Vector3> accelerations;
    std::vector<Vector3> angularVelocities;
    accelerations.reserve(trajectory.size() - 1);
    angularVelocities.reserve(trajectory.size() - 1);

    for (size_t i = 0; i < trajectory.size() - 1; ++i) {
        const NavState& state1 = trajectory[i];
        const NavState& state2 = trajectory[i + 1];

        // Extract states
        Pose3 pose1 = state1.pose();
        Pose3 pose2 = state2.pose();
        Vector3 vel1_world = state1.velocity();
        Vector3 vel2_world = state2.velocity();
        Rot3 rot1 = pose1.rotation();

        // Calculate true acceleration in world frame
        Vector3 accel_world = (vel2_world - vel1_world) / dt;

        // Measured acceleration in body frame = R_bw * (accel_world - gravity_world) + bias_accel
        Vector3 accel_body = rot1.unrotate(accel_world - gravity_world);

        // Calculate true angular velocity in body frame
        Rot3 dR = rot1.between(pose2.rotation());
        Vector3 angular_velocity_body = Rot3::Logmap(dR) / dt;

        // Add IMU biases
        Vector3 measured_accel = accel_body + biases.accelerometer();
        Vector3 measured_omega = angular_velocity_body + biases.gyroscope();

        accelerations.push_back(measured_accel);
        angularVelocities.push_back(measured_omega);
    }

    return {accelerations, angularVelocities};
}

/**
 * @brief Adds Gaussian noise to IMU measurements
 */
std::vector<Vector3> addNoise(
    const std::vector<Vector3>& measurements,
    double stddev,
    std::mt19937& gen) {

    if (stddev <= 0.0) {
        return measurements;
    }

    std::normal_distribution<double> dist(0.0, stddev);
    std::vector<Vector3> noisyMeasurements;
    noisyMeasurements.reserve(measurements.size());

    for (const auto& m : measurements) {
        Vector3 noise(dist(gen), dist(gen), dist(gen));
        noisyMeasurements.push_back(m + noise);
    }

    return noisyMeasurements;
}

/**
 * @brief Calculate the linearization error as defined in the paper
 * This compares the actual error at a perturbed state with the linearized prediction
 */
double calculateLinearizationError(
    const Pose3& pose_i_true, const Vector3& vel_i_true, const imuBias::ConstantBias& bias_i_true,
    const Pose3& pose_j_true, const Vector3& vel_j_true, const imuBias::ConstantBias& bias_j_true,
    double lambda, double preintTime,
    PreintegratedCombinedMeasurements& navPim,
    PreintegratedGalileanMeasurements& galPim) {

    // Factor keys
    Key pose_i = X(0), pose_j = X(1);
    Key vel_i = V(0), vel_j = V(1);
    Key bias_i = B(0), bias_j = B(1);

    // Create the factors
    CombinedImuFactor navFactor(pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, navPim);
    GalileanImuFactor galFactor(pose_i, vel_i, bias_i, pose_j, vel_j, bias_j, galPim);

    // Calculate errors at the linearization point (true state)
    Vector navError = navFactor.evaluateError(pose_i_true, vel_i_true, pose_j_true, vel_j_true,
                                              bias_i_true, bias_j_true);
    Vector galError = galFactor.evaluateError(pose_i_true, vel_i_true, bias_i_true,
                                              pose_j_true, vel_j_true, bias_j_true);

    // Calculate Jacobians at the linearization point
    Matrix navH1, navH2, navH3, navH4, navH5, navH6;
    Matrix galH1, galH2, galH3, galH4, galH5, galH6;

    navFactor.evaluateError(pose_i_true, vel_i_true, pose_j_true, vel_j_true, bias_i_true, bias_j_true,
                            &navH1, &navH2, &navH3, &navH4, &navH5, &navH6);

    galFactor.evaluateError(pose_i_true, vel_i_true, bias_i_true, pose_j_true, vel_j_true, bias_j_true,
                            &galH1, &galH2, &galH3, &galH4, &galH5, &galH6);

    // Define perturbations - scale with lambda to get bigger perturbations for higher noise
    double rot_pert_scale = 0.002 * lambda;
    double pos_pert_scale = 0.01 * lambda;
    double vel_pert_scale = 0.02 * lambda;
    double bias_acc_pert_scale = 0.001 * lambda;
    double bias_gyro_pert_scale = 0.0005 * lambda;

    Vector3 rotDelta(rot_pert_scale, -rot_pert_scale * 0.5, rot_pert_scale * 1.5);
    Vector3 posDelta(pos_pert_scale, -pos_pert_scale, pos_pert_scale * 0.5);
    Vector3 velDelta(vel_pert_scale, -vel_pert_scale * 0.75, vel_pert_scale * 0.5);
    Vector3 accelBiasDelta(bias_acc_pert_scale, -bias_acc_pert_scale * 2.0, bias_acc_pert_scale * 1.5);
    Vector3 gyroBiasDelta(bias_gyro_pert_scale, -bias_gyro_pert_scale * 1.2, bias_gyro_pert_scale * 0.8);

    // Create perturbed state
    Vector6 pose_i_tangent;
    pose_i_tangent << rotDelta, posDelta;
    Pose3 perturbedPose_i = pose_i_true.expmap(pose_i_tangent);
    Vector3 perturbedVel_i = vel_i_true + velDelta;
    imuBias::ConstantBias perturbedBias_i(bias_i_true.accelerometer() + accelBiasDelta,
                                         bias_i_true.gyroscope() + gyroBiasDelta);

    // Calculate true errors at the perturbed state
    Vector navPerturbedError = navFactor.evaluateError(perturbedPose_i, perturbedVel_i,
                                                       pose_j_true, vel_j_true,
                                                       perturbedBias_i, bias_j_true);

    Vector galPerturbedError = galFactor.evaluateError(perturbedPose_i, perturbedVel_i,
                                                      perturbedBias_i, pose_j_true,
                                                      vel_j_true, bias_j_true);

    // Calculate linearized prediction using first-order approximation
    Vector6 bias_i_perturbation;
    bias_i_perturbation << accelBiasDelta, gyroBiasDelta;

    Vector navLinPredictedError = navError + navH1 * pose_i_tangent +
                                  navH2 * velDelta +
                                  navH5 * bias_i_perturbation;

    Vector galLinPredictedError = galError;
    bool galH3_ok = galH3.allFinite();

    if (galH3_ok) {
        galLinPredictedError = galError + galH1 * pose_i_tangent +
                               galH2 * velDelta +
                               galH3 * bias_i_perturbation;
    } else {
        std::cerr << "Warning: GalileanImuFactor H3 Jacobian contains invalid values at preintTime="
                  << preintTime << "s" << std::endl;
        galLinPredictedError = galError + galH1 * pose_i_tangent + galH2 * velDelta;
    }

    // Compute linearization error magnitude
    double navLinErrorMag = (navPerturbedError - navLinPredictedError).norm();
    double galLinErrorMag = (galPerturbedError - galLinPredictedError).norm();

    // Return the difference between the two errors (for visualization comparison)
    // Positive number means Galilean has lower error
    return navLinErrorMag - galLinErrorMag;
}

/**
 * @brief Main experiment that calculates linearization error for different preintegration times
 */
std::pair<std::vector<double>, std::vector<double>> calculateLinearizationErrors(
    double lambda, double maxPreintTime, bool verbose = false) {

    if (verbose) {
        std::cout << "Starting linearization error calculation for lambda = " << lambda << std::endl;
    }

    // Set up noise parameters
    NoiseParameters noiseParams(lambda, kDt);

    // Set up random number generator with fixed seed for reproducibility
    std::mt19937 gen(42 + static_cast<unsigned int>(lambda * 1000));

    // Preintegration times to evaluate (in seconds)
    std::vector<double> preintTimes;
    for (double t = 0.5; t <= maxPreintTime; t += 0.5) {
        preintTimes.push_back(t);
    }

    // Vectors to store linearization errors
    std::vector<double> navStateErrors(preintTimes.size(), 0.0);
    std::vector<double> galileanErrors(preintTimes.size(), 0.0);

    for (size_t mc = 0; mc < kNumMonteCarloRuns; ++mc) {
        if (verbose && mc % 10 == 0) {
            std::cout << "Monte Carlo run " << mc << "/" << kNumMonteCarloRuns << std::endl;
        }

        // Generate trajectory
        double radius = 5.0;      // meters
        double height = 1.0;      // meters
        double angVel = 0.2;      // rad/s
        double zFreq = 0.5;       // Hz
        auto trajectory = generateTrajectory(radius, height, angVel, zFreq, kNumSteps, kDt);

        // Set gravity vector (Z-down convention)
        Vector3 gravity_world(0, 0, kGravity);

        // Set IMU biases
        Vector3 accelBias(0.05, -0.02, 0.01);  // m/s^2
        Vector3 gyroBias(0.001, -0.002, 0.0005); // rad/s
        imuBias::ConstantBias imuBiases(accelBias, gyroBias);

        // Generate true IMU measurements
        auto [accelMeas, gyroMeas] = generateImuMeasurements(trajectory, gravity_world, imuBiases, kDt);

        // Add noise
        std::vector<Vector3> noisyAccelMeas = addNoise(accelMeas, noiseParams.accelNoise, gen);
        std::vector<Vector3> noisyGyroMeas = addNoise(gyroMeas, noiseParams.gyroNoise, gen);

        // For each preintegration time
        for (size_t p = 0; p < preintTimes.size(); ++p) {
            double preintTime = preintTimes[p];
            size_t numPreintSteps = static_cast<size_t>(round(preintTime / kDt));

            if (numPreintSteps == 0 || numPreintSteps >= noisyAccelMeas.size()) {
                if (verbose) {
                    std::cout << "Skipping preintTime " << preintTime << "s: steps="
                              << numPreintSteps << ", max=" << noisyAccelMeas.size() << std::endl;
                }
                continue;
            }

            // Set up NavState preintegration parameters
            auto navParams = PreintegratedCombinedMeasurements::Params::MakeSharedD(gravity_world[2]);
            navParams->gyroscopeCovariance = noiseParams.gyroVariance().asDiagonal();
            navParams->accelerometerCovariance = noiseParams.accelVariance().asDiagonal();
            navParams->integrationCovariance = Matrix33::Identity() * 1e-8;
            navParams->biasAccCovariance = noiseParams.accelBiasRwVariance().asDiagonal();
            navParams->biasOmegaCovariance = noiseParams.gyroBiasRwVariance().asDiagonal();
            navParams->biasAccOmegaInt = Matrix66::Identity() * 1e-5;

            // Set up Galilean preintegration parameters
            auto galParams = std::make_shared<GalileanPreintegrationParams>(gravity_world);
            galParams->gyroscopeCovariance = noiseParams.gyroVariance().asDiagonal();
            galParams->accelerometerCovariance = noiseParams.accelVariance().asDiagonal();
            galParams->integrationCovariance = Matrix33::Identity() * 1e-8;
            galParams->biasAccCovariance = noiseParams.accelBiasRwVariance().asDiagonal();
            galParams->biasOmegaCovariance = noiseParams.gyroBiasRwVariance().asDiagonal();
            galParams->biasAccOmegaInt = Matrix66::Identity() * 1e-5;

            // Create preintegration objects
            PreintegratedCombinedMeasurements navPim(navParams, imuBiases);
            PreintegratedGalileanMeasurements galPim(galParams, imuBiases);

            // Integrate measurements
            for (size_t i = 0; i < numPreintSteps; ++i) {
                navPim.integrateMeasurement(noisyAccelMeas[i], noisyGyroMeas[i], kDt);
                galPim.integrateMeasurement(noisyAccelMeas[i], noisyGyroMeas[i], kDt);
            }

            // Get states at beginning and end
            size_t start_idx = 0;
            size_t end_idx = numPreintSteps;

            const NavState& start_state = trajectory[start_idx];
            const NavState& end_state = trajectory[end_idx];

            const Pose3& pose_i_true = start_state.pose();
            const Vector3& vel_i_true = start_state.velocity();
            const Pose3& pose_j_true = end_state.pose();
            const Vector3& vel_j_true = end_state.velocity();

            // Calculate linearization errors
            try {
                // Create the factors
                Key pose_i = X(0), pose_j = X(1);
                Key vel_i = V(0), vel_j = V(1);
                Key bias_i = B(0), bias_j = B(1);

                CombinedImuFactor navFactor(pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, navPim);
                GalileanImuFactor galFactor(pose_i, vel_i, bias_i, pose_j, vel_j, bias_j, galPim);

                // Calculate errors at true state
                Vector navError = navFactor.evaluateError(pose_i_true, vel_i_true, pose_j_true, vel_j_true,
                                                         imuBiases, imuBiases);
                Vector galError = galFactor.evaluateError(pose_i_true, vel_i_true, imuBiases,
                                                         pose_j_true, vel_j_true, imuBiases);

                // Calculate Jacobians
                Matrix navH1, navH2, navH3, navH4, navH5, navH6;
                Matrix galH1, galH2, galH3, galH4, galH5, galH6;

                navFactor.evaluateError(pose_i_true, vel_i_true, pose_j_true, vel_j_true,
                                       imuBiases, imuBiases,
                                       &navH1, &navH2, &navH3, &navH4, &navH5, &navH6);

                galFactor.evaluateError(pose_i_true, vel_i_true, imuBiases,
                                       pose_j_true, vel_j_true, imuBiases,
                                       &galH1, &galH2, &galH3, &galH4, &galH5, &galH6);

                // Define perturbations
                double rot_pert_scale = 0.002 * lambda;
                double pos_pert_scale = 0.01 * lambda;
                double vel_pert_scale = 0.02 * lambda;
                double bias_acc_pert_scale = 0.001 * lambda;
                double bias_gyro_pert_scale = 0.0005 * lambda;

                Vector3 rotDelta(rot_pert_scale, -rot_pert_scale * 0.5, rot_pert_scale * 1.5);
                Vector3 posDelta(pos_pert_scale, -pos_pert_scale, pos_pert_scale * 0.5);
                Vector3 velDelta(vel_pert_scale, -vel_pert_scale * 0.75, vel_pert_scale * 0.5);
                Vector3 accelBiasDelta(bias_acc_pert_scale, -bias_acc_pert_scale * 2.0, bias_acc_pert_scale * 1.5);
                Vector3 gyroBiasDelta(bias_gyro_pert_scale, -bias_gyro_pert_scale * 1.2, bias_gyro_pert_scale * 0.8);

                // Apply perturbation
                Vector6 pose_i_tangent;
                pose_i_tangent << rotDelta, posDelta;
                Pose3 perturbedPose_i = pose_i_true.expmap(pose_i_tangent);
                Vector3 perturbedVel_i = vel_i_true + velDelta;
                imuBias::ConstantBias perturbedBias_i(imuBiases.accelerometer() + accelBiasDelta,
                                                    imuBiases.gyroscope() + gyroBiasDelta);

                // Calculate true errors at the perturbed state
                Vector navPerturbedError = navFactor.evaluateError(perturbedPose_i, perturbedVel_i,
                                                                  pose_j_true, vel_j_true,
                                                                  perturbedBias_i, imuBiases);

                Vector galPerturbedError = galFactor.evaluateError(perturbedPose_i, perturbedVel_i,
                                                                  perturbedBias_i, pose_j_true,
                                                                  vel_j_true, imuBiases);

                // Calculate linearized prediction
                Vector6 bias_i_perturbation;
                bias_i_perturbation << accelBiasDelta, gyroBiasDelta;

                // Compute linearized predicted error for CombinedImuFactor
                Vector navLinPredictedError = navError + navH1 * pose_i_tangent +
                                             navH2 * velDelta +
                                             navH5 * bias_i_perturbation;

                // Compute linearized predicted error for GalileanImuFactor
                Vector galLinPredictedError = galError;
                if (galH3.allFinite()) {
                    galLinPredictedError = galError + galH1 * pose_i_tangent +
                                          galH2 * velDelta +
                                          galH3 * bias_i_perturbation;
                } else {
                    if (verbose) {
                        std::cerr << "Warning: GalileanImuFactor H3 Jacobian contains invalid values" << std::endl;
                    }
                    galLinPredictedError = galError + galH1 * pose_i_tangent + galH2 * velDelta;
                }

                // Compute linearization error magnitude (limit to 9-dimensional error for comparison)
                double navLinErrorMag = (navPerturbedError.head(9) - navLinPredictedError.head(9)).norm();
                double galLinErrorMag = (galPerturbedError.head(9) - galLinPredictedError.head(9)).norm();

                // Accumulate errors (will be averaged later)
                navStateErrors[p] += navLinErrorMag / kNumMonteCarloRuns;
                galileanErrors[p] += galLinErrorMag / kNumMonteCarloRuns;

            } catch (const std::exception& e) {
                if (verbose) {
                    std::cerr << "Exception during calculation: " << e.what() << std::endl;
                }
                continue;
            }
        }
    }

    if (verbose) {
        std::cout << "Linearization error calculation completed for lambda = " << lambda << std::endl;

        // Print a few results for sanity check
        std::cout << "Sample results (lambda = " << lambda << "):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), preintTimes.size()); ++i) {
            std::cout << "  Time " << std::fixed << std::setprecision(1) << preintTimes[i]
                      << "s: NavState = " << std::scientific << navStateErrors[i]
                      << ", Galilean = " << galileanErrors[i] << std::endl;
        }
    }

    return {navStateErrors, galileanErrors};
}

int main() {
    std::cout << "Calculating linearization error comparison between CombinedImuFactor and GalileanImuFactor..."
              << std::endl;

    // Different noise levels
    std::vector<double> lambdas = {0.1, 1.0, 10.0};
    double maxPreintTime = 30.0; // seconds

    bool verboseOutput = true;

    // Run experiments for each lambda
    for (double lambda : lambdas) {
        std::cout << "\nRunning with noise level scaling λ = " << lambda << std::endl;

        auto [navErrors, galErrors] = calculateLinearizationErrors(lambda, maxPreintTime, verboseOutput);

        // Write results to CSV file
        std::string filename = "linearization_error_lambda_" + std::to_string(lambda) + ".csv";
        std::ofstream outFile(filename);

        outFile << "Preint_Time,NavState_Error,Galilean_Error" << std::endl;

        std::vector<double> times;
        for (double t = 0.5; t <= maxPreintTime; t += 0.5) {
            times.push_back(t);
        }

        size_t numTimePoints = std::min(times.size(), navErrors.size());
        for (size_t i = 0; i < numTimePoints; ++i) {
            outFile << std::fixed << std::setprecision(1) << times[i] << ","
                    << std::scientific << std::setprecision(8) << navErrors[i] << ","
                    << std::scientific << std::setprecision(8) << galErrors[i] << std::endl;
        }

        outFile.close();
        std::cout << "Results written to " << filename << std::endl;
    }

    std::cout << "\nExperiment complete! Plot the results using the following Python script:" << std::endl;

    return 0;
}
