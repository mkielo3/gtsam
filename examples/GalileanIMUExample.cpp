/**
 * @file linearization_error_comparison.cpp
 * @brief Comparison of linearization error between GTSAM's NavState and Galilean IMU factor,
 * using MEDIAN aggregation over Monte Carlo runs.
 * @author Based on Delama et al., "Equivariant IMU Preintegration with Biases: a Galilean Group Approach"
 */

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>           // For PreintegratedImuMeasurements
#include <gtsam/navigation/CombinedImuFactor.h>   // For standard NavState factor
#include <gtsam/navigation/GalileanImuFactor.h> // For Galilean factor
#include <gtsam/navigation/GalileanPreintegrationParams.h>
#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/numericalDerivative.h> // For checking Jacobians

#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip> // For std::setprecision
#include <map>     // For storing results
#include <algorithm> // For std::min/max, std::sort, std::remove_if
#include <limits>  // For std::numeric_limits
#include <functional> // For std::bind
#include <numeric> // For std::accumulate (optional, not used here)
#include <stdexcept> // For std::exception


using namespace gtsam;
using namespace std;

using symbol_shorthand::X; // Pose3
using symbol_shorthand::V; // Velocity
using symbol_shorthand::B; // IMU Bias

// Constants for the simulation
constexpr double kGravity = 9.81;       // m/s^2
constexpr double kDt = 0.005;           // 200 Hz
constexpr size_t kNumSteps = 6000;      // 30 seconds
constexpr size_t kNumMonteCarloRuns = 100; // Number of Monte Carlo runs (set to 1 for deterministic debugging)
// DEBUG FLAG: Set to true to run deterministically (no added noise)
// Set back to false to run with noise
constexpr bool kDeterministicRun = false;

// Noise parameters based on the paper
struct NoiseParameters {
    // Noise densities (units involving 1/sqrt(Hz))
    double gyroNoiseDensity = 7e-2;     // rad/s / sqrt(Hz)
    double accelNoiseDensity = 1.9e-1;  // m/s^2 / sqrt(Hz)
    double gyroBiasRwDensity = 1.5e-4;  // rad/s / s / sqrt(Hz) = rad / s^(3/2)
    double accelBiasRwDensity = 1.2e-2; // m/s^2 / s / sqrt(Hz) = m / s^(5/2)

    // Standard deviations for noise addition (scaled by lambda, units without Hz)
    double gyroNoise;   // rad/s
    double accelNoise;  // m/s^2
    // Standard deviations for bias random walk (scaled by lambda, units without Hz)
    double gyroBiasRW;  // rad/s over interval dt
    double accelBiasRW; // m/s^2 over interval dt

    // Store lambda and dt as members
    double lambda_;     // Scaling factor
    double dt_;         // Time interval

    // Constructor converts densities to stddevs needed for noise addition and stores lambda/dt
    // Reordered initializer list to match declaration order
    NoiseParameters(double lambda_param = 1.0, double dt_param = kDt) // Renamed input params
        : // Initialize in declaration order
          gyroNoise(gyroNoiseDensity * lambda_param / sqrt(dt_param)), // Use params directly here
          accelNoise(accelNoiseDensity * lambda_param / sqrt(dt_param)),
          gyroBiasRW(gyroBiasRwDensity * lambda_param * sqrt(dt_param)),
          accelBiasRW(accelBiasRwDensity * lambda_param * sqrt(dt_param)),
          lambda_(lambda_param), // Store lambda and dt
          dt_(dt_param)
           {}

    // Function to get variances needed by GTSAM Params (variance = stddev^2)
    // Use member variables lambda_ and dt_
    // Measurement noise variance (sigma^2 = density^2 / dt)
    Vector3 gyroVariance() const { return Vector3::Constant(pow(gyroNoiseDensity * lambda_, 2) / dt_); }
    Vector3 accelVariance() const { return Vector3::Constant(pow(accelNoiseDensity * lambda_, 2) / dt_); }
    // Bias random walk variance (sigma^2 = density^2 * dt)
    Vector3 gyroBiasRwVariance() const { return Vector3::Constant(pow(gyroBiasRwDensity * lambda_, 2) * dt_); }
    Vector3 accelBiasRwVariance() const { return Vector3::Constant(pow(accelBiasRwDensity * lambda_, 2) * dt_); }
};

/**
 * @brief Calculate the statistical standard deviation of Vector3 measurements
 * @param measurements Vector of Vector3 measurements
 * @return Vector3 containing standard deviation for each component
 */
Vector3 calculateStdDev(const std::vector<Vector3>& measurements) {
    if (measurements.empty()) {
        return Vector3::Zero();
    }

    // Calculate mean
    Vector3 sum = Vector3::Zero();
    for (const auto& m : measurements) {
        sum += m;
    }
    Vector3 mean = sum / measurements.size();

    // Calculate variance
    Vector3 variance = Vector3::Zero();
    for (const auto& m : measurements) {
        Vector3 diff = m - mean;
        variance += diff.cwiseProduct(diff); // Element-wise square
    }
    // Use N-1 for sample variance
    if (measurements.size() > 1) {
        variance /= (measurements.size() - 1); // Use N-1 for sample variance
    } else {
         variance = Vector3::Zero(); // Variance is zero for single point
    }


    // Calculate standard deviation
    return variance.cwiseSqrt(); // Element-wise square root
}

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

        // Rotation (simplified: initially aligned with world, then yawing)
        double yaw = angularVelocity * t;
        double pitch = 0.0; // Keep pitch constant
        double roll = 0.0;  // Keep roll constant

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
    const Vector3& gravity_world, // Gravity in world frame
    const imuBias::ConstantBias& biases, // Bias in sensor/body frame
    double dt) {

    std::vector<Vector3> accelerations; // Measured acceleration in body frame
    std::vector<Vector3> angularVelocities; // Measured angular velocity in body frame
    accelerations.reserve(trajectory.size() - 1);
    angularVelocities.reserve(trajectory.size() - 1);

    for (size_t i = 0; i < trajectory.size() - 1; ++i) {
        const NavState& state1 = trajectory[i];
        const NavState& state2 = trajectory[i + 1];

        // Extract states
        Pose3 pose1 = state1.pose();
        Pose3 pose2 = state2.pose();
        Vector3 vel1_world = state1.velocity(); // Assuming velocity is in world frame in NavState
        Vector3 vel2_world = state2.velocity(); // Assuming velocity is in world frame in NavState
        Rot3 rot1 = pose1.rotation(); // R_wb(t)
        Rot3 rot2 = pose2.rotation(); // R_wb(t+dt)

        // Calculate true acceleration in world frame
        Vector3 accel_world = (vel2_world - vel1_world) / dt;

        // Measured acceleration in body frame = R_bw * (accel_world - gravity_world) + bias_accel
        // R_bw = R_wb^T = rot1.transpose()
        Vector3 accel_body = rot1.unrotate(accel_world - gravity_world);

        // Calculate true angular velocity in body frame
        // omega_body = Logmap(R_bw(t) * R_wb(t+dt)) / dt = Logmap(rot1.transpose() * rot2) / dt
        Rot3 dR = rot1.between(rot2); // rot1.transpose() * rot2
        Vector3 angular_velocity_body = Rot3::Logmap(dR) / dt;

        // Add IMU biases (already in body frame)
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
    double stddev, // Standard deviation for noise addition
    std::mt19937& gen) {

    if (stddev <= 0.0) { // No noise to add (check for <= 0)
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

// Helper function for numerical derivative check of Galilean H3
// Takes bias_i as input, returns the 9D error vector.
// Captures other needed variables by reference from the outer scope.
Vector galileanErrorWrtBias(const imuBias::ConstantBias& bias_i,
                            const GalileanImuFactor& factor,
                            const Pose3& pose_i, const Vector3& vel_i,
                            const Pose3& pose_j, const Vector3& vel_j,
                            const imuBias::ConstantBias& bias_j) {
    // Ensure the factor uses the provided bias_i for evaluation
    // Note: The standard evaluateError might not directly allow overriding only bias_i
    //       without affecting bias_j if they share the same key internally.
    //       This numerical check might need adjustment depending on how GalileanImuFactor
    //       is implemented or if a specific evaluateError overload exists.
    // Assuming the provided factor's evaluateError correctly uses the passed bias_i:
    return factor.evaluateError(pose_i, vel_i, bias_i, pose_j, vel_j, bias_j);
}


/**
 * @brief Calculate the linearization error for both IMU factor types using MEDIAN aggregation
 */
std::pair<std::vector<double>, std::vector<double>> calculateMedianLinearizationError(
    double lambda, double maxPreintTime, bool verbose = false) {

    if (verbose) {
        std::cout << "\n==================================================\n";
        std::cout << "STARTING MEDIAN CALCULATION FOR LAMBDA = " << lambda << std::endl;
        std::cout << "==================================================\n";
    }

    // Set up noise parameters
    NoiseParameters noiseParams(lambda, kDt);

    // Set up random number generator
    std::random_device rd;
    // Use a fixed seed for deterministic runs if needed, otherwise use random_device + lambda
    // Ensure the seed changes sufficiently with lambda for non-deterministic runs
    unsigned int seed = kDeterministicRun ? static_cast<unsigned int>(lambda * 1000 + 1) : rd() + static_cast<unsigned int>(lambda * 1000 + 1);
    std::mt19937 gen(seed);


    // Preintegration times to evaluate (in seconds)
    std::vector<double> preintTimes;
    for (double t = 0.5; t <= maxPreintTime; t += 0.5) {
        preintTimes.push_back(t);
    }

    // --- MEDIAN CHANGE 1: Store ALL errors from each run ---
    // Vectors of vectors to store linearization errors from all MC runs for each time step
    std::vector<std::vector<double>> allNavStateErrors(preintTimes.size());
    std::vector<std::vector<double>> allGalileanErrors(preintTimes.size());


    // Use kNumMonteCarloRuns = 1 for deterministic runs
    size_t numRuns = kDeterministicRun ? 1 : kNumMonteCarloRuns;

    // Reserve space for efficiency if running Monte Carlo
    if (!kDeterministicRun) {
        for(size_t p=0; p < preintTimes.size(); ++p) {
            allNavStateErrors[p].reserve(numRuns);
            allGalileanErrors[p].reserve(numRuns);
        }
    }

    // Run Monte Carlo simulations (or single deterministic run)
    for (size_t mc = 0; mc < numRuns; ++mc) {
        if (verbose && !kDeterministicRun && mc % (std::max(size_t(1), numRuns/10)) == 0) { // Print progress only for MC runs
            std::cout << "Monte Carlo run " << mc << "/" << numRuns << " for lambda = " << lambda << std::endl;
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
        Vector3 accelBias(0.05, -0.02, 0.01); // m/s^2
        Vector3 gyroBias(0.001, -0.002, 0.0005); // rad/s
        imuBias::ConstantBias imuBiases(accelBias, gyroBias);

        // Generate true IMU measurements
        auto [accelMeas, gyroMeas] = generateImuMeasurements(trajectory, gravity_world, imuBiases, kDt);

        // Add noise (or skip if deterministic)
        std::vector<Vector3> noisyAccelMeas = accelMeas;
        std::vector<Vector3> noisyGyroMeas = gyroMeas;
        if (!kDeterministicRun) {
             noisyAccelMeas = addNoise(accelMeas, noiseParams.accelNoise, gen);
             noisyGyroMeas = addNoise(gyroMeas, noiseParams.gyroNoise, gen);
        }


        // VERIFICATION STEP 1: Check noise std dev if noise was added
        if (verbose && mc == 0 && !kDeterministicRun) {
            Vector3 actualAccelStdDev = calculateStdDev(noisyAccelMeas);
            Vector3 actualGyroStdDev = calculateStdDev(noisyGyroMeas);

            std::cout << "NOISE VERIFICATION (lambda = " << lambda << "):\n";
            std::cout << "Expected accel noise stddev: " << noiseParams.accelNoise << std::endl;
            std::cout << "Actual accel noise stddev (xyz): " << actualAccelStdDev.transpose() << std::endl;
            std::cout << "Expected gyro noise stddev: " << noiseParams.gyroNoise << std::endl;
            std::cout << "Actual gyro noise stddev (xyz): " << actualGyroStdDev.transpose() << std::endl;
        }

        // For each preintegration time
        for (size_t p = 0; p < preintTimes.size(); ++p) {
            double preintTime = preintTimes[p];
            size_t numPreintSteps = static_cast<size_t>(round(preintTime / kDt));

            // Check if enough measurements are available for this preintegration interval
            if (numPreintSteps == 0 || numPreintSteps > noisyAccelMeas.size()) {
                 if (verbose && mc == 0) std::cout << "Skipping preintTime " << preintTime << "s: numPreintSteps=" << numPreintSteps << ", max=" << noisyAccelMeas.size() << std::endl;
                 // If skipping, add a placeholder NaN to maintain vector size alignment for median calculation
                  allNavStateErrors[p].push_back(std::numeric_limits<double>::quiet_NaN());
                  allGalileanErrors[p].push_back(std::numeric_limits<double>::quiet_NaN());
                 continue; // Skip actual calculation for this run/time
            }

            if (verbose && mc == 0 && (p == 0 || p == preintTimes.size()/2 || p == preintTimes.size()-1)) {
                std::cout << "\nProcessing preintegration time: " << preintTime << "s (" << numPreintSteps << " steps)" << std::endl;
            }

            Key pose_i = X(0), pose_j = X(1);
            Key vel_i = V(0), vel_j = V(1);
            Key bias_i = B(0), bias_j = B(1);

            // 1. Set up NavState IMU factor Params
            auto navParams = PreintegratedCombinedMeasurements::Params::MakeSharedD(gravity_world[2]);
            navParams->gyroscopeCovariance = noiseParams.gyroVariance().asDiagonal();
            navParams->accelerometerCovariance = noiseParams.accelVariance().asDiagonal();
            navParams->integrationCovariance = Matrix33::Identity() * 1e-8;
            navParams->biasAccCovariance = noiseParams.accelBiasRwVariance().asDiagonal();
            navParams->biasOmegaCovariance = noiseParams.gyroBiasRwVariance().asDiagonal();
            navParams->biasAccOmegaInt = Matrix66::Identity() * 1e-5;

            PreintegratedCombinedMeasurements navStatePim(navParams, imuBiases);

            // 2. Set up Galilean IMU factor Params
            auto galParams = std::make_shared<GalileanPreintegrationParams>(gravity_world);
            galParams->gyroscopeCovariance = noiseParams.gyroVariance().asDiagonal();
            galParams->accelerometerCovariance = noiseParams.accelVariance().asDiagonal();
            galParams->integrationCovariance = Matrix33::Identity() * 1e-8;
            galParams->biasAccCovariance = noiseParams.accelBiasRwVariance().asDiagonal();
            galParams->biasOmegaCovariance = noiseParams.gyroBiasRwVariance().asDiagonal();
            galParams->biasAccOmegaInt = Matrix66::Identity() * 1e-5;

            PreintegratedGalileanMeasurements galileanPim(galParams, imuBiases);

            // Preintegrate measurements
            bool integration_failed = false;
            for (size_t i = 0; i < numPreintSteps; ++i) {
                // Check for NaN/Inf in measurements before integrating
                if (!noisyAccelMeas[i].allFinite() || !noisyGyroMeas[i].allFinite()) {
                     if (verbose && mc == 0) std::cerr << "Warning: NaN/Inf detected in measurement at step " << i << " during MC run " << mc << std::endl;
                     integration_failed = true;
                     break; // Stop integrating for this run/time
                }
                 // Wrap integration in try-catch in case of internal PIM issues
                try {
                    navStatePim.integrateMeasurement(noisyAccelMeas[i], noisyGyroMeas[i], kDt);
                    galileanPim.integrateMeasurement(noisyAccelMeas[i], noisyGyroMeas[i], kDt);
                } catch (const std::exception& e) {
                    if (verbose) std::cerr << "Exception during integration at step " << i << " MC run " << mc << ": " << e.what() << std::endl;
                    integration_failed = true;
                    break; // Stop integrating for this run/time if PIM fails
                }
            }

            // If integration failed, store NaNs and skip the rest for this run/time
            if (integration_failed) {
                 allNavStateErrors[p].push_back(std::numeric_limits<double>::quiet_NaN());
                 allGalileanErrors[p].push_back(std::numeric_limits<double>::quiet_NaN());
                 continue;
            }

            // Create the factors
            CombinedImuFactor navFactor(pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, navStatePim);
            GalileanImuFactor galFactor(pose_i, vel_i, bias_i, pose_j, vel_j, bias_j, galileanPim);

            // Get states at beginning and end
            size_t start_idx = 0;
            size_t end_idx = numPreintSteps; // Integration succeeded, so end_idx is valid relative to numPreintSteps

            // Check if end_idx exceeds trajectory bounds (shouldn't happen if initial check passed, but good safeguard)
            if (end_idx >= trajectory.size()) {
                 if (verbose) std::cerr << "Error: end_idx=" << end_idx << " exceeds trajectory size=" << trajectory.size() << " after integration for run " << mc << std::endl;
                 allNavStateErrors[p].push_back(std::numeric_limits<double>::quiet_NaN());
                 allGalileanErrors[p].push_back(std::numeric_limits<double>::quiet_NaN());
                 continue;
            }

            const NavState& start_state = trajectory[start_idx];
            const NavState& end_state = trajectory[end_idx];
            const Pose3& pose_i_true = start_state.pose();
            const Vector3& vel_i_true = start_state.velocity();
            const Pose3& pose_j_true = end_state.pose();
            const Vector3& vel_j_true = end_state.velocity();
            const imuBias::ConstantBias& bias_true = imuBiases;

            // Wrap evaluateError and Jacobian calculations in try-catch
            Vector navError, galError;
            Matrix navH1, navH2, navH3, navH4, navH5, navH6;
            Matrix galH1, galH2, galH3, galH4, galH5, galH6;
            Vector navPerturbedError, galPerturbedError;
            Vector navLinPredictedError; // Declare here
            Vector galLinPredictedError; // Declare here
            double navLinErrorMag = std::numeric_limits<double>::quiet_NaN();
            double galLinErrorMag = std::numeric_limits<double>::quiet_NaN();

            try {
                // Calculate errors at the linearization point
                navError = navFactor.evaluateError(pose_i_true, vel_i_true, pose_j_true, vel_j_true, bias_true, bias_true);
                galError = galFactor.evaluateError(pose_i_true, vel_i_true, bias_true, pose_j_true, vel_j_true, bias_true);

                // Calculate Jacobians at the linearization point
                navFactor.evaluateError(pose_i_true, vel_i_true, pose_j_true, vel_j_true, bias_true, bias_true,
                                        &navH1, &navH2, &navH4, &navH5, &navH3, &navH6); // Pass pointers &

                galFactor.evaluateError(pose_i_true, vel_i_true, bias_true, pose_j_true, vel_j_true, bias_true,
                                        &galH1, &galH2, &galH3, &galH4, &galH5, &galH6); // Pass pointers &

                // Define and scale perturbations
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

                // Apply perturbation to state i
                Vector6 pose_i_tangent;
                pose_i_tangent << rotDelta, posDelta;
                Pose3 perturbedPose_i = pose_i_true.expmap(pose_i_tangent);
                Vector3 perturbedVel_i = vel_i_true + velDelta;
                imuBias::ConstantBias perturbedBias_i(bias_true.accelerometer() + accelBiasDelta,
                                                    bias_true.gyroscope() + gyroBiasDelta);

                // Calculate true errors at the perturbed state i
                navPerturbedError = navFactor.evaluateError(perturbedPose_i, perturbedVel_i, pose_j_true, vel_j_true, perturbedBias_i, bias_true);
                galPerturbedError = galFactor.evaluateError(perturbedPose_i, perturbedVel_i, perturbedBias_i, pose_j_true, vel_j_true, bias_true);

                // Calculate linearized prediction
                Vector6 bias_i_perturbation;
                bias_i_perturbation << accelBiasDelta, gyroBiasDelta;
                navLinPredictedError = navError + navH1 * pose_i_tangent + navH2 * velDelta + navH3 * bias_i_perturbation;

                bool galH3_ok = galH3.allFinite(); // Check H3 validity
                 if (!galH3_ok && verbose && mc==0) {std::cerr << "Warning: NaN/Inf in galH3 at " << preintTime << std::endl;}

                Vector galTerm1 = galH1 * pose_i_tangent;
                Vector galTerm2 = galH2 * velDelta;
                // --- FIX: Use if/else instead of ternary operator ---
                Vector galTerm3; // Declare outside
                if (galH3_ok) {
                    galTerm3 = galH3 * bias_i_perturbation;
                } else {
                    // Ensure galError size is valid before using Vector::Zero
                    if (galError.size() > 0) {
                       galTerm3 = Vector::Zero(galError.size());
                    } else {
                       // Handle case where galError might be invalid size (e.g., if evaluateError failed)
                       // Setting to a default zero vector or handling error appropriately
                       galTerm3 = Vector::Zero(9); // Assuming 9D error for Galilean
                       if (verbose) std::cerr << "Warning: galError size invalid when setting Zero for galTerm3." << std::endl;
                    }
                }
                // --- End Fix ---
                galLinPredictedError = galError + galTerm1 + galTerm2 + galTerm3;


                // Compute linearization error magnitude
                navLinErrorMag = (navPerturbedError - navLinPredictedError).norm();
                galLinErrorMag = (galPerturbedError - galLinPredictedError).norm();

                 // Check final results for NaN/Inf
                 if (!std::isfinite(navLinErrorMag)) {
                    if (verbose) std::cerr << "Warning: NavState linearization error is non-finite at " << preintTime << "s, MC run " << mc << std::endl;
                    navLinErrorMag = std::numeric_limits<double>::quiet_NaN(); // Ensure it's NaN for median calculation
                 }
                 if (!std::isfinite(galLinErrorMag) || !galH3_ok) { // Also treat as error if H3 was invalid
                     if (verbose && !galH3_ok && std::isfinite(galLinErrorMag)) {std::cerr << "Warning: H3 invalid but galLinErrorMag finite? "<< preintTime << "s, MC run " << mc << std::endl;}
                     else if (verbose) {std::cerr << "Warning: Galilean linearization error is non-finite/BadH3 at " << preintTime << "s, MC run " << mc << std::endl;}
                     galLinErrorMag = std::numeric_limits<double>::quiet_NaN(); // Ensure it's NaN
                 }

            } catch (const std::exception& e) {
                 if (verbose) {
                     std::cerr << "Exception during linearization error calculation at " << preintTime
                               << "s, MC run " << mc << ": " << e.what() << std::endl;
                 }
                 navLinErrorMag = std::numeric_limits<double>::quiet_NaN();
                 galLinErrorMag = std::numeric_limits<double>::quiet_NaN();
            }


            // --- MEDIAN CHANGE 2: Store individual run errors ---
            allNavStateErrors[p].push_back(navLinErrorMag);
            allGalileanErrors[p].push_back(galLinErrorMag);

        } // End loop over preintegration times
    } // End loop over Monte Carlo runs


    // --- MEDIAN CHANGE 3: Calculate median from stored errors ---
    std::vector<double> medianNavStateErrors(preintTimes.size());
    std::vector<double> medianGalileanErrors(preintTimes.size());

    for (size_t p = 0; p < preintTimes.size(); ++p) {
        // Get copies of the error vectors for this time step
        std::vector<double> currentNavErrors = allNavStateErrors[p];
        std::vector<double> currentGalErrors = allGalileanErrors[p];

        // --- Robust Median Calculation: Remove NaNs/Infs before proceeding ---
        currentNavErrors.erase(std::remove_if(currentNavErrors.begin(), currentNavErrors.end(),
                                                [](double d){ return !std::isfinite(d); }),
                               currentNavErrors.end());
        currentGalErrors.erase(std::remove_if(currentGalErrors.begin(), currentGalErrors.end(),
                                                [](double d){ return !std::isfinite(d); }),
                               currentGalErrors.end());

        size_t validNavRuns = currentNavErrors.size();
        size_t validGalRuns = currentGalErrors.size();
        // --- End NaN/Inf Removal ---


        if (validNavRuns == 0) { // Handle case where no valid runs were completed
             medianNavStateErrors[p] = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Sort the vector of valid errors
            std::sort(currentNavErrors.begin(), currentNavErrors.end());

            // Calculate median
            if (validNavRuns % 2 == 1) {
                // Odd number of runs: take the middle element
                medianNavStateErrors[p] = currentNavErrors[validNavRuns / 2];
            } else {
                // Even number of runs: take the average of the two middle elements
                medianNavStateErrors[p] = (currentNavErrors[validNavRuns / 2 - 1] + currentNavErrors[validNavRuns / 2]) / 2.0;
            }
        }

        // Repeat for Galilean errors
         if (validGalRuns == 0) {
             medianGalileanErrors[p] = std::numeric_limits<double>::quiet_NaN();
         } else {
            std::sort(currentGalErrors.begin(), currentGalErrors.end());
            if (validGalRuns % 2 == 1) {
                medianGalileanErrors[p] = currentGalErrors[validGalRuns / 2];
            } else {
                medianGalileanErrors[p] = (currentGalErrors[validGalRuns / 2 - 1] + currentGalErrors[validGalRuns / 2]) / 2.0;
            }
         }
    } // End loop calculating medians


    if (verbose) {
        std::cout << "\n==================================================\n";
        std::cout << "MEDIAN CALCULATION COMPLETED FOR LAMBDA = " << lambda << std::endl;
        std::cout << "==================================================\n";

        // Print results
        std::cout << "Median Linearization Errors (lambda = " << lambda << ", from " << numRuns << " total runs):\n"; // Indicate median
        std::cout << std::scientific << std::setprecision(5);
        size_t numToPrint = 3;
        std::vector<double> times_vec;
         for (double t = 0.5; t <= maxPreintTime; t += 0.5) { times_vec.push_back(t); }

        for (size_t i = 0; i < std::min({numToPrint, times_vec.size(), medianNavStateErrors.size()}); ++i) {
            std::cout << "  Time " << std::fixed << std::setprecision(2) << times_vec[i] << "s: NavState = "
                      << std::scientific << medianNavStateErrors[i] // Print median
                      << ", Galilean = " << medianGalileanErrors[i] << std::endl; // Print median
        }
        if (times_vec.size() > numToPrint * 2) {
            std::cout << "  ..." << std::endl;
        }
        if (times_vec.size() > numToPrint) {
             size_t startIdx = std::max(numToPrint, times_vec.size() - numToPrint);
             // Ensure startIdx is valid before looping
             if (startIdx < times_vec.size()) {
                 for (size_t i = startIdx; i < std::min({times_vec.size(), medianNavStateErrors.size(), medianGalileanErrors.size()}); ++i) { // Check all vector sizes
                      std::cout << "  Time " << std::fixed << std::setprecision(2) << times_vec[i] << "s: NavState = "
                               << std::scientific << medianNavStateErrors[i] // Print median
                               << ", Galilean = " << medianGalileanErrors[i] << std::endl; // Print median
                 }
             }
         }
        std::cout << std::defaultfloat << std::setprecision(6);
    }

    // Return the vectors containing the median errors
    return {medianNavStateErrors, medianGalileanErrors};
}

int main() {
    std::cout << "Calculating linearization error comparison (using MEDIAN)..." << std::endl; // Update message
    if (kDeterministicRun) {
        std::cout << "*** RUNNING DETERMINISTICALLY (NO RANDOM NOISE ADDED) ***" << std::endl;
    }

    // Different noise levels
    std::vector<double> lambdas = {0.1, 1.0, 10.0};
    double maxPreintTime = 30.0; // seconds

    bool verboseOutput = true; // Set to true for detailed console output

    std::map<double, std::pair<std::vector<double>, std::vector<double>>> allResults;

    for (double lambda : lambdas) {
        std::cout << "\nRunning with noise level scaling λ = " << lambda << " (Median Aggregation)" << std::endl; // Update message

        // Call the median function
        allResults[lambda] = calculateMedianLinearizationError(lambda, maxPreintTime, verboseOutput);

        // Write results to file - maybe change filename to indicate median
        std::ostringstream filename;
        filename << "linearization_error_MEDIAN_lambda_" << lambda << (kDeterministicRun ? "_deterministic" : "") << ".csv"; // Indicate MEDIAN
        std::ofstream outFile(filename.str());

        outFile << "Preint_Time,NavState_Error_Median,Galilean_Error_Median" << std::endl; // Update headers

        const auto& results = allResults[lambda];
        const auto& navErrors = results.first; // These are now medians
        const auto& galErrors = results.second; // These are now medians
        std::vector<double> times_vec;
         for (double t = 0.5; t <= maxPreintTime; t += 0.5) { times_vec.push_back(t); }

        // Ensure we don't write more rows than we have data for any column
        size_t numRowsToWrite = std::min({times_vec.size(), navErrors.size(), galErrors.size()});

        for (size_t i = 0; i < numRowsToWrite; ++i) {
             outFile << std::fixed << std::setprecision(2) << times_vec[i] << ","
                     << std::scientific << std::setprecision(8) << navErrors[i] << "," // Write median
                     << std::scientific << std::setprecision(8) << galErrors[i] << std::endl; // Write median
        }

        outFile.close();
        std::cout << "Median results written to " << filename.str() << std::endl; // Update message
    }

    std::cout << "\nCompleted! Use the MEDIAN CSV files to plot the results." << std::endl; // Update message

    return 0;
}
