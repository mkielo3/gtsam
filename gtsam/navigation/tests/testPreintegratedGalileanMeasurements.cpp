/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    testPreintegratedGalileanMeasurements.cpp
 * @brief   Unit test for the PreintegratedGalileanMeasurements class
 * @author  (Your Name)
 */

#include <gtsam/navigation/GalileanImuFactor.h>
#include <gtsam/navigation/PreintegrationGalileanParams.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Gal3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/TestableAssertions.h> // For assert_equal

#include <CppUnitLite/TestHarness.h> // For TEST, EXPECT, DOUBLES_EQUAL
#include <Eigen/Eigenvalues> // For checking PSD covariance

#include <vector>
#include <cmath> // For std::abs

using namespace std;
using namespace gtsam;

// Define types and constants matching the implementation file
typedef imuBias::ConstantBias Bias;
typedef PreintegratedGalileanMeasurements PIM;
typedef GalileanPreintegrationParams Params;

// Define indices for accessing the 20D state vector/covariance/Jacobian
// Tangent space orderings:
// Upsilon (Gal3 tangent): [rho(p, 0-2), nu(v, 3-5), theta(R, 6-8), t(9)]
// Bias (gal(3) tangent): [b_omega(0-2), b_acc(3-5), b_nu(6-8), b_rho(9)]
// Combined 20D state tangent: [Upsilon_tangent | Bias_tangent]
namespace {
    // Dimensions
    const size_t ups_dim = 10;
    const size_t bias_dim = 10;
    // const size_t total_dim = ups_dim + bias_dim; // Should be 20 // Unused after print removal

    // Indices within Upsilon tangent vector (0-9)
    const size_t ups_p_idx = 0; // Start index for position component (rho)
    const size_t ups_v_idx = 3; // Start index for velocity component (nu)
    const size_t ups_R_idx = 6; // Start index for rotation component (theta)
    const size_t ups_t_idx = 9; // Index for time component (t)

    // Indices within Bias tangent vector (0-9)
    const size_t bias_w_comp_idx = 0; // Start index for gyro bias component (b_omega)
    const size_t bias_a_comp_idx = 3; // Start index for accel bias component (b_acc)
    const size_t bias_nu_comp_idx = 6; // Start index for virtual velocity bias component (b_nu)
    const size_t bias_rho_comp_idx = 9; // Index for virtual time bias component (b_rho)

    // Indices within the full 20D state tangent vector [Upsilon | Bias]
    const size_t bias_w_idx = ups_dim + bias_w_comp_idx; // 10 + 0 = 10
    const size_t bias_a_idx = ups_dim + bias_a_comp_idx; // 10 + 3 = 13
    const size_t bias_nu_idx = ups_dim + bias_nu_comp_idx; // 10 + 6 = 16
    const size_t bias_rho_idx = ups_dim + bias_rho_comp_idx; // 10 + 9 = 19

    // Constants
    const double kDt = 0.1;
    const Vector3 kZero(0.0, 0.0, 0.0);
    const Bias kZeroBias;
    const double kGravity = 9.81; // Example gravity magnitude
    const double kTol = 1e-9; // Tolerance for strict equality checks
    // Increased tolerance for dynamics involving approximations/coupling
    const double kApproxTol = 1e-4;
}

// Helper function to create parameters with specific noise settings
std::shared_ptr<Params> createTestParams(
    bool zero_measurement_noise = false,
    bool zero_integration_noise = true, // Typically zero unless testing integration noise specifically
    bool zero_bias_acc_noise = false,
    bool zero_bias_gyro_noise = false,
    bool zero_init_bias_cov = false)
{
    auto p = Params::MakeSharedU(kGravity); // Use Z-up gravity

    if (zero_measurement_noise) {
        p->accelerometerCovariance.setZero();
        p->gyroscopeCovariance.setZero();
    } else {
        p->accelerometerCovariance = I_3x3 * 1e-4; // Non-zero example
        p->gyroscopeCovariance = I_3x3 * 1e-6;     // Non-zero example
    }

    if (zero_integration_noise) {
        p->integrationCovariance.setZero(); // Base class member
    } else {
        p->integrationCovariance = I_3x3 * 1e-8; // Non-zero example
    }

    if (zero_bias_acc_noise) {
        p->biasAccCovariance.setZero();
    } else {
        p->biasAccCovariance = I_3x3 * 1e-5; // Non-zero example
    }

    if (zero_bias_gyro_noise) {
        p->biasOmegaCovariance.setZero();
    } else {
        p->biasOmegaCovariance = I_3x3 * 1e-7; // Non-zero example
    }

    if (zero_init_bias_cov) {
        p->biasAccOmegaInt.setZero();
    } else {
        p->biasAccOmegaInt = I_6x6 * 1e-3; // Non-zero example
    }

    return p;
}

/* ************************************************************************* */
// Test Case 1: Zero Input, Zero Bias
TEST(PreintegratedGalileanMeasurements, ZeroInputZeroBias) {
    // Use parameters with all noises set to zero for this test
    auto params = createTestParams(true, true, true, true, true);
    PIM pim(params, kZeroBias); // Initialize with zero bias

    // Integrate zero measurements multiple times
    int num_steps = 10;
    for (int i = 0; i < num_steps; ++i) {
        pim.integrateMeasurement(kZero, kZero, kDt);
    }

    // Check: deltaUpsilon should be Identity except for time
    Gal3 expectedUpsilon = Gal3(Rot3(), Point3::Zero(), Vector3::Zero(), num_steps * kDt);
    EXPECT(assert_equal(expectedUpsilon, pim.deltaUpsilon(), kTol));
    DOUBLES_EQUAL(num_steps * kDt, pim.deltaTij(), kTol);

    // Check: Covariance should remain zero
    Matrix20 zeroMat = Matrix20::Zero();
    EXPECT(assert_equal(zeroMat, pim.uncertaintyCovariance(), kTol));
}

/* ************************************************************************* */
// Test Case 2: Constant Bias, Zero Input
TEST(PreintegratedGalileanMeasurements, ConstantBiasZeroInput) {
    // Use parameters with bias random walk noise, zero measurement noise
    auto params = createTestParams(true, true, false, false, false); // Enable bias noise
    params->biasAccCovariance = I_3x3 * 1e-6; // Example bias accel noise
    params->biasOmegaCovariance = I_3x3 * 1e-8; // Example bias gyro noise

    Vector3 b_a(0.01, -0.02, 0.03);
    Vector3 b_w(0.005, 0.01, -0.002);
    Bias bias1(b_a, b_w);
    PIM pim(params, bias1); // Initialize with non-zero bias

    // Integrate zero measurements multiple times
    int num_steps = 10;
    double total_time = 0.0;
    for (int i = 0; i < num_steps; ++i) {
        pim.integrateMeasurement(kZero, kZero, kDt);
        total_time += kDt;
    }

    // Check deltaUpsilon against expected values based on first-order integration
    Rot3 expectedRot = Rot3::Expmap(-b_w * total_time); // First order rotation integration
    Vector3 expectedVel = -b_a * total_time;           // First order velocity integration
    Vector3 expectedPos = -0.5 * b_a * total_time * total_time; // First order position integration

    Gal3 expectedUpsilon(expectedRot, Point3(expectedPos), expectedVel, total_time);

    const double kGroupDynamicsTol = 5e-3; // Tolerance for group dynamics approximation
    EXPECT(assert_equal(expectedRot, pim.deltaRij(), kApproxTol));
    EXPECT(assert_equal(expectedPos, pim.deltaPij(), kApproxTol));
    EXPECT(assert_equal(expectedVel, pim.deltaVij(), kGroupDynamicsTol));
    DOUBLES_EQUAL(total_time, pim.deltaTij(), kTol);
    EXPECT(assert_equal(expectedUpsilon, pim.deltaUpsilon(), kGroupDynamicsTol));

    // Check: Covariance should grow according to bias random walk
    Matrix20 cov = pim.uncertaintyCovariance();
    Matrix cov_transpose = cov.transpose();
    EXPECT(assert_equal(cov, cov_transpose, kTol)); // Symmetry
    Eigen::SelfAdjointEigenSolver<Matrix20> es(cov);
    Vector eigenvalues = es.eigenvalues();
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        EXPECT(eigenvalues(i) >= -kTol); // PSD
    }

    // Check specific blocks related to bias random walk
    Matrix3 bias_w_block = cov.block<3,3>(bias_w_idx, bias_w_idx);
    EXPECT(bias_w_block.trace() > 1e-12); // Expect gyro bias variance to grow

    Matrix3 bias_a_block = cov.block<3,3>(bias_a_idx, bias_a_idx);
    EXPECT(bias_a_block.trace() > 1e-12); // Expect accel bias variance to grow

    // Rotation variance (indices 6-8) should NOT grow significantly from bias RW alone
    Matrix3 rot_block = cov.block<3,3>(ups_R_idx, ups_R_idx);
    EXPECT(rot_block.trace() < 1e-9); // Expect rotation variance to remain small
}


/* ************************************************************************* */
// Test Case 3: Constant Measurement, Zero Bias
TEST(PreintegratedGalileanMeasurements, ConstantMeasurementZeroBias) {
    // Use parameters with measurement noise, zero bias random walk noise
    auto params = createTestParams(false, true, true, true, true); // Enable measurement noise
    params->accelerometerCovariance = I_3x3 * 1e-4;
    params->gyroscopeCovariance = I_3x3 * 1e-6; // Ensure non-zero gyro noise

    PIM pim(params, kZeroBias); // Initialize with zero bias

    // Constant measurements (rotating around Z, accelerating Z+)
    Vector3 constOmega(0.0, 0.0, 0.1); // rad/s
    Vector3 constAcc(0.0, 0.0, kGravity); // Accel = gravity (in Z-up frame)

    // Integrate constant measurements multiple times
    int num_steps = 10;
    double total_time = 0.0;
    for (int i = 0; i < num_steps; ++i) {
        pim.integrateMeasurement(constAcc, constOmega, kDt);
        total_time += kDt;
    }

    // --- Check preintegrated values ---
    Rot3 expectedRot = Rot3::Rz(constOmega.z() * total_time);
    Vector3 expectedVel = constAcc * total_time;
    Vector3 expectedPos = 0.5 * constAcc * total_time * total_time;

    EXPECT(assert_equal(expectedRot, pim.deltaRij(), kApproxTol));
    EXPECT(assert_equal(expectedPos, pim.deltaPij(), kApproxTol));
    EXPECT(assert_equal(expectedVel, pim.deltaVij(), kApproxTol));
    DOUBLES_EQUAL(total_time, pim.deltaTij(), kTol);

    // --- Check Covariance ---
    Matrix20 cov = pim.uncertaintyCovariance();
    Matrix cov_transpose = cov.transpose();
    EXPECT(assert_equal(cov, cov_transpose, kTol)); // Symmetry
    Eigen::SelfAdjointEigenSolver<Matrix20> es(cov);
    Vector eigenvalues = es.eigenvalues();
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        EXPECT(eigenvalues(i) >= -kTol); // PSD
    }

    // Expect non-zero diagonal entries in the Upsilon blocks (0-9) due to measurement noise
    Matrix ups_block = cov.block<ups_dim, ups_dim>(0, 0);
    EXPECT(ups_block.trace() > 1e-12);

    // *** ADDED CHECK: Verify rotation variance specifically ***
    Matrix3 rot_block = cov.block<3,3>(ups_R_idx, ups_R_idx);
    EXPECT(rot_block.trace() > 1e-12); // Expect rotation variance to grow due to gyro noise

    // Bias blocks (10-19) should remain zero as bias random walk noise is zero
    Matrix bias_block = cov.block<bias_dim, bias_dim>(ups_dim, ups_dim);
    DOUBLES_EQUAL(0.0, bias_block.norm(), kApproxTol);
}

/* ************************************************************************* */
// Test Case 4: Bias Correction Accuracy (First-Order)
TEST(PreintegratedGalileanMeasurements, BiasCorrectionAccuracy) {
    // This test checks if the first-order bias correction works relative to the
    // computed (potentially numerical) bias Jacobian.

    auto params = createTestParams(false, true, false, false, false); // Include some noise

    std::vector<Vector3> omegas = {Vector3(0.1, 0.02, -0.03), Vector3(0.11, 0.01, -0.02)};
    std::vector<Vector3> accs = {Vector3(0.1, 0.5, kGravity+0.2), Vector3(0.05, 0.45, kGravity+0.1)};
    std::vector<double> dts = {kDt, kDt};

    Bias bias1(Vector3(0.01, -0.01, 0.02), Vector3(-0.005, 0.002, 0.001));
    Bias bias2(Vector3(0.012, -0.009, 0.021), Vector3(-0.004, 0.003, 0.0015));

    // Integrate with bias1
    PIM pim1(params, bias1);
    for (size_t i = 0; i < dts.size(); ++i) {
        pim1.integrateMeasurement(accs[i], omegas[i], dts[i]);
    }

    // Integrate with bias2
    PIM pim2(params, bias2);
    for (size_t i = 0; i < dts.size(); ++i) {
        pim2.integrateMeasurement(accs[i], omegas[i], dts[i]);
    }

    // Calculate the 9D NavState tangent space correction using pim1 and applying bias2
    Vector9 correction_tangent = pim1.biasCorrectedDelta(bias2);

    // Get the nominal 9D NavState tangent space delta for pim1 [Log(R), p, v]
    Vector9 delta_pim1_tangent;
    delta_pim1_tangent << Rot3::Logmap(pim1.deltaRij()), pim1.deltaPij(), pim1.deltaVij();

    // Apply correction to pim1's tangent vector
    Vector9 corrected_delta_pim1_tangent = delta_pim1_tangent + correction_tangent;

    // Get the nominal 9D NavState tangent space delta for pim2 [Log(R), p, v]
    Vector9 delta_pim2_tangent;
    delta_pim2_tangent << Rot3::Logmap(pim2.deltaRij()), pim2.deltaPij(), pim2.deltaVij();

    // Check: Corrected pim1 tangent should approximately equal pim2 tangent
    // Vector9 error = delta_pim2_tangent - corrected_delta_pim1_tangent; // Variable 'error' is not used
    EXPECT(assert_equal(delta_pim2_tangent, corrected_delta_pim1_tangent, 5e-3)); // Tolerance for 1st order approx

    // Check Jacobian of biasCorrectedDelta numerically
    Matrix96 H_actual;
    pim1.biasCorrectedDelta(bias2, H_actual);

    std::function<Vector9(const Bias&)> fun =
        [&](const Bias& b) { return pim1.biasCorrectedDelta(b); };
    Matrix96 H_expected = numericalDerivative11<Vector9, Bias>(fun, bias2, 1e-7);

    // Check difference between analytical and numerical Jacobians
    // Matrix96 jacobian_diff = H_expected - H_actual; // Variable 'jacobian_diff' is not used
    EXPECT(assert_equal(H_expected, H_actual, kApproxTol)); // Tolerance for numerical derivative
}

/* ************************************************************************* */
// Test Case 5: Covariance Properties (Integrated into other tests)
// The PSD/Symmetry checks are now performed directly inside Test Cases 2 and 3.

/* ************************************************************************* */
// Test Case: Gyroscope Noise to Rotation Variance Coupling
TEST(PreintegratedGalileanMeasurements, GyroNoiseRotationCoupling) {
    // Use parameters with only gyroscope noise enabled (to isolate the issue)
    auto params = createTestParams(true, true, true, true, true); // Start with all noise off
    params->gyroscopeCovariance = I_3x3 * 1e-2; // Significant gyro noise for easier debugging

    PIM pim(params, kZeroBias);

    // Use small but non-zero angular velocity to ensure coupling happens
    Vector3 smallOmega(0.001, 0.002, 0.003); // Small but non-zero

    // Integrate several steps to accumulate covariance
    int num_steps = 5;

    // Diagnose coupling at each step
    for (int i = 0; i < num_steps; ++i) {
        // Before integration
        Matrix3 pre_rot_cov = pim.uncertaintyCovariance().block<3,3>(ups_R_idx, ups_R_idx);
        double pre_rot_trace = pre_rot_cov.trace();

        // Integrate with small constant omega
        pim.integrateMeasurement(kZero, smallOmega, kDt);

        // After integration
        Matrix3 post_rot_cov = pim.uncertaintyCovariance().block<3,3>(ups_R_idx, ups_R_idx);
        double post_rot_trace = post_rot_cov.trace();

        // At each step, the rotation covariance should grow
        EXPECT(post_rot_trace > pre_rot_trace);
    }

    // Final check
    Matrix3 rot_block = pim.uncertaintyCovariance().block<3,3>(ups_R_idx, ups_R_idx);

    // The covariance should have grown due to gyro noise
    EXPECT(rot_block.trace() > 1e-6);
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
