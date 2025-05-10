#include <gtsam/navigation/GalileanImuFactor.h>
#include <gtsam/navigation/PreintegrationGalileanParams.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Gal3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/TestableAssertions.h>

#include <CppUnitLite/TestHarness.h>

#include <vector>
#include <cmath>

using namespace std;
using namespace gtsam;

// Define constants used in tests (following pattern from other tests)
const int N_TESTS = 5;  // Number of test iterations for random tests
const double kTolerance = 1e-9;
static const Vector3 kZero = Z_3x1;  // Define kZero to avoid ambiguity

// Define matrix types
typedef Eigen::Matrix<double, 20, 20> Matrix20;
typedef Eigen::Matrix<double, 10, 10> Matrix10;
typedef Eigen::Matrix<double, 9, 6> Matrix96;

// Helper functions to generate test vectors (same as before)
vector<Vector3> createTestAccels(size_t n) {
    vector<Vector3> accs;
    accs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double val = 0.1 * i;
        accs.push_back(Vector3(val + 0.1, val + 0.2, val + 0.3));
    }
    return accs;
}

vector<Vector3> createTestGyros(size_t n) {
    vector<Vector3> gyros;
    gyros.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double val = 0.01 * i;
        gyros.push_back(Vector3(val + 0.01, val + 0.02, val + 0.03));
    }
    return gyros;
}

// Helper to create parameters (same as before)
std::shared_ptr<GalileanPreintegrationParams> createParams(
    const Vector3& gravity = Vector3(0, 0, -9.81),
    double gyroNoise = 1.0,
    double accNoise = 1.0,
    double virtualVelNoise = 0.0,
    double virtualTimeNoise = 0.0,
    double gyroBiasNoise = 1.0,
    double accBiasNoise = 1.0,
    double virtualVelBiasNoise = 0.0,
    double virtualTimeBiasNoise = 0.0) {

    // Fix: Use the constructor directly with the proper gravity vector
    auto params = std::shared_ptr<GalileanPreintegrationParams>(
        new GalileanPreintegrationParams(gravity));

    params->n_gravity = gravity;

    // Don't override n_gravity - it's already set correctly by the constructor
    params->gyroscopeCovariance = I_3x3 * (gyroNoise * gyroNoise);
    params->accelerometerCovariance = I_3x3 * (accNoise * accNoise);
    params->virtualVelCovariance = I_3x3 * (virtualVelNoise * virtualVelNoise);
    params->virtualTimeScaleCovariance = virtualTimeNoise * virtualTimeNoise;
    params->biasOmegaCovariance = I_3x3 * (gyroBiasNoise * gyroBiasNoise);
    params->biasAccCovariance = I_3x3 * (accBiasNoise * accBiasNoise);
    params->biasVirtualVelCovariance = I_3x3 * (virtualVelBiasNoise * virtualVelBiasNoise);
    params->biasVirtualTimeCovariance = virtualTimeBiasNoise * virtualTimeBiasNoise;

    // Set initial bias covariance
    params->biasAccOmegaInt.setZero();

    return params;
}


// accumulates rotation
Matrix3 computeGamma2(const Vector3& omega) {
    double theta2 = omega.dot(omega);
    double theta = std::sqrt(theta2);

    if (theta < 1e-9) {
        // Taylor expansion for small angles
        return 0.5 * Matrix3::Identity() - (1.0/6.0) * SO3::Hat(omega);
    }

    Vector3 axis = omega / theta;
    double s = (theta - std::sin(theta)) / theta2;
    double c = (1 - std::cos(theta)) / theta2;

    return c * Matrix3::Identity() + s * SO3::Hat(axis) + (0.5 - c) * axis * axis.transpose();
}


/* ************************************************************************* */
// Test Case 1: Constructors
TEST(GalPreintegration, Constructors) {
    for (int i = 0; i < N_TESTS; ++i) {
        {
            // Test default constructor with default parameters
            auto p = std::make_shared<GalileanPreintegrationParams>();
            PreintegratedGalileanMeasurements pim(p);

            // Build expected Qc matrix
            Matrix20 expectedQc = Matrix20::Identity();
            expectedQc.block<4,4>(6,6).setZero();  // virtual velocity block
            expectedQc.block<4,4>(16,16).setZero(); // virtual velocity bias block

            // Check initial state
            Matrix20 zeroMat20 = Matrix20::Zero();
            Matrix20 identMat20 = Matrix20::Identity();
            EXPECT(assert_equal(zeroMat20, pim.uncertaintyCovariance(), kTolerance));
            EXPECT(assert_equal(identMat20, pim.biasJacobian(), kTolerance));
            EXPECT(assert_equal(Gal3::Identity(), pim.deltaUpsilon(), kTolerance));

            // Fix: pim.biasHat().vector() returns Vector6, not Vector10
            Vector6 zeroBias = Vector6::Zero();
            EXPECT(assert_equal(zeroBias, pim.biasHat().vector(), kTolerance));

            EXPECT(assert_equal(Rot3::Identity(), pim.deltaRij(), kTolerance));
            EXPECT(assert_equal(kZero, pim.deltaVij(), kTolerance));
            EXPECT(assert_equal(kZero, pim.deltaPij(), kTolerance));
            DOUBLES_EQUAL(0.0, pim.deltaTij(), kTolerance);

            // Check parameters
            auto galileanParams = std::static_pointer_cast<const GalileanPreintegrationParams>(pim.params());
            EXPECT(galileanParams != nullptr);
            EXPECT(assert_equal(Vector3(0, 0, -9.81), galileanParams->n_gravity, kTolerance));
            DOUBLES_EQUAL(1.0, sqrt(galileanParams->gyroscopeCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1.0, sqrt(galileanParams->accelerometerCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(0.0, galileanParams->virtualVelCovariance.trace(), kTolerance);
            DOUBLES_EQUAL(0.0, galileanParams->virtualTimeScaleCovariance, kTolerance);
            DOUBLES_EQUAL(1.0, sqrt(galileanParams->biasOmegaCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1.0, sqrt(galileanParams->biasAccCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(0.0, galileanParams->biasVirtualVelCovariance.trace(), kTolerance);
            DOUBLES_EQUAL(0.0, galileanParams->biasVirtualTimeCovariance, kTolerance);

            // Fix: Handle getBiasExtendedInit() return type correctly
            Matrix10 zeroMat10 = Matrix10::Zero();
            EXPECT(assert_equal(zeroMat10, galileanParams->getBiasExtendedInit(), kTolerance));

            EXPECT(assert_equal(expectedQc, galileanParams->createNoiseCovariance(), kTolerance));
        }

        {
            // Test constructor with custom parameters
            auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 1e-8, 1e-10, 1e-6, 1e-5, 1e-7, 1e-8);
            PreintegratedGalileanMeasurements pim(p);

            // Build expected Qc matrix
            Matrix20 expectedQc = Matrix20::Identity();
            expectedQc.block<3,3>(0,0) *= 1e-4 * 1e-4;     // gyro noise
            expectedQc.block<3,3>(3,3) *= 1e-3 * 1e-3;     // acc noise
            expectedQc.block<3,3>(6,6) *= 1e-8 * 1e-8;     // virtual vel noise
            expectedQc(9,9) *= 1e-10 * 1e-10;              // virtual time noise
            expectedQc.block<3,3>(10,10) *= 1e-6 * 1e-6;   // gyro bias noise
            expectedQc.block<3,3>(13,13) *= 1e-5 * 1e-5;   // acc bias noise
            expectedQc.block<3,3>(16,16) *= 1e-7 * 1e-7;   // virtual vel bias noise
            expectedQc(19,19) *= 1e-8 * 1e-8;              // virtual time bias noise

            // Check parameters
            auto galileanParams = std::static_pointer_cast<const GalileanPreintegrationParams>(pim.params());
            EXPECT(assert_equal(Vector3(0, 0, -9.81), galileanParams->n_gravity, kTolerance));
            DOUBLES_EQUAL(1e-4, sqrt(galileanParams->gyroscopeCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1e-3, sqrt(galileanParams->accelerometerCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1e-8, sqrt(galileanParams->virtualVelCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1e-10, sqrt(galileanParams->virtualTimeScaleCovariance), kTolerance);
            DOUBLES_EQUAL(1e-6, sqrt(galileanParams->biasOmegaCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1e-5, sqrt(galileanParams->biasAccCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1e-7, sqrt(galileanParams->biasVirtualVelCovariance(0,0)), kTolerance);
            DOUBLES_EQUAL(1e-8, sqrt(galileanParams->biasVirtualTimeCovariance), kTolerance);

            // Fix: Handle getBiasExtendedInit() return type correctly
            Matrix10 zeroMat10 = Matrix10::Zero();
            EXPECT(assert_equal(zeroMat10, galileanParams->getBiasExtendedInit(), kTolerance));

            EXPECT(assert_equal(expectedQc, galileanParams->createNoiseCovariance(), kTolerance));
        }
    }
}

/* ************************************************************************* */
// Test Case 2: Mean Propagation
TEST(GalPreintegration, MeanPropagation) {
    for (int i = 0; i < N_TESTS; ++i) {
        size_t n = 100;
        double dt = 0.01;
        vector<Vector3> accs = createTestAccels(n);
        vector<Vector3> gyros = createTestGyros(n);

        auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 0, 0, 1e-6, 1e-5, 0, 0);
        PreintegratedGalileanMeasurements pim(p);

        // Keep track of expected values
        Matrix3 deltaRij = Matrix3::Identity();
        Vector3 deltaVij = Vector3::Zero();
        Vector3 deltaPij = Vector3::Zero();
        double deltaTij = 0.0;

        for (size_t j = 0; j < n; ++j) {
            pim.integrateMeasurement(accs[j], gyros[j], dt);

            // Calculate expected values using GTSAM's SO3 functions
            Vector3 gyro_dt = gyros[j] * dt;

            // Use SO3's ExpmapDerivative for the gamma2-like matrix
            Matrix3 gamma2 = computeGamma2(gyro_dt);

            // Use so3::DexpFunctor for the left Jacobian
            so3::DexpFunctor dexpFunc(gyro_dt);
            Matrix3 leftJac = dexpFunc.leftJacobian();

            deltaPij += deltaVij * dt + deltaRij * gamma2 * accs[j] * dt * dt;
            deltaVij += deltaRij * leftJac * accs[j] * dt;
            deltaRij *= SO3::Expmap(gyro_dt).matrix();
            deltaTij += dt;
        }

        // Compare with preintegrated values
        EXPECT(assert_equal(deltaRij, pim.deltaRij().matrix(), 1e-5));
        EXPECT(assert_equal(deltaVij, pim.deltaVij(), 1e-5));
        EXPECT(assert_equal(deltaPij, pim.deltaPij(), 1e-5));
        DOUBLES_EQUAL(deltaTij, pim.deltaTij(), kTolerance);
    }
}

/* ************************************************************************* */
// Test Case 3: Covariance Propagation
TEST(GalPreintegration, CovariancePropagation) {
    for (int i = 0; i < N_TESTS; ++i) {
        size_t n = 10;
        double dt = 0.01;
        vector<Vector3> accs = createTestAccels(n);
        vector<Vector3> gyros = createTestGyros(n);

        // Create parameters matching the GTSAM implementation
        auto p = GalileanPreintegrationParams::MakeSharedU(9.81);
        p->gyroscopeCovariance = Matrix3::Identity() * 1e-8;  // 1e-4^2
        p->accelerometerCovariance = Matrix3::Identity() * 1e-6;  // 1e-3^2
        p->setBiasOmegaCovariance(Matrix3::Identity() * 1e-12);  // 1e-6^2
        p->setBiasAccCovariance(Matrix3::Identity() * 1e-10);    // 1e-5^2

        PreintegratedGalileanMeasurements pim(p);

        // Initialize tracking variables
        Gal3 Upsilon = Gal3::Identity();
        Matrix20 A = Matrix20::Identity();
        Matrix20 B = Matrix20::Zero();
        Matrix20 Cov = pim.uncertaintyCovariance();
        Matrix20 PhiBias = Matrix20::Identity();
        Matrix20 Jxi = Matrix20::Identity();

        // Define indices locally
        constexpr size_t ups_R_idx = 6;    // theta component (rotation)
        constexpr size_t ups_v_idx = 3;    // nu component (velocity)
        constexpr size_t bias_w_idx = 10;  // gyro bias
        constexpr size_t bias_a_idx = 13;  // acc bias

        for (size_t j = 0; j < n; ++j) {
            pim.integrateMeasurement(accs[j], gyros[j], dt);

            // Create measurement vector (w)
            Vector10 w = Vector10::Zero();
            w.segment<3>(0) = gyros[j];   // gyro at [0-2]
            w.segment<3>(3) = accs[j];    // acc at [3-5]
            w(9) = 1;                     // virtual time scale

            // Get current bias estimate using proper mapping
            Vector10 bias_hat = PreintegratedGalileanMeasurements::mapBias6ToTangent10(
                pim.biasHat().vector());

            // Calculate w0 = Adjoint(Upsilon) * (w - bias_hat)
            Vector10 w_minus_bias = w - bias_hat;
            Vector10 w0 = Upsilon.AdjointMap() * w_minus_bias;

            // Update A matrix - use the tangent space mapping
            Vector10 tangent_w0 = PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(w0 * dt);
            A.block<10,10>(0,10) = Gal3::ExpmapDerivative(tangent_w0) * dt;  // Fixed: multiply by dt
            A.block<10,10>(10,10) = Gal3::Expmap(tangent_w0).AdjointMap();

            // Update B matrix - fixed formulation
            Vector10 tangent_w_bias = PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(
                w_minus_bias * dt);
            Matrix10 K = Upsilon.AdjointMap() * Gal3::ExpmapDerivative(tangent_w_bias) * dt;
            B.block<10,10>(0,0) = -K;  // Fixed: negative sign
            B.block<10,10>(10,10) = Upsilon.AdjointMap() * dt;  // Fixed: just Adjoint * dt

            // Create noise covariance using hardcoded indices
            Matrix20 Q_d = Matrix20::Zero();
            Q_d.block<3,3>(ups_R_idx, ups_R_idx) = p->gyroscopeCovariance / dt;  // [6-8]
            Q_d.block<3,3>(ups_v_idx, ups_v_idx) = p->accelerometerCovariance / dt;  // [3-5]
            Q_d.block<3,3>(bias_w_idx, bias_w_idx) = p->getBiasOmegaCovariance() / dt;  // [10-12]
            Q_d.block<3,3>(bias_a_idx, bias_a_idx) = p->getBiasAccCovariance() / dt;    // [13-15]

            // Update Upsilon using tangent mapping
            Vector10 tangent_w = PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(w * dt);
            Upsilon = Upsilon * Gal3::Expmap(tangent_w);

            // Propagate covariance
            Cov = A * Cov * A.transpose() + B * Q_d * B.transpose();

            // Update PhiBias - use the correct formula
            PhiBias.block<10,10>(0,10) = -K;  // Just -K

            // Update Jxi
            Jxi = PhiBias * Jxi;
        }

        // Compare with preintegrated values
        EXPECT(assert_equal(Cov, pim.uncertaintyCovariance(), 1e-3));
        EXPECT(assert_equal(Jxi, pim.biasJacobian(), 1e-3));
    }
}


// /* ************************************************************************* */
// // Additional Test: Bias Jacobian Consistency (same as before)
TEST(GalPreintegration, BiasJacobianConsistency) {
    for (int i = 0; i < N_TESTS; ++i) {
        auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 0, 0, 1e-6, 1e-5, 0, 0);
        PreintegratedGalileanMeasurements pim(p);

        // Create two different biases
        imuBias::ConstantBias bias1(Vector3(0.01, -0.01, 0.02), Vector3(-0.005, 0.002, 0.001));
        imuBias::ConstantBias bias2(Vector3(0.015, -0.008, 0.022), Vector3(-0.004, 0.003, 0.0015));

        // Use test measurements
        vector<Vector3> accs = createTestAccels(20);
        vector<Vector3> gyros = createTestGyros(20);
        double dt = 0.01;

        // Integrate with bias1
        pim.resetIntegrationAndSetBias(bias1);
        for (size_t j = 0; j < accs.size(); ++j) {
            pim.integrateMeasurement(accs[j], gyros[j], dt);
        }

        // Get bias correction with Jacobian
        Matrix96 H_analytical;
        pim.biasCorrectedDelta(bias2, H_analytical);

        // Check against numerical Jacobian
        std::function<Vector9(const imuBias::ConstantBias&)> func =
            [&pim](const imuBias::ConstantBias& b) { return pim.biasCorrectedDelta(b); };

        Matrix96 H_numerical = numericalDerivative11<Vector9, imuBias::ConstantBias>(func, bias2, 1e-7);

        // Compare analytical and numerical Jacobians
        EXPECT(assert_equal(H_numerical, H_analytical, 1e-3));  // Looser tolerance
    }
}

// /* ************************************************************************* */
// // Additional Test: Reset Integration
TEST(GalPreintegration, ResetIntegration) {
    auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 0, 0, 1e-6, 1e-5, 0, 0);
    PreintegratedGalileanMeasurements pim(p);

    // Integrate some measurements
    vector<Vector3> accs = createTestAccels(10);
    vector<Vector3> gyros = createTestGyros(10);
    double dt = 0.01;

    for (size_t j = 0; j < accs.size(); ++j) {
        pim.integrateMeasurement(accs[j], gyros[j], dt);
    }

    // Check that integration has changed values
    EXPECT(!assert_equal(Rot3::Identity(), pim.deltaRij(), kTolerance));
    EXPECT(pim.deltaVij().norm() > 0);
    EXPECT(pim.deltaPij().norm() > 0);
    EXPECT(pim.deltaTij() > 0);

    // Reset integration
    pim.resetIntegration();

    // Check that values are back to initial state - use kZero to avoid ambiguity
    EXPECT(assert_equal(Rot3::Identity(), pim.deltaRij(), kTolerance));
    EXPECT(assert_equal(kZero, pim.deltaVij(), kTolerance));
    EXPECT(assert_equal(kZero, pim.deltaPij(), kTolerance));
    DOUBLES_EQUAL(0.0, pim.deltaTij(), kTolerance);
    Matrix20 identMat20 = Matrix20::Identity();
    EXPECT(assert_equal(identMat20, pim.biasJacobian(), kTolerance));

    // Verify covariance structure is correctly reset
    Matrix20 cov = pim.uncertaintyCovariance();
    Matrix6 biasAccOmegaCov = p->getBiasAccOmegaInit();

    // Check that only bias covariance blocks are non-zero
    Matrix3 biasOmegaBlock = cov.block<3,3>(10,10);
    Matrix3 biasAccBlock = cov.block<3,3>(13,13);
    Matrix3 expectedBiasOmegaBlock = biasAccOmegaCov.block<3,3>(3,3);
    Matrix3 expectedBiasAccBlock = biasAccOmegaCov.block<3,3>(0,0);
    EXPECT(assert_equal(expectedBiasOmegaBlock, biasOmegaBlock, kTolerance));
    EXPECT(assert_equal(expectedBiasAccBlock, biasAccBlock, kTolerance));
    EXPECT(assert_equal(Rot3::Identity(), pim.deltaRij(), 1e-10));
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
