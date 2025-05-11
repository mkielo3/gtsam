/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 */

/**
 * @file CombinedGalileanImuFactor.cpp
 * @brief Implementation file for combined Galilean IMU preintegration and factor.
 */

#include <gtsam/navigation/GalileanImuFactor.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/linear/GaussianFactor.h>

#include <stdexcept>
#include <functional>
#include <limits>
#include <iomanip>
#include <ostream>

namespace gtsam {

// Define indices for accessing components in GTSAM convention [rho, nu, theta, t]
namespace {
  // Dimensions
  const size_t UPS_DIM = 10;
  const size_t BIAS_DIM = 10;
  const size_t TOTAL_DIM = 20;

  // Upsilon tangent indices: [rho, nu, theta, t]
  const size_t RHO_IDX = 0;   // position (0-2)
  const size_t NU_IDX = 3;    // velocity (3-5)
  const size_t THETA_IDX = 6; // rotation (6-8)
  const size_t T_IDX = 9;     // time duration (9)

  // Bias tangent indices: [b_omega, b_acc, b_nu, b_rho]
  const size_t B_OMEGA_IDX = 0; // gyro bias (0-2)
  const size_t B_ACC_IDX = 3;   // acc bias (3-5)
  const size_t B_NU_IDX = 6;    // virtual velocity bias (6-8)
  const size_t B_RHO_IDX = 9;   // virtual time bias (9)

  // Full 20D state indices [Upsilon | Bias]
  const size_t BIAS_START_IDX = UPS_DIM;

  // NavState indices [theta, p, v]
  const size_t NAV_R_IDX = 0;
  const size_t NAV_P_IDX = 3;
  const size_t NAV_V_IDX = 6;
}


// --- PreintegratedGalileanMeasurements Implementation ---

PreintegratedGalileanMeasurements::PreintegratedGalileanMeasurements(
    const std::shared_ptr<Params>& p, const Bias& biasHat)
    : Base(p, biasHat),
      deltaUpsilon_(Gal3::Identity()),
      preintMeasCov_(Matrix20::Zero()),
      preintBiasJacobian_(Matrix20::Identity()) {
    resetIntegration();
}

void PreintegratedGalileanMeasurements::resetIntegration() {
  deltaUpsilon_ = Gal3::Identity();
  deltaTij_ = 0.0;
  preintMeasCov_.setZero();
  preintBiasJacobian_.setIdentity();
}

std::shared_ptr<const PreintegratedGalileanMeasurements::Params>
PreintegratedGalileanMeasurements::galileanParams() const {
    if (!p_) {
        throw std::runtime_error("PreintegratedGalileanMeasurements: Parameters pointer is null.");
    }
    auto params_ptr = std::dynamic_pointer_cast<const Params>(p_);
    if (!params_ptr) {
        throw std::runtime_error("PreintegratedGalileanMeasurements: Incorrect parameter type provided.");
    }
    return params_ptr;
}

void PreintegratedGalileanMeasurements::print(const std::string& s) const {
  Base::print(s);
  std::cout << s << " Preintegrated Galilean Measurements:" << std::endl;
  deltaUpsilon_.print("  Delta Upsilon: ");
}

bool PreintegratedGalileanMeasurements::equals(
    const PreintegratedGalileanMeasurements& other, double tol) const {
  if (!Base::matchesParamsWith(other) ||
      !biasHat_.equals(other.biasHat_, tol) ||
      std::fabs(deltaTij_ - other.deltaTij_) > tol) {
    return false;
  }
  return deltaUpsilon_.equals(other.deltaUpsilon_, tol)
      && equal_with_abs_tol(preintMeasCov_, other.preintMeasCov_, tol)
      && equal_with_abs_tol(preintBiasJacobian_, other.preintBiasJacobian_, tol);
}

// Convert 6D bias [gyro; acc] to 10D bias [gyro; acc; virtual_vel; virtual_time]
Vector10 PreintegratedGalileanMeasurements::mapBias6ToTangent10(const Vector6& bias6D) {
    Vector10 result = Vector10::Zero();
    result.segment<3>(B_OMEGA_IDX) = bias6D.head<3>(); // gyro bias
    result.segment<3>(B_ACC_IDX) = bias6D.tail<3>();   // acc bias
    return result;
}

// Convert measurement in Lie++ ordering [omega; acc; virtual_vel; virtual_time]
// to GTSAM Upsilon tangent [rho; nu; theta; t]
Vector10 PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(const Vector10& measurement10D) {
    Vector10 tangent = Vector10::Zero();
    // Convert from Lie++ ordering [omega; acc; virtual_vel; virtual_time]
    // to GTSAM ordering [rho; nu; theta; t]
    tangent.segment<3>(THETA_IDX) = measurement10D.segment<3>(0); // omega -> theta (rotation)
    tangent.segment<3>(NU_IDX) = measurement10D.segment<3>(3);    // acc -> nu (velocity)
    tangent.segment<3>(RHO_IDX) = measurement10D.segment<3>(6);   // virtual_vel -> rho (position)
    tangent(T_IDX) = measurement10D(9);                          // virtual_time -> t (time)
    return tangent;
}

// void PreintegratedGalileanMeasurements::integrateMeasurement(
//     const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {

//     if (dt <= 0) {
//         std::cerr << "WARNING: dt <= 0 in integrateMeasurement. Skipping integration." << std::endl;
//         return;
//     }

//     auto params = galileanParams();

//     // Create input with virtual components set to zero
//     Vector10 w;
//     w << measuredOmega, measuredAcc, Vector3::Zero(), 1.0;

//     // Remove bias from measurement
//     Vector10 bias_10D = mapBias6ToTangent10(biasHat_.vector());
//     Vector10 w_unbiased = w - bias_10D;

//     // Convert to Upsilon tangent space (rearrange components for GTSAM)
//     Vector10 tangent_arg = mapMeasurement10ToTangent10(w_unbiased);

//     // Compute matrices for integration
//     Matrix10 Lj = Gal3::LeftJacobian(tangent_arg * dt);
//     Matrix10 Adj_Upsilon = deltaUpsilon_.AdjointMap();
//     Matrix10 K = Adj_Upsilon * Lj * dt;

//     // Update mean using matrix exponential
//     deltaUpsilon_ = deltaUpsilon_ * Gal3::Expmap(tangent_arg * dt);
//     deltaTij_ += dt;

//     // Propagate covariance
//     Matrix20 A = Matrix20::Identity();
//     A.block<10, 10>(0, UPS_DIM) = Lj * dt;
//     A.block<10, 10>(UPS_DIM, UPS_DIM) = Gal3::Expmap(tangent_arg * dt).AdjointMap();

//     Matrix20 B = Matrix20::Zero();
//     B.block<10, 10>(0, 0) = -K;
//     B.block<10, 10>(UPS_DIM, UPS_DIM) = deltaUpsilon_.AdjointMap() * dt;

//     // Create noise covariance Q_d (20x20) in MEASUREMENT SPACE
//     // This is the key fix - Q_d should be in the measurement space, not tangent space
//     // Create noise covariance Q_d in TANGENT SPACE (not measurement space)
//     Matrix20 Q_d = Matrix20::Zero();

//     // First 10 dimensions in GTSAM tangent space [rho, nu, theta, t]
//     Q_d.block<3,3>(0, 0) = params->virtualVelCovariance / dt;        // rho
//     Q_d.block<3,3>(3, 3) = params->accelerometerCovariance / dt;     // nu
//     Q_d.block<3,3>(6, 6) = params->gyroscopeCovariance / dt;         // theta
//     Q_d(9, 9) = params->virtualTimeScaleCovariance / dt;             // t

//     // Last 10 dimensions are bias random walk in GTSAM bias ordering [b_omega, b_acc, b_nu, b_rho]
//     Q_d.block<3,3>(10, 10) = params->getBiasOmegaCovariance() / dt;       // gyro bias
//     Q_d.block<3,3>(13, 13) = params->getBiasAccCovariance() / dt;         // acc bias
//     Q_d.block<3,3>(16, 16) = params->biasVirtualVelCovariance / dt;       // virtual velocity bias
//     Q_d(19, 19) = params->biasVirtualTimeCovariance / dt;                 // virtual time bias

//     // std::cout << "Q_d diagonal elements:" << std::endl;
//     // std::cout << "  rho:   " << Q_d.block<3,3>(0,0).diagonal().transpose() << std::endl;
//     // std::cout << "  nu:    " << Q_d.block<3,3>(3,3).diagonal().transpose() << std::endl;
//     // std::cout << "  theta: " << Q_d.block<3,3>(6,6).diagonal().transpose() << std::endl;
//     // std::cout << "  t:     " << Q_d(9,9) << std::endl;
//     // std::cout << "  bias:  " << Q_d.diagonal().tail<10>().transpose() << std::endl;

//     // Update covariance
//     // std::cout << "Cov_old (preintMeasCov_ BEFORE update):\n" << preintMeasCov_ << std::endl;
//     preintMeasCov_ = A * preintMeasCov_ * A.transpose() + B * Q_d * B.transpose();
//     // std::cout << "Cov_new (preintMeasCov_ AFTER update):\n" << preintMeasCov_ << std::endl;

//     // std::cout << "A matrix:\n" << A << std::endl;
//     // std::cout << "B matrix:\n" << B << std::endl;
//     // std::cout << "A*Cov*A^T:\n" << (A * preintMeasCov_ * A.transpose()) << std::endl;
//     // std::cout << "B*Q*B^T:\n" << (B * Q_d * B.transpose()) << std::endl;

//     // Update bias Jacobian
//     Matrix20 Phi_b = Matrix20::Identity();
//     Phi_b.block<10, 10>(0, UPS_DIM) = -K;
//     preintBiasJacobian_ = Phi_b * preintBiasJacobian_;

//     // std::cout << "K matrix:\n" << K << std::endl;
//     // std::cout << "A matrix:\n" << A << std::endl;
//     // std::cout << "B matrix:\n" << B << std::endl;
//     // std::cout << "Q_d matrix:\n" << Q_d << std::endl;

// }

void PreintegratedGalileanMeasurements::integrateMeasurement(
    const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {

    if (dt <= 0) {
        std::cerr << "WARNING: dt <= 0 in integrateMeasurement. Skipping integration." << std::endl;
        return;
    }

    auto params = galileanParams();

    // Debug: Print input measurements
    std::cout << "\n===== integrateMeasurement DEBUG =====" << std::endl;
    std::cout << "measuredAcc: " << measuredAcc.transpose() << std::endl;
    std::cout << "measuredOmega: " << measuredOmega.transpose() << std::endl;
    std::cout << "dt: " << dt << std::endl;

    // Create input with virtual components set to zero
    Vector10 w;
    w << measuredOmega, measuredAcc, Vector3::Zero(), 1.0;

    // Remove bias from measurement
    Vector10 bias_10D = mapBias6ToTangent10(biasHat_.vector());
    Vector10 w_unbiased = w - bias_10D;

    // Convert to Upsilon tangent space (rearrange components for GTSAM)
    Vector10 tangent_arg = mapMeasurement10ToTangent10(w_unbiased);

    std::cout << "w (measurement space): " << w.transpose() << std::endl;
    std::cout << "bias_10D: " << bias_10D.transpose() << std::endl;
    std::cout << "w_unbiased: " << w_unbiased.transpose() << std::endl;
    std::cout << "tangent_arg (GTSAM ordering): " << tangent_arg.transpose() << std::endl;

    // Compute matrices for integration
    Matrix10 Lj = Gal3::LeftJacobian(tangent_arg * dt);
    Matrix10 Adj_Upsilon = deltaUpsilon_.AdjointMap();
    Matrix10 K = Adj_Upsilon * Lj * dt;

    // Debug: Print key matrices
    std::cout << "\nK matrix diagonal elements:" << std::endl;
    std::cout << "  " << K.diagonal().transpose() << std::endl;

    // Update mean using matrix exponential
    deltaUpsilon_ = deltaUpsilon_ * Gal3::Expmap(tangent_arg * dt);
    deltaTij_ += dt;

    // Propagate covariance
    Matrix20 A = Matrix20::Identity();
    A.block<10, 10>(0, UPS_DIM) = Lj * dt;
    A.block<10, 10>(UPS_DIM, UPS_DIM) = Gal3::Expmap(tangent_arg * dt).AdjointMap();

    Matrix20 B = Matrix20::Zero();
    B.block<10, 10>(0, 0) = -K;
    B.block<10, 10>(UPS_DIM, UPS_DIM) = deltaUpsilon_.AdjointMap() * dt;

    // Create noise covariance Q_d in TANGENT SPACE
    Matrix20 Q_d = Matrix20::Zero();

    // First 10 dimensions in GTSAM tangent space [rho, nu, theta, t]
    Q_d.block<3,3>(0, 0) = params->getVirtualVelCovariance() / dt;        // rho
    Q_d.block<3,3>(3, 3) = params->accelerometerCovariance / dt;     // nu
    Q_d.block<3,3>(6, 6) = params->gyroscopeCovariance / dt;         // theta
    Q_d(9, 9) = params->getVirtualTimeScaleCovariance() / dt;             // t

    // Last 10 dimensions are bias random walk in GTSAM bias ordering [b_omega, b_acc, b_nu, b_rho]
    Q_d.block<3,3>(10, 10) = params->getBiasOmegaCovariance() / dt;       // gyro bias
    Q_d.block<3,3>(13, 13) = params->getBiasAccCovariance() / dt;         // acc bias
    Q_d.block<3,3>(16, 16) = params->getBiasVirtualVelCovariance() / dt;       // virtual velocity bias
    Q_d(19, 19) = params->getBiasVirtualTimeCovariance() / dt;                 // virtual time bias


    // Alternative Q_d in MEASUREMENT SPACE (for comparison)
    Matrix20 Q_d_measurement = Matrix20::Zero();
    Q_d_measurement.block<3,3>(0, 0) = params->gyroscopeCovariance / dt;         // omega (gyro)
    Q_d_measurement.block<3,3>(3, 3) = params->accelerometerCovariance / dt;     // acc
    Q_d_measurement.block<3,3>(6, 6) = params->getVirtualVelCovariance() / dt;        // virtual_vel
    Q_d_measurement(9, 9) = params->getVirtualTimeScaleCovariance() / dt;             // virtual_time
    Q_d_measurement.block<3,3>(10, 10) = params->getBiasOmegaCovariance() / dt;
    Q_d_measurement.block<3,3>(13, 13) = params->getBiasAccCovariance() / dt;
    Q_d_measurement.block<3,3>(16, 16) = params->getBiasVirtualVelCovariance() / dt;
    Q_d_measurement(19, 19) = params->getBiasVirtualTimeCovariance() / dt;

    // Debug: Print Q_d matrices for comparison
    std::cout << "\nQ_d (TANGENT space) diagonal elements:" << std::endl;
    std::cout << "  rho:   " << Q_d.block<3,3>(0,0).diagonal().transpose() << std::endl;
    std::cout << "  nu:    " << Q_d.block<3,3>(3,3).diagonal().transpose() << std::endl;
    std::cout << "  theta: " << Q_d.block<3,3>(6,6).diagonal().transpose() << std::endl;
    std::cout << "  t:     " << Q_d(9,9) << std::endl;
    std::cout << "  bias:  " << Q_d.diagonal().tail<10>().transpose() << std::endl;

    std::cout << "\nQ_d (MEASUREMENT space) diagonal elements:" << std::endl;
    std::cout << "  omega: " << Q_d_measurement.block<3,3>(0,0).diagonal().transpose() << std::endl;
    std::cout << "  acc:   " << Q_d_measurement.block<3,3>(3,3).diagonal().transpose() << std::endl;
    std::cout << "  v_virt:" << Q_d_measurement.block<3,3>(6,6).diagonal().transpose() << std::endl;
    std::cout << "  t_virt:" << Q_d_measurement(9,9) << std::endl;
    std::cout << "  bias:  " << Q_d_measurement.diagonal().tail<10>().transpose() << std::endl;

    // Update covariance
    std::cout << "\nCov_old (preintMeasCov_ BEFORE update):" << std::endl;
    std::cout << "  diagonal sum: " << preintMeasCov_.diagonal().sum() << std::endl;
    std::cout << "  trace: " << preintMeasCov_.trace() << std::endl;
    std::cout << "  norm: " << preintMeasCov_.norm() << std::endl;

    preintMeasCov_ = A * preintMeasCov_ * A.transpose() + B * Q_d * B.transpose();

    // incase we want to round to 2 for numerical stability
    const double epsilon = 1e-17;  // Adjust based on your precision needs
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            if (std::abs(preintMeasCov_(i, j)) < epsilon) {
                preintMeasCov_(i, j) = 0.0;
            }
        }
    }
    // preintMeasCov_ = 0.5 * (preintMeasCov_ + preintMeasCov_.transpose());

    // preintMeasCov_ = 0.5 * (preintMeasCov_ + preintMeasCov_.transpose());
    // preintMeasCov_ = A * preintMeasCov_ * A.transpose() + B * Q_d_measurement * B.transpose();

    std::cout << "\nCov_new (preintMeasCov_ AFTER update):" << std::endl;
    std::cout << "  diagonal sum: " << preintMeasCov_.diagonal().sum() << std::endl;
    std::cout << "  trace: " << preintMeasCov_.trace() << std::endl;
    std::cout << "  norm: " << preintMeasCov_.norm() << std::endl;

    // Debug: Show the contribution of the two terms in covariance update
    Matrix20 term1 = A * preintMeasCov_ * A.transpose();
    Matrix20 term2 = B * Q_d * B.transpose();
    std::cout << "\nA*Cov*A^T contribution:" << std::endl;
    std::cout << "  trace: " << term1.trace() << std::endl;
    std::cout << "  norm: " << term1.norm() << std::endl;
    std::cout << "\nB*Q*B^T contribution:" << std::endl;
    std::cout << "  trace: " << term2.trace() << std::endl;
    std::cout << "  norm: " << term2.norm() << std::endl;

    // Update bias Jacobian
    Matrix20 Phi_b = Matrix20::Identity();
    Phi_b.block<10, 10>(0, UPS_DIM) = -K;
    preintBiasJacobian_ = Phi_b * preintBiasJacobian_;

    std::cout << "\nJacobians updated successfully" << std::endl;
    std::cout << "===== END integrateMeasurement DEBUG =====\n" << std::endl;
}


Vector9 PreintegratedGalileanMeasurements::biasCorrectedDelta(
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 6> H) const {

    // Get bias difference in 10D tangent space
    Vector10 delta_b = mapBias6ToTangent10(bias_i.vector() - biasHat_.vector());

    // Apply bias correction
    Gal3 correction = Gal3::Expmap(preintBiasJacobian_.block<10,10>(0, UPS_DIM) * delta_b);
    Gal3 corrected = correction * deltaUpsilon_;

    // Extract components in NavState ordering [theta, p, v]
    Vector9 result;
    result.segment<3>(NAV_R_IDX) = Rot3::Logmap(corrected.rotation() * deltaUpsilon_.rotation().inverse());
    result.segment<3>(NAV_P_IDX) = corrected.position() - deltaUpsilon_.position();
    result.segment<3>(NAV_V_IDX) = corrected.velocity() - deltaUpsilon_.velocity();

    // Calculate Jacobian numerically if requested
    if (H) {
        auto compute_delta = [this](const imuBias::ConstantBias& b) {
            return this->biasCorrectedDelta(b, {});
        };
        *H = numericalDerivative11<Vector9, imuBias::ConstantBias, 6>(compute_delta, bias_i);
    }

    return result;
}

Matrix9 PreintegratedGalileanMeasurements::preintegratedNavStateCovariance() const {
  // Selection matrix to extract NavState from Upsilon
  static const Matrix9_10 S = [] {
    Matrix9_10 M = Matrix9_10::Zero();
    M.block<3,3>(NAV_R_IDX, THETA_IDX) = Matrix3::Identity();
    M.block<3,3>(NAV_P_IDX, RHO_IDX) = Matrix3::Identity();
    M.block<3,3>(NAV_V_IDX, NU_IDX) = Matrix3::Identity();
    return M;
  }();

  const Matrix10 JL_inv = Gal3::LogmapDerivative(deltaUpsilon_);
  const Matrix10 Ad = deltaUpsilon_.AdjointMap();

  Eigen::Matrix<double,9,20> T = Eigen::Matrix<double,9,20>::Zero();
  T.block<9,10>(0,0) = S;
  T.block<9,10>(0,UPS_DIM) = -S * JL_inv * Ad;

  Matrix9 P_nav = T * preintMeasCov_ * T.transpose();
  return 0.5 * (P_nav + P_nav.transpose());
}

Vector9 PreintegratedGalileanMeasurements::computeErrorAndJacobians(
    const Pose3& pose_i, const Vector3& vel_i,
    const Pose3& pose_j, const Vector3& vel_j,
    const imuBias::ConstantBias& bias_i,
    boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
    boost::optional<Matrix&> H3, boost::optional<Matrix&> H4,
    boost::optional<Matrix&> H5) const {

    // Apply bias correction
    Vector6 delta_bias = bias_i.vector() - biasHat_.vector();
    Vector10 delta_b_10D = mapBias6ToTangent10(delta_bias);

    Gal3 bias_correction = Gal3::Expmap(preintBiasJacobian_.block<10,10>(0, UPS_DIM) * delta_b_10D);
    Gal3 corrected_delta = bias_correction * deltaUpsilon_;

    // Extract corrected components
    const Rot3& deltaR = corrected_delta.rotation();
    const Vector3 deltaP = corrected_delta.position();
    const Vector3& deltaV = corrected_delta.velocity();

    // Get parameters
    auto params = galileanParams();
    const Vector3& gravity = params->n_gravity;
    double deltaT = deltaTij();

    // Compute predicted deltas
    const Rot3& R_i = pose_i.rotation();
    const Rot3& R_j = pose_j.rotation();

    // Relative rotation error
    Rot3 deltaR_pred = R_i.between(R_j);
    Vector3 error_R = Rot3::Logmap(deltaR_pred * deltaR.inverse());

    // Relative velocity error (in body frame)
    Vector3 v_err_world = vel_j - vel_i - gravity * deltaT;
    Vector3 deltaV_pred = R_i.unrotate(v_err_world);
    Vector3 error_v = deltaV_pred - deltaV;

    // Relative position error (in body frame)
    Vector3 p_err_world = pose_j.translation() - pose_i.translation() -
                         vel_i * deltaT - 0.5 * gravity * deltaT * deltaT;
    Vector3 deltaP_pred = R_i.unrotate(p_err_world);
    Vector3 error_p = deltaP_pred - deltaP;

    // Construct error vector
    Vector9 error;
    error.segment<3>(NAV_R_IDX) = error_R;
    error.segment<3>(NAV_P_IDX) = error_p;
    error.segment<3>(NAV_V_IDX) = error_v;

    // Compute Jacobians if requested
    if (H1 || H2 || H3 || H4 || H5) {
        Matrix3 R_i_matrix = R_i.matrix();
        Matrix3 R_j_matrix = R_j.matrix();

        if (H1) {  // wrt pose_i
            H1->resize(9, 6);
            H1->block<3,3>(NAV_R_IDX, 0) = -R_j_matrix.transpose();
            H1->block<3,3>(NAV_R_IDX, 3).setZero();
            H1->block<3,3>(NAV_P_IDX, 0) = skewSymmetric(deltaP_pred);
            H1->block<3,3>(NAV_P_IDX, 3) = -R_i_matrix.transpose();
            H1->block<3,3>(NAV_V_IDX, 0) = skewSymmetric(deltaV_pred);
            H1->block<3,3>(NAV_V_IDX, 3).setZero();
        }

        if (H2) {  // wrt vel_i
            H2->resize(9, 3);
            H2->block<3,3>(NAV_R_IDX, 0).setZero();
            H2->block<3,3>(NAV_P_IDX, 0) = -R_i_matrix.transpose() * deltaT;
            H2->block<3,3>(NAV_V_IDX, 0) = -R_i_matrix.transpose();
        }

        if (H3) {  // wrt pose_j
            H3->resize(9, 6);
            H3->block<3,3>(NAV_R_IDX, 0) = Matrix3::Identity();
            H3->block<3,3>(NAV_R_IDX, 3).setZero();
            H3->block<3,3>(NAV_P_IDX, 0).setZero();
            H3->block<3,3>(NAV_P_IDX, 3) = R_i_matrix.transpose();
            H3->block<3,3>(NAV_V_IDX, 0).setZero();
            H3->block<3,3>(NAV_V_IDX, 3).setZero();
        }

        if (H4) {  // wrt vel_j
            H4->resize(9, 3);
            H4->block<3,3>(NAV_R_IDX, 0).setZero();
            H4->block<3,3>(NAV_P_IDX, 0).setZero();
            H4->block<3,3>(NAV_V_IDX, 0) = R_i_matrix.transpose();
        }

        if (H5) {  // wrt bias_i
            Matrix96 H_bias = Matrix96::Zero();

            // Rotation component
            so3::DexpFunctor dexp_functor(error_R);
            Matrix3 Jr_inv = dexp_functor.rightJacobianInverse();
            H_bias.block<3,3>(NAV_R_IDX, 3) = -Jr_inv * deltaR_pred.matrix() *
                                            deltaR.inverse().matrix() *
                                            preintBiasJacobian_.block<3,3>(THETA_IDX, BIAS_START_IDX + B_OMEGA_IDX);

            // Position and velocity components
            H_bias.block<3,3>(NAV_P_IDX, 0) = -preintBiasJacobian_.block<3,3>(RHO_IDX, BIAS_START_IDX + B_ACC_IDX);
            H_bias.block<3,3>(NAV_P_IDX, 3) = -preintBiasJacobian_.block<3,3>(RHO_IDX, BIAS_START_IDX + B_OMEGA_IDX);
            H_bias.block<3,3>(NAV_V_IDX, 0) = -preintBiasJacobian_.block<3,3>(NU_IDX, BIAS_START_IDX + B_ACC_IDX);
            H_bias.block<3,3>(NAV_V_IDX, 3) = -preintBiasJacobian_.block<3,3>(NU_IDX, BIAS_START_IDX + B_OMEGA_IDX);

            *H5 = H_bias;
        }
    }

    return error;
}

NavState PreintegratedGalileanMeasurements::predict(const NavState& state_i,
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 9> H_navstate_wrt_navstate_i,
    OptionalJacobian<9, 6> H_navstate_wrt_bias_i) const {

    // Get the bias-corrected delta
    Matrix96 H_correction;
    Vector9 correction = biasCorrectedDelta(bias_i,
                           H_navstate_wrt_bias_i ? &H_correction : nullptr);

    // Extract corrected components
    Vector3 theta = correction.segment<3>(NAV_R_IDX);
    Vector3 deltaP = correction.segment<3>(NAV_P_IDX);
    Vector3 deltaV = correction.segment<3>(NAV_V_IDX);

    // Apply corrections
    Rot3 deltaR_corrected = deltaUpsilon_.rotation() * Rot3::Expmap(theta);
    Vector3 deltaP_corrected = deltaUpsilon_.position() + deltaP;
    Vector3 deltaV_corrected = deltaUpsilon_.velocity() + deltaV;

    // Apply standard physics
    const Rot3& R_i = state_i.attitude();
    const Vector3& v_i = state_i.velocity();
    const Point3& p_i = state_i.position();
    double deltaT = deltaTij();
    const Vector3& gravity = galileanParams()->n_gravity;

    // Predict state
    Rot3 R_j = R_i * deltaR_corrected;
    Vector3 v_j = v_i + R_i * deltaV_corrected + gravity * deltaT;
    Point3 p_j = p_i + Point3(R_i * deltaP_corrected + v_i * deltaT + 0.5 * gravity * deltaT * deltaT);

    NavState predicted(Pose3(R_j, p_j), v_j);

    // Compute Jacobians numerically if requested
    if (H_navstate_wrt_navstate_i) {
        auto predict_wrapper = [this](const NavState& s, const imuBias::ConstantBias& b) {
            return this->predict(s, b, {}, {});
        };
        *H_navstate_wrt_navstate_i = numericalDerivative21<NavState, NavState, imuBias::ConstantBias>(
            predict_wrapper, state_i, bias_i, 1e-7);
    }

    if (H_navstate_wrt_bias_i) {
        *H_navstate_wrt_bias_i = H_correction;
    }

    return predicted;
}


// --- GalileanImuFactor Implementation ---

GalileanImuFactor::GalileanImuFactor(Key pose_i, Key vel_i, Key bias_i,
                                     Key pose_j, Key vel_j, Key bias_j,
                                     const PreintegratedGalileanMeasurements& pim) :
    Base(noiseModel::Gaussian::Covariance(pim.preintegratedNavStateCovariance(), true),
         pose_i, vel_i, bias_i, pose_j, vel_j, bias_j),
    _PIM(pim)
{
}

gtsam::NonlinearFactor::shared_ptr GalileanImuFactor::clone() const {
  return std::static_pointer_cast<NonlinearFactor>(
      NonlinearFactor::shared_ptr(new This(*this)));
}

void GalileanImuFactor::print(const std::string& s,
                              const KeyFormatter& keyFormatter) const {
  std::cout << (s.empty() ? s : s + "\n") << "GalileanImuFactor("
       << keyFormatter(this->key<1>()) << "," << keyFormatter(this->key<2>()) << ","
       << keyFormatter(this->key<3>()) << "," << keyFormatter(this->key<4>()) << ","
       << keyFormatter(this->key<5>()) << "," << keyFormatter(this->key<6>()) << ")\n";
  _PIM.print("  Preintegrated Measurements:");
  if (this->noiseModel_)
    this->noiseModel_->print("  noise model: ");
  else
    std::cout << "  noise model: none" << std::endl;
}

bool GalileanImuFactor::equals(const NonlinearFactor& expected, double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) && _PIM.equals(e->_PIM, tol);
}

Vector GalileanImuFactor::evaluateError(const Pose3& pose_i, const Vector3& vel_i,
                                       const imuBias::ConstantBias& bias_i,
                                       const Pose3& pose_j, const Vector3& vel_j,
                                       const imuBias::ConstantBias& bias_j,
                                       OptionalMatrixType H1, OptionalMatrixType H2,
                                       OptionalMatrixType H3, OptionalMatrixType H4,
                                       OptionalMatrixType H5, OptionalMatrixType H6) const {
    // Part 1: Compute bias evolution error
    Matrix6 Hbias_i, Hbias_j;
    Vector6 fbias = traits<imuBias::ConstantBias>::Between(bias_j, bias_i,
        H6 ? &Hbias_j : nullptr, H3 ? &Hbias_i : nullptr).vector();

    // Part 2: Compute navigation state error
    boost::optional<Matrix&> pimH_pose_i = H1 ? boost::optional<Matrix&>(*H1) : boost::none;
    boost::optional<Matrix&> pimH_vel_i  = H2 ? boost::optional<Matrix&>(*H2) : boost::none;
    boost::optional<Matrix&> pimH_bias_i = H3 ? boost::optional<Matrix&>(*H3) : boost::none;
    boost::optional<Matrix&> pimH_pose_j = H4 ? boost::optional<Matrix&>(*H4) : boost::none;
    boost::optional<Matrix&> pimH_vel_j  = H5 ? boost::optional<Matrix&>(*H5) : boost::none;

    Vector9 r_nav = _PIM.computeErrorAndJacobians(pose_i, vel_i, pose_j, vel_j, bias_i,
                                                pimH_pose_i, pimH_vel_i,
                                                pimH_pose_j, pimH_vel_j,
                                                pimH_bias_i);

    // Combine errors
    Vector15 error;
    error << r_nav, fbias;

    // Structure Jacobians properly
    if (H3) {  // bias_i
        H3->resize(15, 6);
        H3->block<9, 6>(0, 0) = *pimH_bias_i;
        H3->block<6, 6>(9, 0) = Hbias_i;
    }

    if (H6) {  // bias_j
        H6->resize(15, 6);
        H6->block<9, 6>(0, 0).setZero();
        H6->block<6, 6>(9, 0) = Hbias_j;
    }

    return error;
}

} // namespace gtsam
