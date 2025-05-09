/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file CombinedGalileanImuFactor.cpp
 * @brief Implementation file for combined Galilean IMU preintegration and factor.
 * @author Refactored by [Your Tool Name/Your Name]
 */

#include <gtsam/navigation/GalileanImuFactor.h> // Corresponding header

// Additional includes from PreintegratedGalileanMeasurements.cpp
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/linear/GaussianFactor.h> // For Pose3::dimension (used in comments, might not be strictly needed)

#include <stdexcept>
#include <functional> // For std::bind, std::function
#include <limits>     // For NaN/Inf checks
#include <iomanip>    // For std::fixed, std::setprecision, std::scientific

// Additional includes from GalileanImuFactor.cpp
#include <ostream>
// <gtsam/base/Matrix.h> is in CombinedGalileanImuFactor.h
// <stdexcept> is already included

namespace gtsam {

// Anonymous namespace from PreintegratedGalileanMeasurements.cpp
// Define indices for accessing the 20D state vector/covariance/Jacobian
// Tangent space orderings:
// Upsilon tangent (deltaUpsilon_): [rho(p, 0-2), nu(v, 3-5), theta(R, 6-8), t(9)] -> Dim 10
// Bias tangent (b_k): [b_omega(0-2), b_acc(3-5), b_nu(6-8), b_rho(9)] -> Dim 10
// Combined 20D tangent (epsilon_k): [Upsilon_tangent | Bias_tangent]
namespace {
    // Dimensions
    const size_t ups_dim = 10;
    const size_t bias_dim = 10;
    // const size_t total_dim = ups_dim + bias_dim; // 20 // Unused

    // Indices within Upsilon tangent vector (0-9) [rho, nu, theta, t]
    const size_t ups_p_idx = 0; // rho component (position)
    const size_t ups_v_idx = 3; // nu component (velocity)
    const size_t ups_R_idx = 6; // theta component (rotation)
    const size_t ups_t_idx = 9; // t component (time duration)

    // Indices within Bias tangent vector (0-9) [b_omega, b_acc, b_nu, b_rho]
    const size_t bias_w_comp_idx = 0; // b_omega (gyro bias)
    const size_t bias_a_comp_idx = 3; // b_acc (accel bias)
    const size_t bias_nu_comp_idx = 6; // b_nu (virtual velocity bias)
    const size_t bias_rho_comp_idx = 9; // b_rho (virtual time bias)

    // Indices within the full 20D state tangent vector [Upsilon | Bias]
    const size_t bias_w_idx = ups_dim + bias_w_comp_idx; // 10
    const size_t bias_a_idx = ups_dim + bias_a_comp_idx; // 13
    const size_t bias_nu_idx = ups_dim + bias_nu_comp_idx; // 16
    const size_t bias_rho_idx = ups_dim + bias_rho_comp_idx; // 19

    // Define indices for accessing the 9D NavState tangent vector
    // Tangent space order: [theta(R, 0-2), p(3-5), v(6-8)]
    const size_t NAV_R_IDX = 0;
    const size_t NAV_P_IDX = 3;
    const size_t NAV_V_IDX = 6;

} // anonymous namespace


// Helper functions from PreintegratedGalileanMeasurements.cpp
// These functions are used internally by PreintegratedGalileanMeasurements::integrateMeasurement
void calculateQ1OmegaR(const Vector3& omega, const Vector3& r, Matrix3& result) {
    double omega_norm = omega.norm();

    if (omega_norm < 1e-8) {
        result = 0.5 * skewSymmetric(r);
        return;
    }
    double term1 = (omega_norm - sin(omega_norm)) / pow(omega_norm, 3);
    double term2 = (omega_norm * omega_norm + 2.0 * cos(omega_norm) - 2.0) / (2.0 * pow(omega_norm, 4));
    double term3 = (2.0 * omega_norm - 3.0 * sin(omega_norm) + omega_norm * cos(omega_norm)) / (2.0 * pow(omega_norm, 5));
    Matrix3 omega_hat = skewSymmetric(omega);
    Matrix3 r_hat = skewSymmetric(r);
    result = 0.5 * r_hat +
             term1 * (omega_hat * r_hat + r_hat * omega_hat + omega_hat * r_hat * omega_hat) +
             term2 * (omega_hat * omega_hat * r_hat + r_hat * omega_hat * omega_hat - 3.0 * omega_hat * r_hat * omega_hat) +
             term3 * (omega_hat * r_hat * omega_hat * omega_hat + omega_hat * omega_hat * r_hat * omega_hat);
}

void calculateQ2OmegaV(const Vector3& omega, const Vector3& v, Matrix3& result) {
    double omega_norm = omega.norm();

    if (omega_norm < 1e-8) {
        result = (1.0/6.0) * skewSymmetric(v);
        return;
    }
    Matrix3 omega_hat = skewSymmetric(omega);
    Matrix3 v_hat = skewSymmetric(v);
    double term1 = (omega_norm * omega_norm + 2.0 * cos(omega_norm) - 2.0) / (2.0 * pow(omega_norm, 4));
    double term2 = (omega_norm * omega_norm * omega_norm - 6.0 * omega_norm + 6.0 * sin(omega_norm)) / (6.0 * pow(omega_norm, 5));
    double term3 = (-2.0 * cos(omega_norm) - omega_norm * sin(omega_norm) + 2.0) / pow(omega_norm, 4);
    double term4 = (omega_norm * omega_norm * omega_norm + 6.0 * omega_norm * cos(omega_norm) + 6.0 * omega_norm - 12.0 * sin(omega_norm)) / (6.0 * pow(omega_norm, 5));
    double term5 = (-3.0 * omega_norm * cos(omega_norm) - (omega_norm * omega_norm - 3.0) * sin(omega_norm)) / (4.0 * pow(omega_norm, 5));
    double term6 = (omega_norm * cos(omega_norm) + 2.0 * omega_norm - 3.0 * sin(omega_norm)) / (4.0 * pow(omega_norm, 5));
    double term7 = ((omega_norm * omega_norm - 8.0) * cos(omega_norm) - 5.0 * omega_norm * sin(omega_norm) + 8.0) / (4.0 * pow(omega_norm, 6));
    double term8 = (2.0 * omega_norm * omega_norm * omega_norm + 15.0 * omega_norm * cos(omega_norm) + 3.0 * (omega_norm * omega_norm - 5.0) * sin(omega_norm)) / (12.0 * pow(omega_norm, 7));
    result = (1.0/6.0) * v_hat +
             term1 * v_hat * omega_hat +
             term2 * v_hat * omega_hat * omega_hat +
             term3 * omega_hat * v_hat +
             term4 * omega_hat * omega_hat * v_hat +
             term5 * omega_hat * v_hat * omega_hat +
             term6 * omega_hat * omega_hat * v_hat * omega_hat +
             term7 * omega_hat * v_hat * omega_hat * omega_hat +
             term8 * omega_hat * omega_hat * v_hat * omega_hat * omega_hat;
}

void calculateU1Omega(const Vector3& omega, Matrix3& result) {
    double omega_norm = omega.norm();
    if (omega_norm < 1e-8) {
        result = 0.5 * Matrix3::Identity();
        return;
    }
    double term1 = (sin(omega_norm) - omega_norm * cos(omega_norm)) / pow(omega_norm, 3);
    double term2 = (omega_norm * omega_norm - 2.0 * omega_norm * sin(omega_norm) - 2.0 * cos(omega_norm) + 2.0) / (2.0 * pow(omega_norm, 4));
    Matrix3 omega_hat = skewSymmetric(omega);
    result = 0.5 * Matrix3::Identity() + term1 * omega_hat + term2 * omega_hat * omega_hat;
}

void calculateQ1OmegaV(const Vector3& omega, const Vector3& z, Matrix3& result) {
    double omega_norm = omega.norm();
    if (omega_norm < 1e-8) {
        result = 0.5 * skewSymmetric(z);
        return;
    }
    double term1 = (omega_norm - sin(omega_norm)) / pow(omega_norm, 3);
    double term2 = (omega_norm * omega_norm + 2.0 * cos(omega_norm) - 2.0) / (2.0 * pow(omega_norm, 4));
    double term3 = (2.0 * omega_norm - 3.0 * sin(omega_norm) + omega_norm * cos(omega_norm)) / (2.0 * pow(omega_norm, 5));
    Matrix3 omega_hat = skewSymmetric(omega);
    Matrix3 z_hat = skewSymmetric(z);
    result = 0.5 * z_hat +
             term1 * (omega_hat * z_hat + z_hat * omega_hat + omega_hat * z_hat * omega_hat) +
             term2 * (omega_hat * omega_hat * z_hat + z_hat * omega_hat * omega_hat - 3.0 * omega_hat * z_hat * omega_hat) +
             term3 * (omega_hat * z_hat * omega_hat * omega_hat + omega_hat * omega_hat * z_hat * omega_hat);
}


// --- Implementations for PreintegratedGalileanMeasurements ---

// Constructor
PreintegratedGalileanMeasurements::PreintegratedGalileanMeasurements(
    const std::shared_ptr<Params>& p, const Bias& biasHat)
    : Base(p, biasHat),
      deltaUpsilon_(Gal3::Identity()),
      preintMeasCov_(Matrix20::Zero()),
      preintBiasJacobian_(Matrix20::Identity()),
      integration_step_counter_(0)
      {
    if (!std::dynamic_pointer_cast<Params>(p_)) {
        throw std::runtime_error("PreintegratedGalileanMeasurements requires a valid shared_ptr to GalileanPreintegrationParams.");
    }
    resetIntegration();
}

// resetIntegration
void PreintegratedGalileanMeasurements::resetIntegration() {
  deltaUpsilon_ = Gal3::Identity();
  deltaTij_ = 0.0;
  preintMeasCov_.setZero();
  preintBiasJacobian_.setIdentity();
  integration_step_counter_ = 0;

  if (p_) {
    auto galileanParamsPtr = std::dynamic_pointer_cast<const Params>(p_);
    if (galileanParamsPtr) {
        Matrix6 biasAccOmegaCov = galileanParamsPtr->getBiasAccOmegaInit();
        preintMeasCov_.block<3,3>(bias_w_idx, bias_w_idx) = biasAccOmegaCov.block<3,3>(3,3);
        preintMeasCov_.block<3,3>(bias_a_idx, bias_a_idx) = biasAccOmegaCov.block<3,3>(0,0);
        preintMeasCov_.block<3,3>(bias_nu_idx, bias_nu_idx).setZero();
        preintMeasCov_(bias_rho_idx, bias_rho_idx) = 0.0;
    } else {
        throw std::runtime_error("PreintegratedGalileanMeasurements requires GalileanPreintegrationParams in resetIntegration.");
    }
  }
}

std::shared_ptr<const PreintegratedGalileanMeasurements::Params>
PreintegratedGalileanMeasurements::galileanParams() const {
    if (!p_) {
        throw std::runtime_error("PreintegratedGalileanMeasurements: Parameters pointer is null.");
    }
    auto params_ptr = std::dynamic_pointer_cast<const Params>(p_);
    if (!params_ptr) {
        auto base_params_ptr = std::dynamic_pointer_cast<const PreintegrationParams>(p_);
        if (base_params_ptr) {
            throw std::runtime_error("PreintegratedGalileanMeasurements: Incorrect parameter type provided (expected GalileanPreintegrationParams).");
        } else {
            throw std::runtime_error("PreintegratedGalileanMeasurements: Invalid parameter pointer provided.");
        }
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
      && equal_with_abs_tol(preintBiasJacobian_, other.preintBiasJacobian_, tol)
      && integration_step_counter_ == other.integration_step_counter_;
}

Vector10 PreintegratedGalileanMeasurements::mapBias6ToTangent10(const Vector6& bias6D) {
    Vector10 result = Vector10::Zero();
    result.segment<3>(bias_w_comp_idx) = bias6D.tail<3>();
    result.segment<3>(bias_a_comp_idx) = bias6D.head<3>();
    return result;
}

Vector10 PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(const Vector10& measurement10D) {
    Vector10 tangent10D = Vector10::Zero();
    tangent10D.segment<3>(ups_v_idx) = measurement10D.segment<3>(bias_a_comp_idx);
    tangent10D.segment<3>(ups_R_idx) = measurement10D.segment<3>(bias_w_comp_idx);
    tangent10D(ups_t_idx) = measurement10D(bias_rho_comp_idx);
    return tangent10D;
}

void PreintegratedGalileanMeasurements::integrateMeasurement(
    const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {

    if (dt <= 0) {
        return;
    }
    integration_step_counter_++;

    auto params = galileanParams();
    // Per original PreintegratedGalileanMeasurements.cpp, measuredAcc and measuredOmega are used directly as bodyAcc/bodyOmega.
    // No sensor to body transformation block was present here in the original.
    Vector3 bodyAcc = measuredAcc;
    Vector3 bodyOmega = measuredOmega;

    const Gal3& Upsilon_hat_k = deltaUpsilon_;
    Vector10 b_hat_k = mapBias6ToTangent10(biasHat_.vector());

    Vector10 w_tilde_k = Vector10::Zero();
    w_tilde_k.segment<3>(bias_w_comp_idx) = bodyOmega;
    w_tilde_k.segment<3>(bias_a_comp_idx) = bodyAcc;
    w_tilde_k(bias_rho_comp_idx) = 1.0;

    Matrix20 Q_d = Matrix20::Zero();
    Q_d.block<3,3>(ups_R_idx, ups_R_idx) = params->gyroscopeCovariance * dt;
    Q_d.block<3,3>(ups_v_idx, ups_v_idx) = params->accelerometerCovariance * dt;
    Q_d.block<3,3>(bias_w_idx, bias_w_idx) = params->getBiasOmegaCovariance() * dt;
    Q_d.block<3,3>(bias_a_idx, bias_a_idx) = params->getBiasAccCovariance() * dt;

    Vector10 w_hat_k = w_tilde_k - b_hat_k;
    Vector10 zeta_k = mapMeasurement10ToTangent10(w_hat_k) * dt;

    Gal3 Exp_zeta_k = Gal3::Expmap(zeta_k);
    deltaUpsilon_ = Upsilon_hat_k * Exp_zeta_k;
    deltaTij_ += dt;

    bool is_zero_motion = (bodyAcc.norm() < 1e-10 && bodyOmega.norm() < 1e-10);

    Matrix10 Ad_Upsilon_hat_k = Upsilon_hat_k.AdjointMap();
    Matrix10 Ad_Upsilon_hat_inv_k = Upsilon_hat_k.inverse().AdjointMap();
    Vector10 w_bar_k_dt_tangent = mapMeasurement10ToTangent10(Ad_Upsilon_hat_inv_k * w_hat_k) * dt;
    Matrix10 JL_w_bar_k_dt = Gal3::ExpmapDerivative(w_bar_k_dt_tangent);
    Matrix10 JL_zeta_k = Gal3::ExpmapDerivative(zeta_k);
    Gal3 Exp_w_bar_k_dt = Gal3::Expmap(w_bar_k_dt_tangent);
    Matrix10 Ad_Exp_w_bar_k_dt = Exp_w_bar_k_dt.AdjointMap();

    Matrix20 A_hat_k1 = Matrix20::Identity();
    if (!is_zero_motion) {
        A_hat_k1.block<10, 10>(0, ups_dim) = JL_w_bar_k_dt * dt;
    }
    A_hat_k1.block<10,10>(ups_dim, ups_dim) = Ad_Exp_w_bar_k_dt;

    Matrix20 B_hat_k1 = Matrix20::Zero();
    B_hat_k1.block<10, 10>(0, 0) = -Ad_Upsilon_hat_k * JL_zeta_k * dt;
    B_hat_k1.block<10, 10>(ups_dim, ups_dim) = deltaUpsilon_.AdjointMap() * dt;

    preintMeasCov_ = A_hat_k1 * preintMeasCov_ * A_hat_k1.transpose() + B_hat_k1 * Q_d * B_hat_k1.transpose();
    preintMeasCov_ = (preintMeasCov_ + preintMeasCov_.transpose()) / 2.0;
    preintBiasJacobian_ = A_hat_k1 * preintBiasJacobian_;
}


Vector9 PreintegratedGalileanMeasurements::biasCorrectedDelta(
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 6> H) const {

    Vector6 delta_bias_6D = bias_i.vector() - biasHat_.vector();
    Vector10 delta_b_10D = mapBias6ToTangent10(delta_bias_6D);
    Vector10 bias_induced_correction_tangent = preintBiasJacobian_.block<10,10>(0, ups_dim) * delta_b_10D;
    Gal3 corrected_deltaUpsilon = Gal3::Expmap(bias_induced_correction_tangent) * deltaUpsilon_;

    Vector9 result_navstate_correction_9D;
    Rot3 orig_R = deltaUpsilon_.rotation();
    Vector3 orig_P = deltaUpsilon_.position();
    Vector3 orig_V = deltaUpsilon_.velocity();
    Rot3 corr_R = corrected_deltaUpsilon.rotation();
    Vector3 corr_P = corrected_deltaUpsilon.position();
    Vector3 corr_V = corrected_deltaUpsilon.velocity();

    result_navstate_correction_9D.segment<3>(NAV_R_IDX) = Rot3::Logmap(corr_R * orig_R.inverse());
    result_navstate_correction_9D.segment<3>(NAV_P_IDX) = corr_P - orig_P;
    result_navstate_correction_9D.segment<3>(NAV_V_IDX) = corr_V - orig_V;

    if (H) {
        std::function<Vector9(const imuBias::ConstantBias&)> fun =
            [this](const imuBias::ConstantBias& b) {
            return this->biasCorrectedDelta(b, {});
        };
        *H = numericalDerivative11<Vector9, imuBias::ConstantBias, 6>(fun, bias_i, 1e-7);
    }
    return result_navstate_correction_9D;
}

Matrix9 PreintegratedGalileanMeasurements::preintegratedNavStateCovariance() const {
  static const Matrix9_10 S = [] {
    Matrix9_10 M = Matrix9_10::Zero();
    M.block<3,3>(NAV_R_IDX, ups_R_idx) = Matrix3::Identity();
    M.block<3,3>(NAV_P_IDX, ups_p_idx) = Matrix3::Identity();
    M.block<3,3>(NAV_V_IDX, ups_v_idx) = Matrix3::Identity();
    return M;
  }();

  const Matrix10 JL_inv_log_deltaUpsilon = Gal3::LogmapDerivative(deltaUpsilon_);
  const Matrix10 Ad_deltaUpsilon = deltaUpsilon_.AdjointMap();
  Eigen::Matrix<double,9,20> T_nav_transform = Eigen::Matrix<double,9,20>::Zero();
  T_nav_transform.block<9,10>(0,0)         = S;
  T_nav_transform.block<9,10>(0,ups_dim) = -S * JL_inv_log_deltaUpsilon * Ad_deltaUpsilon;
  Matrix9 P_nav = T_nav_transform * preintMeasCov_ * T_nav_transform.transpose();
  return 0.5 * (P_nav + P_nav.transpose());
}

Vector9 PreintegratedGalileanMeasurements::computeErrorAndJacobians(
    const Pose3& pose_i, const Vector3& vel_i,
    const Pose3& pose_j, const Vector3& vel_j,
    const imuBias::ConstantBias& bias_i,
    boost::optional<Matrix&> H_pose_i, boost::optional<Matrix&> H_vel_i,
    boost::optional<Matrix&> H_pose_j, boost::optional<Matrix&> H_vel_j,
    boost::optional<Matrix&> H_bias_i) const {

    Matrix96 H_pim_correction_wrt_bias_i_calc;
    Vector9 pim_navstate_correction = biasCorrectedDelta(bias_i, H_pim_correction_wrt_bias_i_calc);

    const Rot3& deltaR_nominal = deltaUpsilon_.rotation();
    const Vector3 deltaP_nominal = deltaUpsilon_.position();
    const Vector3& deltaV_nominal = deltaUpsilon_.velocity();

    Rot3 deltaR_corrected = deltaR_nominal * Rot3::Expmap(pim_navstate_correction.segment<3>(NAV_R_IDX));
    Vector3 deltaP_corrected = deltaP_nominal + pim_navstate_correction.segment<3>(NAV_P_IDX);
    Vector3 deltaV_corrected = deltaV_nominal + pim_navstate_correction.segment<3>(NAV_V_IDX);

    auto params = galileanParams();
    const Vector3& n_gravity_w = params->n_gravity;
    double deltaT = deltaTij();
    const Rot3& R_i = pose_i.rotation();
    const Vector3 p_i_val = pose_i.translation(); // Use a different name to avoid conflict with p_i in lambda
    const Rot3& R_j = pose_j.rotation();
    const Vector3 p_j_val = pose_j.translation(); // Use a different name

    Rot3 deltaR_pred = R_i.between(R_j);
    Vector3 v_err_w = vel_j - vel_i - n_gravity_w * deltaT;
    Vector3 deltaV_pred = R_i.unrotate(v_err_w);
    Vector3 p_err_w = p_j_val - p_i_val - vel_i * deltaT - 0.5 * n_gravity_w * deltaT * deltaT;
    Vector3 deltaP_pred = R_i.unrotate(p_err_w);

    Vector3 error_R = Rot3::Logmap(deltaR_pred * deltaR_corrected.inverse());
    Vector3 error_p = deltaP_pred - deltaP_corrected;
    Vector3 error_v = deltaV_pred - deltaV_corrected;

    Vector9 error9D;
    error9D.segment<3>(NAV_R_IDX) = error_R;
    error9D.segment<3>(NAV_P_IDX) = error_p;
    error9D.segment<3>(NAV_V_IDX) = error_v;

    if (H_pose_i || H_vel_i || H_pose_j || H_vel_j || H_bias_i) {
        std::function<Vector9(const Pose3&, const Vector3&, const Pose3&, const Vector3&, const imuBias::ConstantBias&)>
        compute_full_error_for_jacobian_lambda =
            [this](const Pose3& current_pose_i, const Vector3& current_vel_i,
                   const Pose3& current_pose_j, const Vector3& current_vel_j,
                   const imuBias::ConstantBias& current_bias_i) -> Vector9 {

            Vector9 current_pim_corr = this->biasCorrectedDelta(current_bias_i, {});
            const Rot3& c_deltaR_nominal = this->deltaUpsilon_.rotation();
            const Vector3 c_deltaP_nominal = this->deltaUpsilon_.position();
            const Vector3& c_deltaV_nominal = this->deltaUpsilon_.velocity();
            Rot3 c_deltaR_corr = c_deltaR_nominal * Rot3::Expmap(current_pim_corr.segment<3>(NAV_R_IDX));
            Vector3 c_deltaP_corr = c_deltaP_nominal + current_pim_corr.segment<3>(NAV_P_IDX);
            Vector3 c_deltaV_corr = c_deltaV_nominal + current_pim_corr.segment<3>(NAV_V_IDX);

            const Rot3& cR_i = current_pose_i.rotation();
            const Vector3 cp_i = current_pose_i.translation();
            const Rot3& cR_j = current_pose_j.rotation();
            const Vector3 cp_j = current_pose_j.translation();
            double cDeltaT = this->deltaTij();
            const Vector3& cn_gravity_w = this->galileanParams()->n_gravity;

            Rot3 cDeltaR_pred = cR_i.between(cR_j);
            Vector3 cv_err_w = current_vel_j - current_vel_i - cn_gravity_w * cDeltaT;
            Vector3 cDeltaV_pred = cR_i.unrotate(cv_err_w);
            Vector3 cp_err_w = cp_j - cp_i - current_vel_i * cDeltaT - 0.5 * cn_gravity_w * cDeltaT * cDeltaT;
            Vector3 cDeltaP_pred = cR_i.unrotate(cp_err_w);

            Vector3 cError_R = Rot3::Logmap(cDeltaR_pred * c_deltaR_corr.inverse());
            Vector3 cError_p = cDeltaP_pred - c_deltaP_corr;
            Vector3 cError_v = cDeltaV_pred - c_deltaV_corr;

            Vector9 cError9D_lambda; // Use different name for clarity
            cError9D_lambda.segment<3>(NAV_R_IDX) = cError_R;
            cError9D_lambda.segment<3>(NAV_P_IDX) = cError_p;
            cError9D_lambda.segment<3>(NAV_V_IDX) = cError_v;
            return cError9D_lambda;
        };

        double numerical_step = 1e-7;
        if (H_pose_i) {
            *H_pose_i = numericalDerivative51<Vector9, Pose3, Vector3, Pose3, Vector3, imuBias::ConstantBias>(
                compute_full_error_for_jacobian_lambda, pose_i, vel_i, pose_j, vel_j, bias_i, numerical_step);
        }
        if (H_vel_i) {
            *H_vel_i = numericalDerivative52<Vector9, Pose3, Vector3, Pose3, Vector3, imuBias::ConstantBias>(
                compute_full_error_for_jacobian_lambda, pose_i, vel_i, pose_j, vel_j, bias_i, numerical_step);
        }
        if (H_pose_j) {
             *H_pose_j = numericalDerivative53<Vector9, Pose3, Vector3, Pose3, Vector3, imuBias::ConstantBias>(
                compute_full_error_for_jacobian_lambda, pose_i, vel_i, pose_j, vel_j, bias_i, numerical_step);
        }
        if (H_vel_j) {
            *H_vel_j = numericalDerivative54<Vector9, Pose3, Vector3, Pose3, Vector3, imuBias::ConstantBias>(
                compute_full_error_for_jacobian_lambda, pose_i, vel_i, pose_j, vel_j, bias_i, numerical_step);
        }
        if (H_bias_i) {
            *H_bias_i = -H_pim_correction_wrt_bias_i_calc;
        }
    }
    return error9D;
}

NavState PreintegratedGalileanMeasurements::predict(const NavState& state_i,
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 9> H_navstate_wrt_navstate_i,
    OptionalJacobian<9, 6> H_navstate_wrt_bias_i) const {

    Matrix96 H_pim_correction_wrt_bias_diff_calc;
    Vector9 pim_navstate_correction = biasCorrectedDelta(bias_i, H_pim_correction_wrt_bias_diff_calc);

    const Rot3& deltaR_nominal = deltaUpsilon_.rotation();
    const Vector3 deltaP_nominal = deltaUpsilon_.position();
    const Vector3& deltaV_nominal = deltaUpsilon_.velocity();

    Rot3 deltaR_corrected = deltaR_nominal * Rot3::Expmap(pim_navstate_correction.segment<3>(NAV_R_IDX));
    Vector3 deltaP_corrected = deltaP_nominal + pim_navstate_correction.segment<3>(NAV_P_IDX);
    Vector3 deltaV_corrected = deltaV_nominal + pim_navstate_correction.segment<3>(NAV_V_IDX);

    const Pose3& pose_i = state_i.pose();
    const Rot3& R_i = pose_i.rotation();
    const Point3& p_i_pt = pose_i.translation();
    const Vector3& vel_i = state_i.velocity();

    auto params = galileanParams();
    const Vector3& n_gravity_w = params->n_gravity;
    double deltaT = deltaTij();
    double deltaT2 = deltaT * deltaT;

    Rot3 R_j = R_i * deltaR_corrected;
    Vector3 vel_j = vel_i + R_i * deltaV_corrected + n_gravity_w * deltaT;
    Point3 p_j = p_i_pt + Point3(R_i * deltaP_corrected + vel_i * deltaT + 0.5 * n_gravity_w * deltaT2);

    if (H_navstate_wrt_navstate_i || H_navstate_wrt_bias_i) {
        std::function<NavState(const NavState&, const imuBias::ConstantBias&)>
        predict_wrapper_for_jacobian_lambda =
            [this](const NavState& current_state_i, const imuBias::ConstantBias& current_bias_i) -> NavState {
            Vector9 current_pim_corr = this->biasCorrectedDelta(current_bias_i, {});

            const Rot3& c_deltaR_nominal = this->deltaUpsilon_.rotation();
            const Vector3 c_deltaP_nominal = this->deltaUpsilon_.position();
            const Vector3& c_deltaV_nominal = this->deltaUpsilon_.velocity();

            Rot3 c_deltaR = c_deltaR_nominal * Rot3::Expmap(current_pim_corr.segment<3>(NAV_R_IDX));
            Vector3 c_deltaP = c_deltaP_nominal + current_pim_corr.segment<3>(NAV_P_IDX);
            Vector3 c_deltaV = c_deltaV_nominal + current_pim_corr.segment<3>(NAV_V_IDX);

            const Rot3& cR_i = current_state_i.attitude();
            const Point3& cp_i_pt = current_state_i.position();
            const Vector3& cvel_i = current_state_i.velocity();
            double cDeltaT = this->deltaTij();
            double cDeltaT2 = cDeltaT * cDeltaT;
            const Vector3& cn_gravity_w = this->galileanParams()->n_gravity;

            Rot3 cR_j = cR_i * c_deltaR;
            Vector3 cvel_j = cvel_i + cR_i * c_deltaV + cn_gravity_w * cDeltaT;
            Point3 cp_j = cp_i_pt + Point3(cR_i * c_deltaP + cvel_i * cDeltaT + 0.5 * cn_gravity_w * cDeltaT2);
            return NavState(Pose3(cR_j, cp_j), cvel_j);
        };

        double numerical_step_predict = 1e-7; // Consistent delta for these Jacobians

        if (H_navstate_wrt_navstate_i) {
            *H_navstate_wrt_navstate_i = numericalDerivative21<NavState, NavState, imuBias::ConstantBias>(
                predict_wrapper_for_jacobian_lambda, state_i, bias_i, numerical_step_predict);
        }
        if (H_navstate_wrt_bias_i) {
            *H_navstate_wrt_bias_i = numericalDerivative22<NavState, NavState, imuBias::ConstantBias>(
                predict_wrapper_for_jacobian_lambda, state_i, bias_i, numerical_step_predict);
        }
    }
    return NavState(Pose3(R_j, p_j), vel_j);
}


// --- Implementations for GalileanImuFactor ---

GalileanImuFactor::GalileanImuFactor(Key key_pose_i, Key key_vel_i, Key key_bias_i,
                                     Key key_pose_j, Key key_vel_j, Key key_bias_j,
                                     const PreintegratedGalileanMeasurements& pim) :
    Base(noiseModel::Gaussian::Covariance(pim.preintegratedNavStateCovariance(), true /* smart */),
         key_pose_i, key_vel_i, key_bias_i, key_pose_j, key_vel_j, key_bias_j),
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
    this->noiseModel_->print("  noise model (based on PIM 9x9 Covariance): ");
  else
    std::cout << "  noise model: none" << std::endl;
}

bool GalileanImuFactor::equals(const NonlinearFactor& expected, double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  return e != nullptr && Base::equals(*e, tol) && _PIM.equals(e->_PIM, tol);
}

Vector GalileanImuFactor::evaluateError(const Pose3& pose_i, const Vector3& vel_i, const imuBias::ConstantBias& bias_i,
                                       const Pose3& pose_j, const Vector3& vel_j, const imuBias::ConstantBias& bias_j,
                                       OptionalMatrixType H1, OptionalMatrixType H2,
                                       OptionalMatrixType H3,
                                       OptionalMatrixType H4, OptionalMatrixType H5,
                                       OptionalMatrixType H6) const {
    boost::optional<Matrix&> pimH_pose_i = H1 ? boost::optional<Matrix&>(*H1) : boost::none;
    boost::optional<Matrix&> pimH_vel_i  = H2 ? boost::optional<Matrix&>(*H2) : boost::none;
    boost::optional<Matrix&> pimH_pose_j = H4 ? boost::optional<Matrix&>(*H4) : boost::none;
    boost::optional<Matrix&> pimH_vel_j  = H5 ? boost::optional<Matrix&>(*H5) : boost::none;
    boost::optional<Matrix&> pimH_bias_i = H3 ? boost::optional<Matrix&>(*H3) : boost::none;

    Vector9 error = _PIM.computeErrorAndJacobians(pose_i, vel_i, pose_j, vel_j, bias_i,
                                                pimH_pose_i, pimH_vel_i,
                                                pimH_pose_j,
                                                pimH_vel_j,
                                                pimH_bias_i);
    if (H6) {
        H6->setZero(9, 6);
    }
    return error;
}

} // namespace gtsam
