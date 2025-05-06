/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file PreintegratedGalileanMeasurements.cpp
 * @brief Implementation file for PreintegratedGalileanMeasurements class.
 * @author (Your Name)
 */

#include <gtsam/navigation/PreintegratedGalileanMeasurements.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/OptionalJacobian.h>
#include <gtsam/linear/GaussianFactor.h> // For Pose3::dimension

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include <limits> // For NaN/Inf checks

// Define this macro to switch between numerical and (future) analytical Jacobians
// #define GTSAM_GALILEAN_USE_ANALYTICAL_JACOBIANS 0 // Set to 1 when implemented

// *** DEBUGGING FLAG ***
#define GTSAM_GALILEAN_PIM_DEBUG_PRINT 0 // Set back to 0 or remove for final version

namespace gtsam {

// Define indices for accessing the 20D state vector/covariance/Jacobian
// Tangent space orderings:
// Upsilon tangent (deltaUpsilon_): [rho(p, 0-2), nu(v, 3-5), theta(R, 6-8), t(9)] -> Dim 10
// Bias tangent (b_k): [b_omega(0-2), b_acc(3-5), b_nu(6-8), b_rho(9)] -> Dim 10
// Combined 20D tangent (epsilon_k): [Upsilon_tangent | Bias_tangent]
namespace {
    // Dimensions
    const size_t ups_dim = 10;
    const size_t bias_dim = 10;
    const size_t total_dim = ups_dim + bias_dim; // 20

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


// Constructor
PreintegratedGalileanMeasurements::PreintegratedGalileanMeasurements(
    const std::shared_ptr<Params>& p, const Bias& biasHat)
    : Base(p, biasHat), // Call base class constructor
      deltaUpsilon_(Gal3::Identity()),
      preintMeasCov_(Matrix20::Zero()),
      preintBiasJacobian_(Matrix20::Identity())
      {
    // Check that the provided parameters pointer is valid and of the correct type
    if (!std::dynamic_pointer_cast<Params>(p_)) {
         throw std::runtime_error("PreintegratedGalileanMeasurements requires a valid shared_ptr to GalileanPreintegrationParams.");
    }
    resetIntegration(); // Initialize state and covariance correctly
}

// resetIntegration
void PreintegratedGalileanMeasurements::resetIntegration() {
  deltaUpsilon_ = Gal3::Identity();
  deltaTij_ = 0.0; // Reset time in base class
  preintMeasCov_.setZero();
  preintBiasJacobian_.setIdentity();

  // Initialize bias covariance based on initial uncertainty from params
  if (p_) {
    auto galileanParamsPtr = std::dynamic_pointer_cast<const Params>(p_);
    if (galileanParamsPtr) {
        Matrix6 biasAccOmegaCov = galileanParamsPtr->getBiasAccOmegaInit();
        // Map the 6x6 initial bias covariance into the bias block of 20x20 Sigma_0
        // Order: [rho, nu, theta, t | b_omega, b_acc, b_nu, b_rho]
        preintMeasCov_.block<3,3>(bias_w_idx, bias_w_idx) = biasAccOmegaCov.block<3,3>(3,3); // Gyro bias cov
        preintMeasCov_.block<3,3>(bias_a_idx, bias_a_idx) = biasAccOmegaCov.block<3,3>(0,0); // Accel bias cov
        // Initial covariance for virtual biases b_nu, b_rho is zero
        preintMeasCov_.block<3,3>(bias_nu_idx, bias_nu_idx).setZero();
        preintMeasCov_(bias_rho_idx, bias_rho_idx) = 0.0;
    } else {
        // Constructor should have already caught this, but double-check
        throw std::runtime_error("PreintegratedGalileanMeasurements requires GalileanPreintegrationParams in resetIntegration.");
    }
  }
}

// galileanParams (const version)
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


// print
void PreintegratedGalileanMeasurements::print(const std::string& s) const {
  Base::print(s); // Call base class print (prints biasHat_ and deltaTij_)
  std::cout << "  deltaUpsilon:" << std::endl;
  deltaUpsilon_.print("    "); // Use Gal3's print method
}

// equals
bool PreintegratedGalileanMeasurements::equals(
    const PreintegratedGalileanMeasurements& other, double tol) const {
  // Check base class equality (params pointer, biasHat_, deltaTij_)
  if (!Base::matchesParamsWith(other) || // Check if params pointers match
      !biasHat_.equals(other.biasHat_, tol) ||
      std::fabs(deltaTij_ - other.deltaTij_) > tol) {
    return false;
  }
  // Check derived class members
  return deltaUpsilon_.equals(other.deltaUpsilon_, tol)
      && equal_with_abs_tol(preintMeasCov_, other.preintMeasCov_, tol)
      && equal_with_abs_tol(preintBiasJacobian_, other.preintBiasJacobian_, tol);
}

// mapBias6ToTangent10
Vector10 PreintegratedGalileanMeasurements::mapBias6ToTangent10(const Vector6& bias6D) {
    Vector10 bias10D = Vector10::Zero();
    bias10D.segment<3>(bias_a_comp_idx) = bias6D.head<3>();
    bias10D.segment<3>(bias_w_comp_idx) = bias6D.tail<3>();
    return bias10D;
}

// mapMeasurement10ToTangent10
Vector10 PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(const Vector10& measurement10D) {
    Vector10 tangent10D = Vector10::Zero();
    tangent10D.segment<3>(ups_v_idx) = measurement10D.segment<3>(bias_a_comp_idx); // acc -> nu
    tangent10D.segment<3>(ups_R_idx) = measurement10D.segment<3>(bias_w_comp_idx); // omega -> theta
    tangent10D(ups_t_idx)           = measurement10D(bias_rho_comp_idx);          // rho=1 -> t=1
    return tangent10D;
}


// integrateMeasurement - Algorithm 1 Core
void PreintegratedGalileanMeasurements::integrateMeasurement(
    const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {

    if (dt <= 0) {
         return; // Skip integration for non-positive dt
    }

    // DEBUG - Print input values
    std::cout << "DEBUG [integrateMeasurement]:" << std::endl;
    std::cout << "  dt: " << dt << std::endl;
    std::cout << "  measuredAcc: " << measuredAcc.transpose() << std::endl;
    std::cout << "  measuredOmega: " << measuredOmega.transpose() << std::endl;

    auto params = galileanParams();
    Vector3 bodyAcc = measuredAcc;
    Vector3 bodyOmega = measuredOmega;
    // Sensor frame handling omitted for brevity, assume body frame measurements

    const Gal3& Upsilon_hat_k = deltaUpsilon_;
    Vector10 b_hat_k = mapBias6ToTangent10(biasHat_.vector());
    Vector10 w_tilde_k = Vector10::Zero();
    w_tilde_k.segment<3>(bias_w_comp_idx) = bodyOmega;
    w_tilde_k.segment<3>(bias_a_comp_idx) = bodyAcc;
    w_tilde_k(bias_rho_comp_idx) = 1.0;

    // DEBUG - Print transformed quantities
    std::cout << "  b_hat_k: " << b_hat_k.transpose() << std::endl;
    std::cout << "  w_tilde_k: " << w_tilde_k.transpose() << std::endl;

    Matrix20 Q_d = Matrix20::Zero();
    double inv_dt = 1.0 / dt;
    Q_d.block<3,3>(ups_R_idx, ups_R_idx) = params->gyroscopeCovariance * inv_dt;
    Q_d.block<3,3>(bias_a_comp_idx, bias_a_comp_idx) = params->accelerometerCovariance * inv_dt;
    Q_d.block<3,3>(bias_w_idx, bias_w_idx) = params->getBiasOmegaCovariance() * dt;
    Q_d.block<3,3>(bias_a_idx, bias_a_idx) = params->getBiasAccCovariance() * dt;
    std::cout << "  Q_d gyro block (0-2,0-2):\n" << Q_d.block<3,3>(bias_w_comp_idx, bias_w_comp_idx) << std::endl;
    std::cout << "  Q_d rot indices block (6-8,6-8):\n" << Q_d.block<3,3>(ups_R_idx, ups_R_idx) << std::endl;

    // DEBUG - Print noise matrix Q_d
    std::cout << "  Q_d diagonal: " << Q_d.diagonal().transpose() << std::endl;

    Vector10 w_hat_k = w_tilde_k - b_hat_k;
    Vector10 zeta_k = mapMeasurement10ToTangent10(w_hat_k) * dt;

    // DEBUG - Print intermediate values
    std::cout << "  w_hat_k: " << w_hat_k.transpose() << std::endl;
    std::cout << "  zeta_k: " << zeta_k.transpose() << std::endl;

    Gal3 Exp_zeta_k = Gal3::Expmap(zeta_k);
    Gal3 prev_deltaUpsilon = deltaUpsilon_;
    deltaUpsilon_ = Upsilon_hat_k * Exp_zeta_k;
    deltaTij_ += dt;

    // DEBUG - Print Galilean operations
    std::cout << "  deltaUpsilon_ before update:" << std::endl;
    prev_deltaUpsilon.print("    ");
    std::cout << "  Exp_zeta_k:" << std::endl;
    Exp_zeta_k.print("    ");
    std::cout << "  deltaUpsilon_ after update:" << std::endl;
    deltaUpsilon_.print("    ");

    Matrix10 Ad_Upsilon_hat_inv_k = Upsilon_hat_k.inverse().AdjointMap();
    Vector10 w_hat_lifted_k = Ad_Upsilon_hat_inv_k * w_hat_k;
    Vector10 exp_arg_lifted = w_hat_lifted_k * dt;
    Matrix10 JL_exp_arg_lifted = Gal3::ExpmapDerivative(exp_arg_lifted);
    Matrix10 JL_zeta_k = Gal3::ExpmapDerivative(zeta_k);
    Gal3 Exp_w_hat_lifted_dt = Gal3::Expmap(exp_arg_lifted);
    Matrix10 Ad_Exp_w_hat_lifted_dt = Exp_w_hat_lifted_dt.AdjointMap();
    Matrix10 Ad_Upsilon_hat_k = Upsilon_hat_k.AdjointMap();

    // DEBUG - Print adjoint operations
    std::cout << "  Ad_Upsilon_hat_inv_k norm: " << Ad_Upsilon_hat_inv_k.norm() << std::endl;
    std::cout << "  JL_exp_arg_lifted norm: " << JL_exp_arg_lifted.norm() << std::endl;
    std::cout << "  JL_zeta_k norm: " << JL_zeta_k.norm() << std::endl;

    // Check for NaN/Inf in matrices
    if (!JL_exp_arg_lifted.allFinite()) {
        std::cerr << "ERROR: NaN/Inf in JL_exp_arg_lifted!" << std::endl;
    }
    if (!JL_zeta_k.allFinite()) {
        std::cerr << "ERROR: NaN/Inf in JL_zeta_k!" << std::endl;
    }

    Matrix20 A_hat_k1 = Matrix20::Identity();
    A_hat_k1.block<10, 10>(0, 10) = JL_exp_arg_lifted * dt;
    A_hat_k1.block<10, 10>(10, 10) = Ad_Exp_w_hat_lifted_dt;

    Matrix20 B_hat_k1 = Matrix20::Zero();
    B_hat_k1.block<10, 10>(0, 0) = -Ad_Upsilon_hat_k * JL_zeta_k * dt;
    B_hat_k1.block<10, 10>(10, 10) = deltaUpsilon_.AdjointMap() * dt;
    std::cout << "  B_hat_k1 mapping block (6-8,0-2):\n" << B_hat_k1.block<3,3>(ups_R_idx, bias_w_comp_idx) << std::endl;

    // DEBUG - Check matrix norms
    std::cout << "  A_hat_k1 norm: " << A_hat_k1.norm() << std::endl;
    std::cout << "  B_hat_k1 norm: " << B_hat_k1.norm() << std::endl;

    Matrix3 rot_cov_before = preintMeasCov_.block<3,3>(ups_R_idx, ups_R_idx);

    Matrix20 prev_preintMeasCov = preintMeasCov_;
    preintMeasCov_ = A_hat_k1 * preintMeasCov_ * A_hat_k1.transpose() + B_hat_k1 * Q_d * B_hat_k1.transpose();
    preintMeasCov_ = (preintMeasCov_ + preintMeasCov_.transpose()) / 2.0; // Ensure symmetry

    // DEBUG - Print covariance changes
    std::cout << "  preintMeasCov_ trace before: " << prev_preintMeasCov.trace() << std::endl;
    std::cout << "  preintMeasCov_ trace after: " << preintMeasCov_.trace() << std::endl;
    Matrix3 rot_cov_after = preintMeasCov_.block<3,3>(ups_R_idx, ups_R_idx);
    std::cout << "  preintMeasCov_ rot block before:\n" << rot_cov_before << std::endl;
    std::cout << "  preintMeasCov_ rot block after:\n" << rot_cov_after << std::endl;
    std::cout << "  rot block trace before: " << rot_cov_before.trace() << std::endl;
    std::cout << "  rot block trace after: " << rot_cov_after.trace() << std::endl;

    // Check for NaN/Inf in covariance
    if (!preintMeasCov_.allFinite()) {
        std::cerr << "ERROR: NaN/Inf detected in preintMeasCov_!" << std::endl;
        std::cout << "  A_hat_k1 * prev_preintMeasCov * A_hat_k1.T norm: "
                  << (A_hat_k1 * prev_preintMeasCov * A_hat_k1.transpose()).norm() << std::endl;
        std::cout << "  B_hat_k1 * Q_d * B_hat_k1.T norm: "
                  << (B_hat_k1 * Q_d * B_hat_k1.transpose()).norm() << std::endl;
    }

    Matrix20 Phi_b_k1 = Matrix20::Identity();
    Matrix10 J_bias_update = -Ad_Upsilon_hat_k * JL_zeta_k * dt;
    Phi_b_k1.block<10, 10>(0, 10) = J_bias_update;
    preintBiasJacobian_ = Phi_b_k1 * preintBiasJacobian_;

    // DEBUG - Print bias Jacobian info
    std::cout << "  J_bias_update norm: " << J_bias_update.norm() << std::endl;
    std::cout << "  preintBiasJacobian_ norm: " << preintBiasJacobian_.norm() << std::endl;

    // Check for NaN/Inf in Jacobian
    if (!preintBiasJacobian_.allFinite()) {
        std::cerr << "ERROR: NaN/Inf detected in preintBiasJacobian_!" << std::endl;
    }

    std::cout << "  Integration complete for dt = " << dt << std::endl;
}



// biasCorrectedDelta - Eq 39 applied for NavState error
Vector9 PreintegratedGalileanMeasurements::biasCorrectedDelta(
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 6> H) const {

    // DEBUG - Print input bias vs original bias
    std::cout << "DEBUG [biasCorrectedDelta]:" << std::endl;
    std::cout << "  biasHat_: " << biasHat_.vector().transpose() << std::endl;
    std::cout << "  bias_i: " << bias_i.vector().transpose() << std::endl;

    Vector6 delta_bias_6D = bias_i.vector() - biasHat_.vector();
    const Matrix10& J_Upsilon_bias = preintBiasJacobian_.block<10, 10>(0, ups_dim);
    Vector10 delta_b_10D = mapBias6ToTangent10(delta_bias_6D);

    // DEBUG - Print mapped bias
    std::cout << "  delta_bias_6D: " << delta_bias_6D.transpose() << std::endl;
    std::cout << "  delta_b_10D: " << delta_b_10D.transpose() << std::endl;
    std::cout << "  J_Upsilon_bias front corner: \n" << J_Upsilon_bias.block<3,3>(0,0) << std::endl;

    Vector10 correction_tangent = J_Upsilon_bias * delta_b_10D;

    // DEBUG - Check for NaN/Inf
    if (!J_Upsilon_bias.allFinite()) {
        std::cerr << "ERROR: NaN/Inf in J_Upsilon_bias!" << std::endl;
    }
    if (!correction_tangent.allFinite()) {
        std::cerr << "ERROR: NaN/Inf in correction_tangent!" << std::endl;
    }

    Vector9 result;
    result.segment<3>(NAV_R_IDX) = correction_tangent.segment<3>(ups_R_idx);
    result.segment<3>(NAV_P_IDX) = correction_tangent.segment<3>(ups_p_idx);
    result.segment<3>(NAV_V_IDX) = correction_tangent.segment<3>(ups_v_idx);

    // DEBUG - Print output correction
    std::cout << "  correction_tangent: " << correction_tangent.transpose() << std::endl;
    std::cout << "  result: " << result.transpose() << std::endl;

    if (H) {
        Eigen::Matrix<double, 10, 6> Mapper = Eigen::Matrix<double, 10, 6>::Zero();
        Mapper.block<3,3>(bias_a_comp_idx, 0) = Matrix3::Identity();
        Mapper.block<3,3>(bias_w_comp_idx, 3) = Matrix3::Identity();

        Matrix9_10 Selector = Matrix9_10::Zero();
        Selector.block<3,3>(NAV_R_IDX, ups_R_idx) = Matrix3::Identity();
        Selector.block<3,3>(NAV_P_IDX, ups_p_idx) = Matrix3::Identity();
        Selector.block<3,3>(NAV_V_IDX, ups_v_idx) = Matrix3::Identity();

        *H = Selector * J_Upsilon_bias * Mapper;

        // DEBUG - Print Jacobian info
        std::cout << "  Mapper norm: " << Mapper.norm() << std::endl;
        std::cout << "  Selector norm: " << Selector.norm() << std::endl;
        std::cout << "  H Jacobian norm: " << H->norm() << std::endl;
        std::cout << "  H Jacobian first few values: " << H->block<1,6>(0,0) << std::endl;

        // Check for NaN/Inf in H
        if (!H->allFinite()) {
             std::cerr << "ERROR: NaN/Inf computed for H!" << std::endl;
        }
    }

    return result;
}


// preintegratedNavStateCovariance
Matrix9 PreintegratedGalileanMeasurements::preintegratedNavStateCovariance() const {
    Matrix9 cov9x9;
    cov9x9.block<3,3>(NAV_R_IDX, NAV_R_IDX) = preintMeasCov_.block<3,3>(ups_R_idx, ups_R_idx);
    cov9x9.block<3,3>(NAV_P_IDX, NAV_P_IDX) = preintMeasCov_.block<3,3>(ups_p_idx, ups_p_idx);
    cov9x9.block<3,3>(NAV_V_IDX, NAV_V_IDX) = preintMeasCov_.block<3,3>(ups_v_idx, ups_v_idx);
    cov9x9.block<3,3>(NAV_R_IDX, NAV_P_IDX) = preintMeasCov_.block<3,3>(ups_R_idx, ups_p_idx);
    cov9x9.block<3,3>(NAV_P_IDX, NAV_R_IDX) = preintMeasCov_.block<3,3>(ups_p_idx, ups_R_idx);
    cov9x9.block<3,3>(NAV_R_IDX, NAV_V_IDX) = preintMeasCov_.block<3,3>(ups_R_idx, ups_v_idx);
    cov9x9.block<3,3>(NAV_V_IDX, NAV_R_IDX) = preintMeasCov_.block<3,3>(ups_v_idx, ups_R_idx);
    cov9x9.block<3,3>(NAV_P_IDX, NAV_V_IDX) = preintMeasCov_.block<3,3>(ups_p_idx, ups_v_idx);
    cov9x9.block<3,3>(NAV_V_IDX, NAV_P_IDX) = preintMeasCov_.block<3,3>(ups_v_idx, ups_p_idx);
    return cov9x9;
}


// computeErrorAndJacobians - Calculates 9D error for factor.
Vector9 PreintegratedGalileanMeasurements::computeErrorAndJacobians(
    const Pose3& pose_i, const Vector3& vel_i,
    const Pose3& pose_j, const Vector3& vel_j,
    const imuBias::ConstantBias& bias_i,
    boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
    boost::optional<Matrix&> H3, boost::optional<Matrix&> H4,
    boost::optional<Matrix&> H5) const {

    // DEBUG - Print inputs
    std::cout << "DEBUG [computeErrorAndJacobians]:" << std::endl;
    std::cout << "  deltaT: " << deltaTij() << std::endl;
    std::cout << "  pose_i: " << pose_i.translation().transpose() << std::endl;
    std::cout << "  vel_i: " << vel_i.transpose() << std::endl;
    std::cout << "  pose_j: " << pose_j.translation().transpose() << std::endl;
    std::cout << "  vel_j: " << vel_j.transpose() << std::endl;
    std::cout << "  bias_i: " << bias_i.vector().transpose() << std::endl;

    // --- 1. Calculate bias-corrected PIM mean ---
    const Gal3& Upsilon_nominal = deltaUpsilon_;
    std::cout << "  Upsilon_nominal:" << std::endl;
    Upsilon_nominal.print("    ");

    Matrix96 H_biasCorrected_bias;
    Vector9 delta_navstate = biasCorrectedDelta(bias_i, H_biasCorrected_bias);
    std::cout << "  delta_navstate: " << delta_navstate.transpose() << std::endl;

    // --- 2. Extract nominal delta measurements ---
    const Rot3& deltaR = Upsilon_nominal.rotation();
    const Vector3 deltaP = Upsilon_nominal.position();
    const Velocity3& deltaV = Upsilon_nominal.velocity();

    std::cout << "  deltaR Logmap: " << Rot3::Logmap(deltaR).transpose() << std::endl;
    std::cout << "  deltaP: " << deltaP.transpose() << std::endl;
    std::cout << "  deltaV: " << deltaV.transpose() << std::endl;

    // --- 3. Calculate predicted state change from states i and j ---
    auto params = galileanParams();
    const Vector3& n_gravity_w = params->n_gravity;
    double deltaT = deltaTij();
    const Rot3& R_i = pose_i.rotation();
    const Vector3 p_i = pose_i.translation();
    const Rot3& R_j = pose_j.rotation();
    const Vector3 p_j = pose_j.translation();

    std::cout << "  gravity: " << n_gravity_w.transpose() << std::endl;

    Rot3 deltaR_pred = R_i.between(R_j);
    Vector3 v_err_w = vel_j - vel_i - n_gravity_w * deltaT;
    Vector3 deltaV_pred = R_i.unrotate(v_err_w);
    Vector3 p_err_w = p_j - p_i - vel_i * deltaT - 0.5 * n_gravity_w * deltaT * deltaT;
    Vector3 deltaP_pred = R_i.unrotate(p_err_w);

    std::cout << "  deltaR_pred Logmap: " << Rot3::Logmap(deltaR_pred).transpose() << std::endl;
    std::cout << "  deltaP_pred: " << deltaP_pred.transpose() << std::endl;
    std::cout << "  deltaV_pred: " << deltaV_pred.transpose() << std::endl;

    // --- 4. Calculate nominal error components (before bias correction) ---
    Vector3 error_R_nominal = Rot3::Logmap(deltaR_pred * deltaR.inverse());
    Vector3 error_p_nominal = deltaP_pred - deltaP;
    Vector3 error_v_nominal = deltaV_pred - deltaV;

    std::cout << "  error_R_nominal: " << error_R_nominal.transpose() << std::endl;
    std::cout << "  error_p_nominal: " << error_p_nominal.transpose() << std::endl;
    std::cout << "  error_v_nominal: " << error_v_nominal.transpose() << std::endl;

    // --- 5. Assemble 9D nominal error and apply bias correction ---
    Vector9 error9D_nominal;
    error9D_nominal.segment<3>(NAV_R_IDX) = error_R_nominal;
    error9D_nominal.segment<3>(NAV_P_IDX) = error_p_nominal;
    error9D_nominal.segment<3>(NAV_V_IDX) = error_v_nominal;

    // Apply first-order bias correction to the error vector
    // error = nominal_error - biasCorrectedDelta
    Vector9 error9D = error9D_nominal - delta_navstate;

    std::cout << "  error9D_nominal: " << error9D_nominal.transpose() << std::endl;
    std::cout << "  error9D (after bias corr): " << error9D.transpose() << std::endl;
    std::cout << "  error9D norm: " << error9D.norm() << std::endl;

    // --- 6. Compute Jacobians (Optional) ---
    if (H1 || H2 || H3 || H4 || H5) {
        // Define the error function *without* bias correction for numerical derivatives
        auto compute_nominal_error_for_jacobian =
            [&](const Pose3& p_i_arg, const Vector3& v_i_arg,
                const Pose3& p_j_arg, const Vector3& v_j_arg,
                const imuBias::ConstantBias& b_i_arg /* unused */) -> Vector9 {

            // --- Use nominal PIM components ---
            const Rot3& deltaR_num = deltaR;
            const Vector3 deltaP_num = deltaP;
            const Velocity3& deltaV_num = deltaV;

            // --- Recalculate predicted state change components ---
            const Rot3& R_i_num = p_i_arg.rotation();
            const Vector3 p_i_num = p_i_arg.translation();
            const Rot3& R_j_num = p_j_arg.rotation();
            const Vector3 p_j_num = p_j_arg.translation();
            double deltaT_num = deltaT;
            const Vector3& n_gravity_w_num = n_gravity_w;

            Rot3 deltaR_pred_num = R_i_num.between(R_j_num);
            Vector3 v_err_w_num = v_j_arg - v_i_arg - n_gravity_w_num * deltaT_num;
            Vector3 deltaV_pred_num = R_i_num.unrotate(v_err_w_num);
            Vector3 p_err_w_num = p_j_num - p_i_num - v_i_arg * deltaT_num - 0.5 * n_gravity_w_num * deltaT_num * deltaT_num;
            Vector3 deltaP_pred_num = R_i_num.unrotate(p_err_w_num);

            // --- Calculate error components ---
            Vector3 error_R_num = Rot3::Logmap(deltaR_pred_num * deltaR_num.inverse());
            Vector3 error_p_num = deltaP_pred_num - deltaP_num;
            Vector3 error_v_num = deltaV_pred_num - deltaV_num;

            // --- Assemble 9D error ---
            Vector9 error9D_num;
            error9D_num.segment<3>(NAV_R_IDX) = error_R_num;
            error9D_num.segment<3>(NAV_P_IDX) = error_p_num;
            error9D_num.segment<3>(NAV_V_IDX) = error_v_num;

            return error9D_num;
        };

        // Calculate Jacobians using numerical differentiation for H1, H2, H3, H4
        double numerical_step = 1e-7;
        if (H1) {
            std::cout << "  Computing H1 (wrt pose_i)..." << std::endl;
            *H1 = numericalDerivative11<Vector9, Pose3>(
                std::bind(compute_nominal_error_for_jacobian, std::placeholders::_1, vel_i, pose_j, vel_j, bias_i),
                pose_i, numerical_step);
            std::cout << "  H1 norm: " << H1->norm() << std::endl;
            std::cout << "  H1 first few values: " << H1->block<1,6>(0,0) << std::endl;
        }
        if (H2) {
            std::cout << "  Computing H2 (wrt vel_i)..." << std::endl;
            *H2 = numericalDerivative11<Vector9, Vector3>(
                std::bind(compute_nominal_error_for_jacobian, pose_i, std::placeholders::_1, pose_j, vel_j, bias_i),
                vel_i, numerical_step);
            std::cout << "  H2 norm: " << H2->norm() << std::endl;
            std::cout << "  H2 first few values: " << H2->block<1,3>(0,0) << std::endl;
        }
        if (H3) {
            std::cout << "  Computing H3 (wrt pose_j)..." << std::endl;
            *H3 = numericalDerivative11<Vector9, Pose3>(
                std::bind(compute_nominal_error_for_jacobian, pose_i, vel_i, std::placeholders::_1, vel_j, bias_i),
                pose_j, numerical_step);
            std::cout << "  H3 norm: " << H3->norm() << std::endl;
            std::cout << "  H3 first few values: " << H3->block<1,6>(0,0) << std::endl;
        }
        if (H4) {
            std::cout << "  Computing H4 (wrt vel_j)..." << std::endl;
            *H4 = numericalDerivative11<Vector9, Vector3>(
                std::bind(compute_nominal_error_for_jacobian, pose_i, vel_i, pose_j, std::placeholders::_1, bias_i),
                vel_j, numerical_step);
            std::cout << "  H4 norm: " << H4->norm() << std::endl;
            std::cout << "  H4 first few values: " << H4->block<1,3>(0,0) << std::endl;
        }
        if (H5) {
            std::cout << "  Computing H5 (wrt bias_i)..." << std::endl;
            *H5 = -H_biasCorrected_bias;
            std::cout << "  H5 norm: " << H5->norm() << std::endl;
            std::cout << "  H5 first few values: " << H5->block<1,6>(0,0) << std::endl;

            if (!H5->allFinite()) {
                std::cerr << "ERROR: NaN/Inf computed for H5 (bias Jacobian) using analytical method!" << std::endl;
            }
        }
    }

    return error9D;
}


// predict - Using standard IMU propagation
NavState PreintegratedGalileanMeasurements::predict(const NavState& state_i,
    const imuBias::ConstantBias& bias_i, OptionalJacobian<9, 9> H1,
    OptionalJacobian<9, 6> H2) const {

    // --- 1. Calculate bias-corrected PIM mean using first order approximation ---
    Matrix96 H_biasCorr_bias; // Jacobian of correction wrt bias
    Vector9 delta_navstate = biasCorrectedDelta(bias_i, H_biasCorr_bias);

    // Apply correction using NavState::retract
    NavState nominal_navstate(deltaUpsilon_.rotation(), deltaUpsilon_.position(), deltaUpsilon_.velocity());
    NavState corrected_navstate = nominal_navstate.retract(delta_navstate);

    // --- 2. Extract corrected delta measurements ---
    const Rot3& deltaR = corrected_navstate.attitude();
    const Vector3 deltaP = corrected_navstate.position();
    const Velocity3& deltaV = corrected_navstate.velocity();

    // --- 3. Get initial state and parameters ---
    const Pose3& pose_i = state_i.pose();
    const Rot3& R_i = pose_i.rotation();
    const Point3& p_i_pt = pose_i.translation();
    const Vector3& vel_i = state_i.velocity();

    auto params = galileanParams();
    const Vector3& n_gravity_w = params->n_gravity;
    double deltaT = deltaTij();
    double deltaT2 = deltaT * deltaT;

    // --- 4. Apply standard IMU propagation ---
    Rot3 R_j = R_i * deltaR;
    Vector3 vel_j = vel_i + n_gravity_w * deltaT + R_i * deltaV;
    Point3 p_j = p_i_pt + Point3(vel_i * deltaT + 0.5 * n_gravity_w * deltaT2 + R_i * deltaP);

    // --- 5. Compute Jacobians (Optional - Placeholders/Numerical for now) ---
    // Analytical Jacobians for standard IMU propagation are known but complex.
    // See PreintegrationBase.cpp or ImuFactor.cpp for examples if needed.
    // Setting placeholders as they are complex and not the core focus here.
    if (H1) {
        // Placeholder: Jacobian of predict wrt state_i (NavState)
        *H1 = Matrix9::Identity(); // Incorrect, just a placeholder
    }
    if (H2) {
         // Placeholder: Jacobian of predict wrt bias_i
         // This involves the chain rule through the bias correction: d(predict)/d(bias) = d(predict)/d(delta) * d(delta)/d(bias)
         // d(predict)/d(delta) is complex. d(delta)/d(bias) is H_biasCorr_bias.
        *H2 = Matrix96::Zero(); // Incorrect, just a placeholder
    }

    // --- 6. Return predicted state ---
    return NavState(Pose3(R_j, p_j), vel_j);
}


} // namespace gtsam
