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

#include <stdexcept>
#include <functional>
#include <limits> // For NaN/Inf checks
#include <iomanip> // For std::fixed, std::setprecision, std::scientific

// Define this macro to switch between numerical and (future) analytical Jacobians
// #define GTSAM_GALILEAN_USE_ANALYTICAL_JACOBIANS 0 // Set to 1 when implemented

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


// Constructor
PreintegratedGalileanMeasurements::PreintegratedGalileanMeasurements(
    const std::shared_ptr<Params>& p, const Bias& biasHat)
    : Base(p, biasHat), // Call base class constructor
      deltaUpsilon_(Gal3::Identity()),
      preintMeasCov_(Matrix20::Zero()), // Initialized according to header declaration order
      preintBiasJacobian_(Matrix20::Identity()), // Initialized according to header declaration order
      integration_step_counter_(0) // Initialized according to header declaration order
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
  integration_step_counter_ = 0; // Reset step counter

  // Initialize bias covariance based on initial uncertainty from params
  if (p_) {
    auto galileanParamsPtr = std::dynamic_pointer_cast<const Params>(p_);
    if (galileanParamsPtr) {
        Matrix6 biasAccOmegaCov = galileanParamsPtr->getBiasAccOmegaInit();

        // DEBUG PRINT: Initial bias covariance from params
        std::cout << std::fixed << std::setprecision(12);
        std::cout << "--- DEBUG: resetIntegration ---" << std::endl;
        std::cout << "Initial biasAccOmegaCov (from params):\n" << biasAccOmegaCov << std::endl;

        // Map the 6x6 initial bias covariance into the bias block of 20x20 Sigma_0
        // Order: [rho, nu, theta, t | b_omega, b_acc, b_nu, b_rho]
        preintMeasCov_.block<3,3>(bias_w_idx, bias_w_idx) = biasAccOmegaCov.block<3,3>(3,3); // Gyro bias cov
        preintMeasCov_.block<3,3>(bias_a_idx, bias_a_idx) = biasAccOmegaCov.block<3,3>(0,0); // Accel bias cov
        // Initial covariance for virtual biases b_nu, b_rho is zero
        preintMeasCov_.block<3,3>(bias_nu_idx, bias_nu_idx).setZero();
        preintMeasCov_(bias_rho_idx, bias_rho_idx) = 0.0;

        // DEBUG PRINT: Initialized bias block of preintMeasCov_
        std::cout << "Initialized preintMeasCov_ (Bias Block Diagonal " << bias_dim << "x" << bias_dim << "):\n"
                  << preintMeasCov_.block<bias_dim,bias_dim>(ups_dim, ups_dim).diagonal().transpose() << std::endl;
        std::cout << "--- END DEBUG: resetIntegration ---" << std::endl;

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
  // deltaUpsilon_.print("    "); // Removed print statement
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
    Vector10 result = Vector10::Zero();
    // DEBUG OUTPUT
    // std::cout << "--- DEBUG: mapBias6ToTangent10 ---" << std::endl;
    // std::cout << "Input bias6D: " << bias6D.transpose() << std::endl;

    // Map the biases according to the paper - check if this mapping is correct
    result.segment<3>(bias_w_comp_idx) = bias6D.tail<3>();  // Gyro bias
    result.segment<3>(bias_a_comp_idx) = bias6D.head<3>();  // Accel bias

    // std::cout << "Output bias10D: " << result.transpose() << std::endl;
    return result;
}


// mapMeasurement10ToTangent10
Vector10 PreintegratedGalileanMeasurements::mapMeasurement10ToTangent10(const Vector10& measurement10D) {
    Vector10 tangent10D = Vector10::Zero();
    tangent10D.segment<3>(ups_v_idx) = measurement10D.segment<3>(bias_a_comp_idx); // acc measurement -> nu component of Upsilon tangent
    tangent10D.segment<3>(ups_R_idx) = measurement10D.segment<3>(bias_w_comp_idx); // omega measurement -> theta component of Upsilon tangent
    tangent10D(ups_t_idx)            = measurement10D(bias_rho_comp_idx);       // rho measurement (time=1) -> t component of Upsilon tangent
    return tangent10D;
}



void calculateQ1OmegaR(const Vector3& omega, const Vector3& r, Matrix3& result) {
    double omega_norm = omega.norm();

    if (omega_norm < 1e-8) {
        // Near-zero omega case - use first term of series expansion
        result = 0.5 * skewSymmetric(r);
        return;
    }

    // Calculate terms from the closed-form expression
    double term1 = (omega_norm - sin(omega_norm)) / pow(omega_norm, 3);
    double term2 = (omega_norm * omega_norm + 2.0 * cos(omega_norm) - 2.0) / (2.0 * pow(omega_norm, 4));
    double term3 = (2.0 * omega_norm - 3.0 * sin(omega_norm) + omega_norm * cos(omega_norm)) / (2.0 * pow(omega_norm, 5));

    Matrix3 omega_hat = skewSymmetric(omega);
    Matrix3 r_hat = skewSymmetric(r);

    // Compute the full expression from Appendix A
    result = 0.5 * r_hat +
             term1 * (omega_hat * r_hat + r_hat * omega_hat + omega_hat * r_hat * omega_hat) +
             term2 * (omega_hat * omega_hat * r_hat + r_hat * omega_hat * omega_hat - 3.0 * omega_hat * r_hat * omega_hat) +
             term3 * (omega_hat * r_hat * omega_hat * omega_hat + omega_hat * omega_hat * r_hat * omega_hat);
}

void calculateQ2OmegaV(const Vector3& omega, const Vector3& v, Matrix3& result) {
    double omega_norm = omega.norm();

    if (omega_norm < 1e-8) {
        // Near-zero omega case - use first term of series expansion
        result = (1.0/6.0) * skewSymmetric(v);
        return;
    }

    Matrix3 omega_hat = skewSymmetric(omega);
    Matrix3 v_hat = skewSymmetric(v);

    // Calculate all terms from the closed-form expression in Appendix A
    double term1 = (omega_norm * omega_norm + 2.0 * cos(omega_norm) - 2.0) / (2.0 * pow(omega_norm, 4));
    double term2 = (omega_norm * omega_norm * omega_norm - 6.0 * omega_norm + 6.0 * sin(omega_norm)) / (6.0 * pow(omega_norm, 5));
    double term3 = (-2.0 * cos(omega_norm) - omega_norm * sin(omega_norm) + 2.0) / pow(omega_norm, 4);
    double term4 = (omega_norm * omega_norm * omega_norm + 6.0 * omega_norm * cos(omega_norm) + 6.0 * omega_norm - 12.0 * sin(omega_norm)) / (6.0 * pow(omega_norm, 5));
    double term5 = (-3.0 * omega_norm * cos(omega_norm) - (omega_norm * omega_norm - 3.0) * sin(omega_norm)) / (4.0 * pow(omega_norm, 5));
    double term6 = (omega_norm * cos(omega_norm) + 2.0 * omega_norm - 3.0 * sin(omega_norm)) / (4.0 * pow(omega_norm, 5));
    double term7 = ((omega_norm * omega_norm - 8.0) * cos(omega_norm) - 5.0 * omega_norm * sin(omega_norm) + 8.0) / (4.0 * pow(omega_norm, 6));
    double term8 = (2.0 * omega_norm * omega_norm * omega_norm + 15.0 * omega_norm * cos(omega_norm) + 3.0 * (omega_norm * omega_norm - 5.0) * sin(omega_norm)) / (12.0 * pow(omega_norm, 7));

    // Compute the full expression
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
        // Near-zero omega case - use first term of series expansion
        result = 0.5 * Matrix3::Identity();
        return;
    }

    // Calculate terms from the closed-form expression
    double term1 = (sin(omega_norm) - omega_norm * cos(omega_norm)) / pow(omega_norm, 3);
    double term2 = (omega_norm * omega_norm - 2.0 * omega_norm * sin(omega_norm) - 2.0 * cos(omega_norm) + 2.0) / (2.0 * pow(omega_norm, 4));

    Matrix3 omega_hat = skewSymmetric(omega);

    // Compute the full expression from Appendix A
    result = 0.5 * Matrix3::Identity() + term1 * omega_hat + term2 * omega_hat * omega_hat;

}


void calculateQ1OmegaV(const Vector3& omega, const Vector3& z, Matrix3& result) {
    double omega_norm = omega.norm();
    if (omega_norm < 1e-8) {
        result = 0.5 * skewSymmetric(z);
        return;
    }

    // Calculate the terms according to the closed-form expression in Appendix A
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


// integrateMeasurement - Algorithm 1 Core
void PreintegratedGalileanMeasurements::integrateMeasurement(
    const Vector3& measuredAcc, const Vector3& measuredOmega, double dt) {

    const size_t ups_v_idx = 3; // nu component (velocity)
    const size_t ups_R_idx = 6; // theta component (rotation)

    if (dt <= 0) {
        return; // Skip integration for non-positive dt
    }
    integration_step_counter_++; // Increment step counter

    auto params = galileanParams();
    Vector3 bodyAcc = measuredAcc;
    Vector3 bodyOmega = measuredOmega;

    const Gal3& Upsilon_hat_k = deltaUpsilon_; // Current PIM mean estimate
    Vector10 b_hat_k = mapBias6ToTangent10(biasHat_.vector()); // Current bias estimate in 10D tangent space form

    // Construct w_tilde_k: raw measurements + virtual inputs in 10D tangent space form
    Vector10 w_tilde_k = Vector10::Zero();
    w_tilde_k.segment<3>(bias_w_comp_idx) = bodyOmega;     // omega measurement
    w_tilde_k.segment<3>(bias_a_comp_idx) = bodyAcc;       // accel measurement
    w_tilde_k(bias_rho_comp_idx) = 1.0;                    // virtual time input rho_k = 1

    // Construct Qd: noise covariance matrix for the 20D error propagation
    Matrix20 Q_d = Matrix20::Zero();
    // double inv_dt = 1.0 / dt;

    // Measurement noise part (eta_wk): affects Upsilon components
    // Q_d.block<3,3>(ups_R_idx, ups_R_idx) = params->gyroscopeCovariance * inv_dt;
    // Q_d.block<3,3>(ups_v_idx, ups_v_idx) = params->accelerometerCovariance * inv_dt;
    Q_d.block<3,3>(ups_R_idx, ups_R_idx) = params->gyroscopeCovariance * dt;
    Q_d.block<3,3>(ups_v_idx, ups_v_idx) = params->accelerometerCovariance * dt;

    // Bias random walk part (eta_tau_k): affects Bias components directly
    Q_d.block<3,3>(bias_w_idx, bias_w_idx) = params->getBiasOmegaCovariance() * dt;
    Q_d.block<3,3>(bias_a_idx, bias_a_idx) = params->getBiasAccCovariance() * dt;
    // std::cout << "--- DEBUG biasCorrectedDelta 99999 ---" << std::endl;

    // Corrected measurement (w_k - b_k) in 10D tangent space form
    Vector10 w_hat_k = w_tilde_k - b_hat_k;

    // Map the bias-corrected measurement to Upsilon tangent space
    Vector10 zeta_k = mapMeasurement10ToTangent10(w_hat_k) * dt;

    // Mean propagation for Upsilon (PIM)
    Gal3 Exp_zeta_k = Gal3::Expmap(zeta_k);
    deltaUpsilon_ = Upsilon_hat_k * Exp_zeta_k;
    deltaTij_ += dt; // Update total integration time
    bool is_zero_motion = (measuredAcc.norm() < 1e-10 && measuredOmega.norm() < 1e-10);

    // Jacobians for covariance propagation
    Matrix10 Ad_Upsilon_hat_k = Upsilon_hat_k.AdjointMap();
    Matrix10 Ad_Upsilon_hat_inv_k = Upsilon_hat_k.inverse().AdjointMap();

    Vector10 w_bar_k = Ad_Upsilon_hat_inv_k * w_hat_k;
    Vector10 mapped_w_bar_k = mapMeasurement10ToTangent10(w_bar_k);
    Vector10 exp_arg_for_A_coupling = mapped_w_bar_k * dt;
    Matrix10 JL_exp_arg_for_A_coupling = Gal3::ExpmapDerivative(exp_arg_for_A_coupling);

    Matrix10 JL_zeta_k = Gal3::ExpmapDerivative(zeta_k);

    Gal3 Exp_w_bar_k_dt = Gal3::Expmap(exp_arg_for_A_coupling);
    Matrix10 Ad_Exp_w_bar_k_dt = Exp_w_bar_k_dt.AdjointMap();

    // Construct A_hat_k1 (Eq. 36)
    Matrix20 A_hat_k1 = Matrix20::Identity();
    if (!is_zero_motion) {
        A_hat_k1.block<10, 10>(0, 10) = JL_exp_arg_for_A_coupling * dt;
    }
    A_hat_k1.block<10, 10>(10, 10) = Ad_Exp_w_bar_k_dt;

    // Construct B_hat_k1 (Eq. 37)
    Matrix20 B_hat_k1 = Matrix20::Zero();
    B_hat_k1.block<10, 10>(0, 0) = -Ad_Upsilon_hat_k * JL_zeta_k * dt;
    B_hat_k1.block<10, 10>(10, 10) = deltaUpsilon_.AdjointMap() * dt;

    // Update covariance
    Matrix20 term1 = A_hat_k1 * preintMeasCov_ * A_hat_k1.transpose();
    Matrix20 term2 = B_hat_k1 * Q_d * B_hat_k1.transpose();
    preintMeasCov_ = term1 + term2;
    preintMeasCov_ = (preintMeasCov_ + preintMeasCov_.transpose()) / 2.0; // Ensure symmetry

    // FIXED: Properly create the bias Jacobian with the correct coupling
    Matrix20 Phi_b_k1 = Matrix20::Identity();

    Vector10 w_hat_bar_dt = Ad_Upsilon_hat_inv_k * w_hat_k * dt;
    Vector3 omega_dt = w_hat_bar_dt.segment<3>(0);
    Vector3 v_dt = w_hat_bar_dt.segment<3>(3);
    Vector3 r_dt = w_hat_bar_dt.segment<3>(6);
    double alpha_dt = w_hat_bar_dt(9);

    // Calculate the necessary components for the Jacobian
    double omega_norm = omega_dt.norm();
    Matrix3 Gamma1, Gamma2;  // SO(3) Jacobians
    Matrix3 Q1_omega_v, Q1_omega_r, Q2_omega_v, U1_omega;

    // Calculate Gamma1 (SO(3) left Jacobian)
    if (omega_norm < 1e-8) {
        Gamma1 = Matrix3::Identity();
    } else {
        double k1 = (1.0 - cos(omega_norm)) / (omega_norm * omega_norm);
        double k2 = (omega_norm - sin(omega_norm)) / (omega_norm * omega_norm * omega_norm);
        Gamma1 = Matrix3::Identity() + k1 * skewSymmetric(omega_dt) + k2 * skewSymmetric(omega_dt) * skewSymmetric(omega_dt);
    }

    // Calculate Gamma2 (SO(3) auxiliary function)
    if (omega_norm < 1e-8) {
        Gamma2 = Matrix3::Identity() * 0.5;
    } else {
        double k2 = (omega_norm - sin(omega_norm)) / (omega_norm * omega_norm * omega_norm);
        double k3 = (omega_norm * omega_norm + 2.0 * cos(omega_norm) - 2.0) / (2.0 * pow(omega_norm, 4));
        Gamma2 = Matrix3::Identity() * 0.5 + k2 * skewSymmetric(omega_dt) + k3 * skewSymmetric(omega_dt) * skewSymmetric(omega_dt);
    }

    // Calculate the remaining components (use the closed-form expressions from Appendix A)
    calculateQ1OmegaV(omega_dt, v_dt, Q1_omega_v);  // Implement using the formula in Appendix A
    calculateQ1OmegaR(omega_dt, r_dt, Q1_omega_r);  // Implement using the formula in Appendix A
    calculateQ2OmegaV(omega_dt, v_dt, Q2_omega_v);  // Implement using the formula in Appendix A
    calculateU1Omega(omega_dt, U1_omega);           // Implement using the formula in Appendix A

    // Construct the full Gal(3) left Jacobian matrix JL
    Matrix10 JL_gal3 = Matrix10::Zero();
    JL_gal3.block<3,3>(0,0) = Gamma1;
    JL_gal3.block<3,3>(3,0) = Q1_omega_v;
    JL_gal3.block<3,3>(3,3) = Gamma1;
    JL_gal3.block<3,3>(6,0) = Q1_omega_r - alpha_dt * Q2_omega_v;
    JL_gal3.block<3,3>(6,3) = -alpha_dt * U1_omega;
    JL_gal3.block<3,3>(6,6) = Gamma1;
    JL_gal3.block<3,1>(6,9) = Gamma2 * v_dt;
    JL_gal3(9,9) = 1.0;

    // Now use the proper Jacobian for bias updates instead of the manual coupling
    if (!is_zero_motion) {
        // Create direct coupling from bias space to rotation/velocity in Upsilon space
        Matrix10 J_bias_update_for_PIM = -Ad_Upsilon_hat_k * JL_zeta_k * dt;
        Phi_b_k1.block<10, 10>(0, 10) = J_bias_update_for_PIM;
    } else {
       // Explicitly zero out the coupling block for zero motion case
       Phi_b_k1.block<10, 10>(0, 10).setZero();
   }

    preintBiasJacobian_ = Phi_b_k1 * preintBiasJacobian_;
}


Vector9 PreintegratedGalileanMeasurements::biasCorrectedDelta(
    const imuBias::ConstantBias& bias_i, // The "new" or "correct" bias
    OptionalJacobian<9, 6> H) const {   // Jacobian of the 9D correction wrt the 6D bias_i

    // --- BEGIN DEBUG PRINTS for biasCorrectedDelta ---
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "--- DEBUG biasCorrectedDelta ---" << std::endl;
    std::cout << "bias_i (new): " << bias_i.vector().transpose() << std::endl;
    std::cout << "biasHat_ (pim): " << biasHat_.vector().transpose() << std::endl;
    // --- END DEBUG PRINTS ---

    // Step 1: Calculate the bias difference (new_bias - preintegrated_bias) in 6D, then map to 10D tangent space.
    Vector6 delta_bias_6D = bias_i.vector() - biasHat_.vector(); // biasHat_ is the bias used for preintegration
    Vector10 delta_b_10D = mapBias6ToTangent10(delta_bias_6D);   // (b_i - b_hat) in 10D tangent space
    // Add at beginning of biasCorrectedDelta
    std::cout << "Delta bias tangent space representation:\n";
    std::cout << "Gyro part: " << delta_b_10D.segment<3>(bias_w_comp_idx).transpose() << std::endl;
    std::cout << "Accel part: " << delta_b_10D.segment<3>(bias_a_comp_idx).transpose() << std::endl;

    // Step 2: Use preintBiasJacobian_ directly as per Eq. 39 in the paper
    // Get the correction in 10D tangent space
    Vector10 bias_induced_correction_tangent = preintBiasJacobian_.block<10,10>(0, ups_dim) * delta_b_10D;
    std::cout << "Correction tangent space components:\n";
    std::cout << "Rotation correction: " << bias_induced_correction_tangent.segment<3>(ups_R_idx).transpose() << std::endl;
    std::cout << "Velocity correction: " << bias_induced_correction_tangent.segment<3>(ups_v_idx).transpose() << std::endl;

    // Step 3: Apply the correction using Lie group operation
    Gal3 corrected_deltaUpsilon = Gal3::Expmap(bias_induced_correction_tangent) * deltaUpsilon_;

    // Step 4: Convert to 9D NavState correction
    Vector9 result_navstate_correction_9D;

    // Extract differences between corrected and original state
    // For rotation, we need to compute the correction in tangent space
    Rot3 orig_R = deltaUpsilon_.rotation();
    Rot3 corr_R = corrected_deltaUpsilon.rotation();
    result_navstate_correction_9D.segment<3>(NAV_R_IDX) = Rot3::Logmap(orig_R.between(corr_R));

    // For position and velocity, we can compute the direct difference
    result_navstate_correction_9D.segment<3>(NAV_P_IDX) = corrected_deltaUpsilon.position() - deltaUpsilon_.position();
    result_navstate_correction_9D.segment<3>(NAV_V_IDX) = corrected_deltaUpsilon.velocity() - deltaUpsilon_.velocity();

    // Handle Jacobian calculation if requested: d(result_navstate_correction_9D) / d(bias_i)
    // Todo: replace with analytical derivative
    if (H) {
        // Use a lambda instead of std::bind for numerical differentiation
        auto bias_correction_func = [this](const imuBias::ConstantBias& b) -> Vector9 {
            // Pass OptionalJacobian<9, 6>{} instead of boost::none
            return this->biasCorrectedDelta(b, OptionalJacobian<9, 6>{});
        };

        *H = numericalDerivative11<Vector9, imuBias::ConstantBias>(
            bias_correction_func, bias_i);
    }

    return result_navstate_correction_9D;
}





// preintegratedNavStateCovariance
// Matrix9 PreintegratedGalileanMeasurements::preintegratedNavStateCovariance() const {
//     // Extracts the 9x9 NavState block from the 10x10 Upsilon block of the 20x20 covariance
//     Matrix9 cov9x9 = Matrix9::Zero();
//     // Diagonal blocks
//     cov9x9.block<3,3>(NAV_R_IDX, NAV_R_IDX) = preintMeasCov_.block<3,3>(ups_R_idx, ups_R_idx); // Rot-Rot
//     cov9x9.block<3,3>(NAV_P_IDX, NAV_P_IDX) = preintMeasCov_.block<3,3>(ups_p_idx, ups_p_idx); // Pos-Pos
//     cov9x9.block<3,3>(NAV_V_IDX, NAV_V_IDX) = preintMeasCov_.block<3,3>(ups_v_idx, ups_v_idx); // Vel-Vel
//     // Off-diagonal blocks
//     cov9x9.block<3,3>(NAV_R_IDX, NAV_P_IDX) = preintMeasCov_.block<3,3>(ups_R_idx, ups_p_idx); // Rot-Pos
//     cov9x9.block<3,3>(NAV_P_IDX, NAV_R_IDX) = preintMeasCov_.block<3,3>(ups_p_idx, ups_R_idx); // Pos-Rot
//     cov9x9.block<3,3>(NAV_R_IDX, NAV_V_IDX) = preintMeasCov_.block<3,3>(ups_R_idx, ups_v_idx); // Rot-Vel
//     cov9x9.block<3,3>(NAV_V_IDX, NAV_R_IDX) = preintMeasCov_.block<3,3>(ups_v_idx, ups_R_idx); // Vel-Rot
//     cov9x9.block<3,3>(NAV_P_IDX, NAV_V_IDX) = preintMeasCov_.block<3,3>(ups_p_idx, ups_v_idx); // Pos-Vel
//     cov9x9.block<3,3>(NAV_V_IDX, NAV_P_IDX) = preintMeasCov_.block<3,3>(ups_v_idx, ups_p_idx); // Vel-Pos
//     return cov9x9;
// }

// -----------------------------------------------------------------------------
//  Nav‑state covariance Σ_nav = T_nav  Σ_k  T_navᵀ   (Alg.‑1, Eq. 41)
// -----------------------------------------------------------------------------
Matrix9 PreintegratedGalileanMeasurements::
preintegratedNavStateCovariance() const {

  /* ------------------------------------------------------------------------
   * 1. Selector S ∈ ℝ⁹ˣ¹⁰  – picks [θ, ρ, ν] from the 10‑D Gal‑3 tangent
   *    Order in our code:            [ρ(0‑2)  ν(3‑5)  θ(6‑8)  t(9)]
   *    Desired nav order (SO3×ℝ⁶):   [θ       ρ       ν        ]
   * ---------------------------------------------------------------------- */
  static const Matrix9_10 S = [] {
    Matrix9_10 M = Matrix9_10::Zero();
    M.block<3,3>(0,6) = Matrix3::Identity();   // θ  (rows 0‑2)
    M.block<3,3>(3,0) = Matrix3::Identity();   // ρ  (rows 3‑5)
    M.block<3,3>(6,3) = Matrix3::Identity();   // ν  (rows 6‑8)
    return M;
  }();

  /* ------------------------------------------------------------------------
   * 2. Left‑Jacobian inverse  J_L⁻¹( log(ΔΥ̂) )
   * ---------------------------------------------------------------------- */
  const Matrix10 JL_inv = Gal3::LogmapDerivative(deltaUpsilon_);

  /* ------------------------------------------------------------------------
   * 3. Adjoint of the group element  Ad_{ΔΥ̂}
   * ---------------------------------------------------------------------- */
  const Matrix10 Ad = deltaUpsilon_.AdjointMap();

  /* ------------------------------------------------------------------------
   * 4. Navigation‑space transform  T_nav  ∈ ℝ⁹ˣ²⁰
   *    (first 10 columns → Upsilon tangent, last 10 → bias tangent)
   * ---------------------------------------------------------------------- */
  Eigen::Matrix<double,9,20> T_nav   = Eigen::Matrix<double,9,20>::Zero();
  T_nav.block<9,10>(0,0)   =  S;                     // direct navigation part
  T_nav.block<9,10>(0,10)  = -S * JL_inv * Ad;       // geometric coupling

  /* ------------------------------------------------------------------------
   * 5. Form the 9×9 covariance and force symmetry
   * ---------------------------------------------------------------------- */
  Matrix9 P_nav = T_nav * preintMeasCov_ * T_nav.transpose();
  return 0.5 * (P_nav + P_nav.transpose());
}


// computeErrorAndJacobians - Calculates 9D error for factor.
Vector9 PreintegratedGalileanMeasurements::computeErrorAndJacobians(
    const Pose3& pose_i, const Vector3& vel_i,
    const Pose3& pose_j, const Vector3& vel_j,
    const imuBias::ConstantBias& bias_i, // This is the bias estimate at time i (or current estimate for the interval)
    boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
    boost::optional<Matrix&> H3, boost::optional<Matrix&> H4,
    boost::optional<Matrix&> H5) const {

    // --- 1. Calculate the PIM correction due to difference between bias_i and biasHat_ ---
    Matrix96 H_pim_correction_wrt_bias_diff;
    Vector9 pim_navstate_correction = biasCorrectedDelta(bias_i, H_pim_correction_wrt_bias_diff);

    // --- 2. Get nominal PIM components (preintegrated with biasHat_) ---
    const Rot3& deltaR_nominal = deltaUpsilon_.rotation();
    const Vector3 deltaP_nominal = deltaUpsilon_.position();
    const Velocity3& deltaV_nominal = deltaUpsilon_.velocity();

    // --- 3. Apply correction to nominal PIM components to get "bias_i-corrected" PIM components ---
    // The pim_navstate_correction is already in the tangent space of the nominal NavState components.
    // So, R_corrected = R_nominal * Exp(correction_R_tangent)
    //     P_corrected = P_nominal + correction_P_tangent (since position is vector space)
    //     V_corrected = V_nominal + correction_V_tangent (since velocity is vector space)
    Rot3 deltaR_corrected = deltaR_nominal * Rot3::Expmap(pim_navstate_correction.segment<3>(NAV_R_IDX));
    Vector3 deltaP_corrected = deltaP_nominal + pim_navstate_correction.segment<3>(NAV_P_IDX);
    Vector3 deltaV_corrected = deltaV_nominal + pim_navstate_correction.segment<3>(NAV_V_IDX);

    // --- 4. Calculate predicted state change from states i and j (standard IMU factor math) ---
    auto params = galileanParams();
    const Vector3& n_gravity_w = params->n_gravity; // Gravity in world frame
    double deltaT = deltaTij();
    const Rot3& R_i = pose_i.rotation();    // Orientation of body frame i in world frame
    const Vector3 p_i = pose_i.translation(); // Position of body frame i in world frame
    const Rot3& R_j = pose_j.rotation();    // Orientation of body frame j in world frame
    const Vector3 p_j = pose_j.translation(); // Position of body frame j in world frame

    // Predicted relative rotation: R_i^T * R_j
    Rot3 deltaR_pred = R_i.between(R_j);
    // Predicted relative velocity in frame i: R_i^T * (vel_j - vel_i - g * deltaT)
    Vector3 v_err_w = vel_j - vel_i - n_gravity_w * deltaT;
    Vector3 deltaV_pred = R_i.unrotate(v_err_w);
    // Predicted relative position in frame i: R_i^T * (p_j - p_i - vel_i * deltaT - 0.5 * g * deltaT^2)
    Vector3 p_err_w = p_j - p_i - vel_i * deltaT - 0.5 * n_gravity_w * deltaT * deltaT;
    Vector3 deltaP_pred = R_i.unrotate(p_err_w);

    // --- 5. Calculate error components using the bias_i-corrected PIM values ---
    Vector3 error_R = Rot3::Logmap(deltaR_pred * deltaR_corrected.inverse());
    Vector3 error_p = deltaP_pred - deltaP_corrected;
    Vector3 error_v = deltaV_pred - deltaV_corrected;

    // --- 6. Assemble 9D error vector ---
    Vector9 error9D;
    error9D.segment<3>(NAV_R_IDX) = error_R;
    error9D.segment<3>(NAV_P_IDX) = error_p;
    error9D.segment<3>(NAV_V_IDX) = error_v;

    // --- 7. Compute Jacobians (Optional) ---
    if (H1 || H2 || H3 || H4 || H5) {
        // Define the full error function for numerical differentiation
        auto compute_full_error_for_jacobian =
            [&](const Pose3& current_pose_i, const Vector3& current_vel_i,
                const Pose3& current_pose_j, const Vector3& current_vel_j,
                const imuBias::ConstantBias& current_bias_i) -> Vector9 {

            // Recalculate PIM correction for current_bias_i
            Vector9 current_pim_corr = this->biasCorrectedDelta(current_bias_i);

            Rot3 c_deltaR = deltaR_nominal * Rot3::Expmap(current_pim_corr.segment<3>(NAV_R_IDX));
            Vector3 c_deltaP = deltaP_nominal + current_pim_corr.segment<3>(NAV_P_IDX);
            Vector3 c_deltaV = deltaV_nominal + current_pim_corr.segment<3>(NAV_V_IDX);

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

            Vector3 cError_R = Rot3::Logmap(cDeltaR_pred * c_deltaR.inverse());
            Vector3 cError_p = cDeltaP_pred - c_deltaP;
            Vector3 cError_v = cDeltaV_pred - c_deltaV;

            Vector9 cError9D;
            cError9D.segment<3>(NAV_R_IDX) = cError_R;
            cError9D.segment<3>(NAV_P_IDX) = cError_p;
            cError9D.segment<3>(NAV_V_IDX) = cError_v;
            return cError9D;
        };

        double numerical_step = 1e-7; // Step for numerical differentiation
        if (H1) { // Jacobian w.r.t. pose_i
            *H1 = numericalDerivative11<Vector9, Pose3>(
                std::bind(compute_full_error_for_jacobian, std::placeholders::_1, vel_i, pose_j, vel_j, bias_i),
                pose_i, numerical_step);
        }
        if (H2) { // Jacobian w.r.t. vel_i
            *H2 = numericalDerivative11<Vector9, Vector3>(
                std::bind(compute_full_error_for_jacobian, pose_i, std::placeholders::_1, pose_j, vel_j, bias_i),
                vel_i, numerical_step);
        }
        if (H3) { // Jacobian w.r.t. pose_j
            *H3 = numericalDerivative11<Vector9, Pose3>(
                std::bind(compute_full_error_for_jacobian, pose_i, vel_i, std::placeholders::_1, vel_j, bias_i),
                pose_j, numerical_step);
        }
        if (H4) { // Jacobian w.r.t. vel_j
            *H4 = numericalDerivative11<Vector9, Vector3>(
                std::bind(compute_full_error_for_jacobian, pose_i, vel_i, pose_j, std::placeholders::_1, bias_i),
                vel_j, numerical_step);
        }
        if (H5) { // Jacobian w.r.t. bias_i
            // The error is error_pred(states) - corrected_PIM(bias_i)
            // So, d(error)/d(bias_i) = - d(corrected_PIM)/d(bias_i)
            // d(corrected_PIM)/d(bias_i) was H_pim_correction_wrt_bias_diff (Jacobian of the correction wrt bias_i)
            *H5 = -H_pim_correction_wrt_bias_diff;
        }
    }

    return error9D;
}


// predict - Using standard IMU propagation with bias-corrected PIM
NavState PreintegratedGalileanMeasurements::predict(const NavState& state_i,
    const imuBias::ConstantBias& bias_i, // The bias to use for prediction
    OptionalJacobian<9, 9> H1,      // Jacobian of predicted NavState_j w.r.t. NavState_i
    OptionalJacobian<9, 6> H2) const {   // Jacobian of predicted NavState_j w.r.t. bias_i

    // --- 1. Calculate the PIM correction due to difference between bias_i and biasHat_ ---
    Matrix96 H_pim_correction_wrt_bias_diff; // Not used for predict's H2 directly here, but calculated by biasCorrectedDelta
    Vector9 pim_navstate_correction = biasCorrectedDelta(bias_i, H_pim_correction_wrt_bias_diff);

    // --- 2. Get nominal PIM components (preintegrated with biasHat_) ---
    const Rot3& deltaR_nominal = deltaUpsilon_.rotation();
    const Vector3 deltaP_nominal = deltaUpsilon_.position();
    const Velocity3& deltaV_nominal = deltaUpsilon_.velocity();

    // --- 3. Apply correction to nominal PIM components to get "bias_i-corrected" PIM components ---
    Rot3 deltaR_corrected = deltaR_nominal * Rot3::Expmap(pim_navstate_correction.segment<3>(NAV_R_IDX));
    Vector3 deltaP_corrected = deltaP_nominal + pim_navstate_correction.segment<3>(NAV_P_IDX);
    Vector3 deltaV_corrected = deltaV_nominal + pim_navstate_correction.segment<3>(NAV_V_IDX);

    // --- 4. Get initial state and parameters ---
    const Pose3& pose_i = state_i.pose();
    const Rot3& R_i = pose_i.rotation();
    const Point3& p_i_pt = pose_i.translation();
    const Vector3& vel_i = state_i.velocity();

    auto params = galileanParams();
    const Vector3& n_gravity_w = params->n_gravity; // Gravity in world frame
    double deltaT = deltaTij();
    double deltaT2 = deltaT * deltaT;

    // --- 5. Apply standard IMU propagation using bias_i-corrected PIM deltas ---
    Rot3 R_j = R_i * deltaR_corrected;
    Vector3 vel_j = vel_i + n_gravity_w * deltaT + R_i * deltaV_corrected;
    Point3 p_j = p_i_pt + Point3(vel_i * deltaT + 0.5 * n_gravity_w * deltaT2 + R_i * deltaP_corrected);

    // --- 6. Compute Jacobians (Optional) ---
    // Analytical Jacobians for this are involved due to the bias correction chain rule.
    // Using numerical differentiation for now if Jacobians are requested.
    if (H1 || H2) {
        auto predict_wrapper_for_jacobian =
            [&](const NavState& current_state_i, const imuBias::ConstantBias& current_bias_i) -> NavState {
            Vector9 current_pim_corr = this->biasCorrectedDelta(current_bias_i);
            Rot3 c_deltaR = deltaR_nominal * Rot3::Expmap(current_pim_corr.segment<3>(NAV_R_IDX));
            Vector3 c_deltaP = deltaP_nominal + current_pim_corr.segment<3>(NAV_P_IDX);
            Vector3 c_deltaV = deltaV_nominal + current_pim_corr.segment<3>(NAV_V_IDX);

            const Rot3& cR_i = current_state_i.attitude();
            const Point3& cp_i_pt = current_state_i.position();
            const Vector3& cvel_i = current_state_i.velocity();
            double cDeltaT = this->deltaTij();
            double cDeltaT2 = cDeltaT * cDeltaT;
            const Vector3& cn_gravity_w = this->galileanParams()->n_gravity;

            Rot3 cR_j = cR_i * c_deltaR;
            Vector3 cvel_j = cvel_i + cn_gravity_w * cDeltaT + cR_i * c_deltaV;
            Point3 cp_j = cp_i_pt + Point3(cvel_i * cDeltaT + 0.5 * cn_gravity_w * cDeltaT2 + cR_i * c_deltaP);
            return NavState(Pose3(cR_j, cp_j), cvel_j);
        };

        if (H1) { // Jacobian of predicted NavState_j w.r.t. NavState_i
            *H1 = numericalDerivative11<NavState, NavState>(
                std::bind(predict_wrapper_for_jacobian, std::placeholders::_1, bias_i),
                state_i);
        }
        if (H2) { // Jacobian of predicted NavState_j w.r.t. bias_i
            *H2 = numericalDerivative11<NavState, imuBias::ConstantBias>(
                std::bind(predict_wrapper_for_jacobian, state_i, std::placeholders::_1),
                bias_i);
        }
    }

    // --- 7. Return predicted state ---
    return NavState(Pose3(R_j, p_j), vel_j);
}


} // namespace gtsam
