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


// --- Implementations for PreintegratedGalileanMeasurements ---

// Constructor
PreintegratedGalileanMeasurements::PreintegratedGalileanMeasurements(
    const std::shared_ptr<Params>& p, const Bias& biasHat)
    : Base(p, biasHat),
      deltaUpsilon_(Gal3::Identity()),
      preintMeasCov_(Matrix20::Zero()),
      preintBiasJacobian_(Matrix20::Identity())
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
      && equal_with_abs_tol(preintBiasJacobian_, other.preintBiasJacobian_, tol);
      }

Vector10 PreintegratedGalileanMeasurements::mapBias6ToTangent10(const Vector6& bias6D) {
    // Gyroscope bias (3D, positions 0-2): bias_w_comp_idx
    // Accelerometer bias (3D, positions 3-5): bias_a_comp_idx
    // Virtual velocity bias (3D, positions 6-8): bias_nu_comp_idx
    // Virtual time bias (1D, position 9): bias_rho_comp_idx
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

    if (dt <= 0) return;

    auto params = galileanParams();

    // Create input from measurements
    GalileanInput u(measuredOmega, measuredAcc);

    // Get current state
    auto state = xi();

    // Compute K matrix
    Matrix10 K = state.Upsilon.AdjointMap() *
                Gal3::ExpmapDerivative(mapMeasurement10ToTangent10(u.w - state.bias) * dt) * dt;

    // Transform input
    auto u0 = psi(state.Upsilon.inverse(), u);

    // Update mean (deltaUpsilon_)
    Gal3 Lambda_update = Lambda(state, u, dt);
    deltaUpsilon_ = deltaUpsilon_ * Lambda_update;
    deltaTij_ += dt;

    // Propagate covariance
    Matrix20 A = Matrix20::Identity();
    A.block<10, 10>(0, ups_dim) = Gal3::ExpmapDerivative(
        mapMeasurement10ToTangent10(u0.w) * dt) * dt;
    A.block<10, 10>(ups_dim, ups_dim) = Gal3::Expmap(
        mapMeasurement10ToTangent10(u0.w) * dt).AdjointMap();

    Matrix20 B = Matrix20::Zero();
    B.block<10, 10>(0, 0) = -K;
    B.block<10, 10>(ups_dim, ups_dim) = state.Upsilon.AdjointMap() * dt;

    // Create noise covariance
    Matrix20 Q_d = Matrix20::Zero();
    // bc σ²_discrete = σ²_continuous / dt -- gtsam assumes continuous time
    Q_d.block<3,3>(ups_R_idx, ups_R_idx) = params->gyroscopeCovariance / dt;
    Q_d.block<3,3>(ups_v_idx, ups_v_idx) = params->accelerometerCovariance / dt;
    Q_d.block<3,3>(bias_w_idx, bias_w_idx) = params->getBiasOmegaCovariance() / dt;
    Q_d.block<3,3>(bias_a_idx, bias_a_idx) = params->getBiasAccCovariance() / dt;

    // Update covariance
    preintMeasCov_ = A * preintMeasCov_ * A.transpose() + B * Q_d * B.transpose();

    // Update Jacobian with respect to bias
    Matrix20 Phi_b = Matrix20::Identity();
    Phi_b.block<10,10>(0, ups_dim) = -(Matrix10::Identity() - K);                       //   -K        term
    Phi_b.block<10,10>(0, 0) = (Matrix10::Identity() - K) *
        Gal3::ExpmapDerivative(mapMeasurement10ToTangent10(u0.w) * dt) * dt;

    preintBiasJacobian_ = Phi_b * preintBiasJacobian_;
}

Vector9 PreintegratedGalileanMeasurements::biasCorrectedDelta(
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 6> H) const {

    // Get bias difference in 10D tangent space
    Vector10 delta_b = mapBias6ToTangent10(bias_i.vector() - biasHat_.vector());

    // Compute corrected measurement (uses bias Jacobian)
    Gal3 corrected = Gal3::Expmap(preintBiasJacobian_.block<10,10>(0, ups_dim) * delta_b) * deltaUpsilon_;

    // Initialize result vector for NavState tangent space
    Vector9 result;

    // Compute delta in rotation (using Lie algebra)
    result.segment<3>(NAV_R_IDX) = Rot3::Logmap(corrected.rotation() * deltaUpsilon_.rotation().inverse());

    // Compute delta in position and velocity (direct subtraction)
    result.segment<3>(NAV_P_IDX) = corrected.position() - deltaUpsilon_.position();
    result.segment<3>(NAV_V_IDX) = corrected.velocity() - deltaUpsilon_.velocity();

    // Calculate Jacobian if requested
    if (H) {
        auto compute_delta = [this](const imuBias::ConstantBias& b) {
            return this->biasCorrectedDelta(b, {});
        };
        *H = numericalDerivative11<Vector9, imuBias::ConstantBias, 6>(compute_delta, bias_i);
    }

    return result;
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
    boost::optional<Matrix&> H1, boost::optional<Matrix&> H2,
    boost::optional<Matrix&> H3, boost::optional<Matrix&> H4,
    boost::optional<Matrix&> H5) const {

    // 1. Get bias correction delta in tangent space
    Vector6 delta_bias_6D = bias_i.vector() - biasHat_.vector();
    Vector10 delta_b_10D = mapBias6ToTangent10(delta_bias_6D);

    // 2. Apply bias correction to preintegrated measurements
    Gal3 bias_correction = Gal3::Expmap(preintBiasJacobian_.block<10,10>(0, ups_dim) * delta_b_10D);
    Gal3 corrected_deltaUpsilon = bias_correction * deltaUpsilon_;

    // 3. Extract components from corrected measurements
    const Rot3& deltaR_corrected = corrected_deltaUpsilon.rotation();
    const Vector3 deltaP_corrected = corrected_deltaUpsilon.position();
    const Vector3& deltaV_corrected = corrected_deltaUpsilon.velocity();

    // 4. Get navigation parameters
    auto params = galileanParams();
    const Vector3& gravity = params->n_gravity;
    double deltaT = deltaTij();

    // 5. Compute predicted deltas based on the poses and velocities
    const Rot3& R_i = pose_i.rotation();
    const Vector3 p_i = pose_i.translation();
    const Rot3& R_j = pose_j.rotation();
    const Vector3 p_j = pose_j.translation();

    // Compute predicted relative rotation
    Rot3 deltaR_pred = R_i.between(R_j);

    // Compute predicted relative velocity (in body frame)
    Vector3 v_err_world = vel_j - vel_i - gravity * deltaT;
    Vector3 deltaV_pred = R_i.unrotate(v_err_world);

    // Compute predicted relative position (in body frame)
    Vector3 p_err_world = p_j - p_i - vel_i * deltaT - 0.5 * gravity * deltaT * deltaT;
    Vector3 deltaP_pred = R_i.unrotate(p_err_world);

    // 6. Compute error terms
    Vector3 error_R = Rot3::Logmap(deltaR_pred * deltaR_corrected.inverse());
    Vector3 error_p = deltaP_pred - deltaP_corrected;
    Vector3 error_v = deltaV_pred - deltaV_corrected;

    // 7. Construct the complete error vector
    Vector9 error;
    error.segment<3>(NAV_R_IDX) = error_R;
    error.segment<3>(NAV_P_IDX) = error_p;
    error.segment<3>(NAV_V_IDX) = error_v;

    // 8. Compute analytical Jacobians if requested
    if (H1 || H2 || H3 || H4 || H5) {
        // For now, let's use numerical Jacobians just for H5 (bias) and analytical for the rest
        Matrix3 R_i_matrix = R_i.matrix();
        Matrix3 R_j_matrix = R_j.matrix();
        Matrix3 R_between = deltaR_pred.matrix();
        Matrix3 R_corrected_inv = deltaR_corrected.inverse().matrix();
        Matrix3 R_error = (deltaR_pred * deltaR_corrected.inverse()).matrix();

        if (H1) {  // Jacobian wrt pose_i
            H1->resize(9, 6);
            // Fill rotation error Jacobian wrt pose_i (rotation part)
            // For rotational error, we need to compute d(Log(R_i.between(R_j) * deltaR_corr^-1))/d(R_i)
            Matrix3 D_error_R_wrt_R_i = -R_error.transpose() * R_j_matrix.transpose();
            H1->block<3,3>(NAV_R_IDX, 0) = D_error_R_wrt_R_i;

            // Position error is d(R_i^T * (p_j - p_i - v_i*dt - 0.5*g*dt^2) - deltaP_corr)/d(R_i, p_i)
            H1->block<3,3>(NAV_P_IDX, 0) = skewSymmetric(deltaP_pred);
            H1->block<3,3>(NAV_P_IDX, 3) = -R_i_matrix.transpose();

            // Velocity error is d(R_i^T * (v_j - v_i - g*dt) - deltaV_corr)/d(R_i, p_i)
            H1->block<3,3>(NAV_V_IDX, 0) = skewSymmetric(deltaV_pred);
            H1->block<3,3>(NAV_V_IDX, 3).setZero();
        }

        if (H2) {  // Jacobian wrt vel_i
            H2->resize(9, 3);
            H2->block<3,3>(NAV_R_IDX, 0).setZero();
            H2->block<3,3>(NAV_P_IDX, 0) = -R_i_matrix.transpose() * deltaT;
            H2->block<3,3>(NAV_V_IDX, 0) = -R_i_matrix.transpose();
        }

        if (H3) {  // Jacobian wrt pose_j
            H3->resize(9, 6);
            // Rotation error wrt R_j: d(Log(R_i.between(R_j) * deltaR_corr^-1))/d(R_j)
            H3->block<3,3>(NAV_R_IDX, 0) = R_error.transpose();
            H3->block<3,3>(NAV_R_IDX, 3).setZero();

            // Position and velocity errors wrt pose_j
            H3->block<3,3>(NAV_P_IDX, 0).setZero();
            H3->block<3,3>(NAV_P_IDX, 3) = R_i_matrix.transpose();
            H3->block<3,3>(NAV_V_IDX, 0).setZero();
            H3->block<3,3>(NAV_V_IDX, 3).setZero();
        }

        if (H4) {  // Jacobian wrt vel_j
            H4->resize(9, 3);
            H4->block<3,3>(NAV_R_IDX, 0).setZero();
            H4->block<3,3>(NAV_P_IDX, 0).setZero();
            H4->block<3,3>(NAV_V_IDX, 0) = R_i_matrix.transpose();
        }

        if (H5) {  // Jacobian wrt bias_i
            // For the bias Jacobian, we'll use a simpler approach based on the bias Jacobian
            // stored in preintBiasJacobian_
            Matrix96 H_bias = Matrix96::Zero();

            // Here we use the chain rule:
            // d(error)/d(bias) = d(error)/d(deltaUpsilon_corrected) * d(deltaUpsilon_corrected)/d(bias)

            // For the rotation component:
            // We know that error_R = Log(deltaR_pred * deltaR_corrected^-1)
            // d(error_R)/d(deltaR_corrected) = -J_r^-1(error_R) * deltaR_pred
            so3::DexpFunctor dexp_functor(error_R);
            Matrix3 Jr_inv = dexp_functor.rightJacobianInverse();
            Matrix3 D_error_R_deltaR = -Jr_inv * R_between * R_corrected_inv;

            // Rotation error Jacobian wrt bias
            H_bias.block<3,3>(NAV_R_IDX, 3) = D_error_R_deltaR * preintBiasJacobian_.block<3,3>(ups_R_idx, bias_w_idx);

            // Position and velocity error Jacobians wrt bias - direct effect of bias on measurements
            H_bias.block<3,3>(NAV_P_IDX, 0) = -preintBiasJacobian_.block<3,3>(ups_p_idx, bias_a_idx);
            H_bias.block<3,3>(NAV_P_IDX, 3) = -preintBiasJacobian_.block<3,3>(ups_p_idx, bias_w_idx);
            H_bias.block<3,3>(NAV_V_IDX, 0) = -preintBiasJacobian_.block<3,3>(ups_v_idx, bias_a_idx);
            H_bias.block<3,3>(NAV_V_IDX, 3) = -preintBiasJacobian_.block<3,3>(ups_v_idx, bias_w_idx);

            *H5 = H_bias;
        }
    }

    return error;
}

NavState PreintegratedGalileanMeasurements::predict(const NavState& state_i,
    const imuBias::ConstantBias& bias_i,
    OptionalJacobian<9, 9> H_navstate_wrt_navstate_i,
    OptionalJacobian<9, 6> H_navstate_wrt_bias_i) const {

    // Get the bias correction using the existing method
    Matrix96 H_correction_wrt_bias;
    Vector9 correction = biasCorrectedDelta(bias_i,
                            H_navstate_wrt_bias_i ? &H_correction_wrt_bias : nullptr);

    // Extract components from correction vector (in NavState tangent space)
    Vector3 theta = correction.segment<3>(NAV_R_IDX);
    Vector3 pos_correction = correction.segment<3>(NAV_P_IDX);
    Vector3 vel_correction = correction.segment<3>(NAV_V_IDX);

    // Apply corrections to obtain bias-corrected values
    Rot3 deltaR_corrected = deltaUpsilon_.rotation() * Rot3::Expmap(theta);
    Vector3 deltaP_corrected = deltaUpsilon_.position() + pos_correction;
    Vector3 deltaV_corrected = deltaUpsilon_.velocity() + vel_correction;

    // Apply standard physics to predict new state
    const Rot3& R_i = state_i.attitude();
    const Point3& p_i = state_i.position();
    const Vector3& v_i = state_i.velocity();
    double deltaT = deltaTij();
    const Vector3& gravity = galileanParams()->n_gravity;

    Rot3 R_j = R_i * deltaR_corrected;
    Vector3 v_j = v_i + R_i * deltaV_corrected + gravity * deltaT;
    Point3 p_j = p_i + Point3(R_i * deltaP_corrected + v_i * deltaT + 0.5 * gravity * deltaT * deltaT);

    NavState predicted(Pose3(R_j, p_j), v_j);

    // Calculate Jacobians if requested
    if (H_navstate_wrt_navstate_i) {
        auto predict_wrapper = [this, &gravity](const NavState& s, const imuBias::ConstantBias& b) {
            Vector9 corr = this->biasCorrectedDelta(b);

            // Extract and apply corrections
            const Rot3& R = s.attitude();
            const Point3& p = s.position();
            const Vector3& v = s.velocity();

            Rot3 deltaR_corr = this->deltaUpsilon_.rotation() *
                               Rot3::Expmap(corr.segment<3>(NAV_R_IDX));
            Vector3 deltaP_corr = this->deltaUpsilon_.position() +
                                  corr.segment<3>(NAV_P_IDX);
            Vector3 deltaV_corr = this->deltaUpsilon_.velocity() +
                                  corr.segment<3>(NAV_V_IDX);

            double dt = this->deltaTij();

            return NavState(
                Pose3(R * deltaR_corr,
                      p + Point3(R * deltaP_corr + v * dt + 0.5 * gravity * dt * dt)),
                v + R * deltaV_corr + gravity * dt
            );
        };

        *H_navstate_wrt_navstate_i = numericalDerivative21<NavState, NavState, imuBias::ConstantBias>(
            predict_wrapper, state_i, bias_i, 1e-7);
    }

    if (H_navstate_wrt_bias_i) {
        *H_navstate_wrt_bias_i = H_correction_wrt_bias; // Already computed above
    }

    return predicted;
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

Vector GalileanImuFactor::evaluateError(const Pose3& pose_i, const Vector3& vel_i,
                                       const imuBias::ConstantBias& bias_i,
                                       const Pose3& pose_j, const Vector3& vel_j,
                                       const imuBias::ConstantBias& bias_j,
                                       OptionalMatrixType H1, OptionalMatrixType H2,
                                       OptionalMatrixType H3, OptionalMatrixType H4,
                                       OptionalMatrixType H5, OptionalMatrixType H6) const {
    // Part 1: Compute bias evolution error (as in CombinedImuFactor)
    Matrix6 Hbias_i, Hbias_j;
    Vector6 fbias = traits<imuBias::ConstantBias>::Between(bias_j, bias_i,
        H6 ? &Hbias_j : nullptr, H3 ? &Hbias_i : nullptr).vector();

    // Part 2: Compute navigation state error using Galilean formulation
    boost::optional<Matrix&> pimH_pose_i = H1 ? boost::optional<Matrix&>(*H1) : boost::none;
    boost::optional<Matrix&> pimH_vel_i  = H2 ? boost::optional<Matrix&>(*H2) : boost::none;
    boost::optional<Matrix&> pimH_bias_i = H3 ? boost::optional<Matrix&>(*H3) : boost::none;
    boost::optional<Matrix&> pimH_pose_j = H4 ? boost::optional<Matrix&>(*H4) : boost::none;
    boost::optional<Matrix&> pimH_vel_j  = H5 ? boost::optional<Matrix&>(*H5) : boost::none;

    Vector9 r_nav = _PIM.computeErrorAndJacobians(pose_i, vel_i, pose_j, vel_j, bias_i,
                                                pimH_pose_i, pimH_vel_i,
                                                pimH_pose_j, pimH_vel_j,
                                                pimH_bias_i);

    // Combine and return the full error vector
    Vector15 error;
    error << r_nav, fbias;

    // Properly structure the Jacobians to match the error vector
    if (H3) {  // bias_i Jacobian needs special handling
        H3->resize(15, 6);
        H3->block<9, 6>(0, 0) = *pimH_bias_i;  // Navigation error w.r.t bias_i
        H3->block<6, 6>(9, 0) = Hbias_i;       // Bias error w.r.t bias_i
    }

    if (H6) {  // bias_j Jacobian
        H6->resize(15, 6);
        H6->block<9, 6>(0, 0).setZero();  // Navigation error doesn't depend on bias_j
        H6->block<6, 6>(9, 0) = Hbias_j;  // Bias error w.r.t bias_j
    }

    return error; // @TODO - should probably return full 15D vector, but need to modify tests
    // return r_nav;
}

} // namespace gtsam
