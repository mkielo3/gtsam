/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file CombinedGalileanImuFactor.h
 * @brief Combined header for Galilean IMU preintegration and factor.
 * Contains PreintegratedGalileanMeasurements and GalileanImuFactor.
 * @author Refactored by [Your Tool Name/Your Name]
 */

#pragma once

// Includes from PreintegratedGalileanMeasurements.h
#include <gtsam/navigation/PreintegrationBase.h>
#include <gtsam/navigation/PreintegrationGalileanParams.h> // Assumed to exist
#include <gtsam/geometry/Gal3.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/base/OptionalJacobian.h>
#include <boost/optional.hpp>

// Includes from GalileanImuFactor.h
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>
#include <gtsam/navigation/ImuBias.h> // Using ConstantBias
#include <gtsam/base/Matrix.h>

// Define the size of the augmented state tangent space (Upsilon tangent + Bias tangent)
constexpr size_t GALILEAN_PREINTEGRATION_DIM = 20;
using Matrix20 = Eigen::Matrix<double, GALILEAN_PREINTEGRATION_DIM, GALILEAN_PREINTEGRATION_DIM>;
using Vector20 = Eigen::Matrix<double, GALILEAN_PREINTEGRATION_DIM, 1>;

// Define matrix types needed for Jacobians (matching PIM internal calculations)
using Matrix10 = Eigen::Matrix<double, 10, 10>;
using Matrix10_6 = Eigen::Matrix<double, 10, 6>;
using Matrix9_10 = Eigen::Matrix<double, 9, 10>;
using Matrix96 = Eigen::Matrix<double, 9, 6>; // Jacobian size for bias
using Matrix99 = Eigen::Matrix<double, 9, 9>; // For NavState Jacobians
using Matrix93 = Eigen::Matrix<double, 9, 3>; // Vel Jacobian
using Matrix9 = Eigen::Matrix<double, 9, 9>;  // NavState covariance

namespace gtsam {

/**
 * PreintegratedGalileanMeasurements integrates IMU measurements on the
 * manifold M = Gal(3) x R^10 according to Alg. 1 in Delama et al., RA-L 2024.
 * It stores the mean preintegrated measurement deltaUpsilon (as a Gal3 object)
 * and the 20x20 covariance matrix preintMeasCov_.
 *
 * The GalileanPreintegrationParams should be used.
 * @author Original author: (From PreintegratedGalileanMeasurements.h)
 */
class GTSAM_EXPORT PreintegratedGalileanMeasurements : public PreintegrationBase {
 public:
 typedef PreintegratedGalileanMeasurements This;
 typedef PreintegrationBase Base;
 typedef GalileanPreintegrationParams Params; // Specific parameters for this method

 // Add these index constants
 static constexpr size_t ups_dim = 10;
 static constexpr size_t bias_dim = 10;

 // Indices within Upsilon tangent vector (0-9) [rho, nu, theta, t]
 static constexpr size_t ups_p_idx = 0; // rho component (position)
 static constexpr size_t ups_v_idx = 3; // nu component (velocity)
 static constexpr size_t ups_R_idx = 6; // theta component (rotation)
 static constexpr size_t ups_t_idx = 9; // t component (time duration)

 // Indices within Bias tangent vector (0-9) [b_omega, b_acc, b_nu, b_rho]
 static constexpr size_t bias_w_comp_idx = 0; // b_omega (gyro bias)
 static constexpr size_t bias_a_comp_idx = 3; // b_acc (accel bias)
 static constexpr size_t bias_nu_comp_idx = 6; // b_nu (virtual velocity bias)
 static constexpr size_t bias_rho_comp_idx = 9; // b_rho (virtual time bias)

 // Indices within the full 20D state tangent vector [Upsilon | Bias]
 static constexpr size_t bias_w_idx = ups_dim + bias_w_comp_idx; // 10
 static constexpr size_t bias_a_idx = ups_dim + bias_a_comp_idx; // 13
 static constexpr size_t bias_nu_idx = ups_dim + bias_nu_comp_idx; // 16
 static constexpr size_t bias_rho_idx = ups_dim + bias_rho_comp_idx; // 19

 // Define indices for accessing the 9D NavState tangent vector
 static constexpr size_t NAV_R_IDX = 0;
 static constexpr size_t NAV_P_IDX = 3;
 static constexpr size_t NAV_V_IDX = 6;

 // Returns the current state as a pair (Upsilon, bias)
 struct GalileanState {
   Gal3 Upsilon;
   Vector10 bias;
 };

 // Input structure for measurements
 struct GalileanInput {
   Vector10 w;     // Combined measurements
   Vector10 tau;   // Bias random walk component

   GalileanInput(const Vector3& gyro, const Vector3& acc)
     : w(Vector10::Zero()), tau(Vector10::Zero()) {
     w.segment<3>(bias_w_comp_idx) = gyro;
     w.segment<3>(bias_a_comp_idx) = acc;
     w(bias_rho_comp_idx) = 1.0;
   }
 };

 /**
  * @brief Current state accessor
  * @return Current state as a GalileanState object
  */
  GalileanState xi() const {
      // Should properly compute current state from deltaUpsilon_ and biasHat_
      Vector10 biasVec = Vector10::Zero();
      biasVec.segment<3>(bias_w_comp_idx) = biasHat_.gyroscope();
      biasVec.segment<3>(bias_a_comp_idx) = biasHat_.accelerometer();
      return {deltaUpsilon_, biasVec};
  }


 /**
  * @brief Transforms an input by the inverse of a state
  * @param Upsilon_inv Inverse of an Upsilon matrix
  * @param u Input to transform
  * @return Transformed input
  */
 GalileanInput psi(const Gal3& Upsilon_inv, const GalileanInput& u) const {
     GalileanInput result = u;
     result.w = Upsilon_inv.AdjointMap() * (u.w - mapBias6ToTangent10(biasHat_.vector()));
     return result;
 }

 /**
  * @brief Computes the state update from an input
  * @param state Current state
  * @param u Input measurement
  * @param dt Time step
  * @return Updated Gal3 measurement
  */
  Gal3 Lambda(const GalileanState& state, const GalileanInput& u, double dt) const {
      Vector10 w_hat = u.w - state.bias;

      // Convert to tangent space ordering for GTSAM's Gal3
      Vector10 tangent = Vector10::Zero();
      tangent.segment<3>(ups_p_idx) = w_hat.segment<3>(bias_nu_comp_idx);  // virtual vel -> position
      tangent.segment<3>(ups_v_idx) = w_hat.segment<3>(bias_a_comp_idx);   // acc -> velocity
      tangent.segment<3>(ups_R_idx) = w_hat.segment<3>(bias_w_comp_idx);   // gyro -> rotation
      tangent(ups_t_idx) = w_hat(bias_rho_comp_idx);                       // virtual time -> time

      return Gal3::Expmap(tangent * dt);
  }


 protected: // Internal state representation
  /// Preintegrated measurement mean \hat{\Upsilon}_k (stores deltaR, deltaP, deltaV, deltaT)
  Gal3 deltaUpsilon_;

  /// Preintegrated measurement covariance Sigma_k (20x20 matrix). Eq 35.
  Matrix20 preintMeasCov_;

  /// Jacobian of preintegrated state w.r.t. initial bias J_xi (20x20 matrix). Eq 38.
  Matrix20 preintBiasJacobian_;

  /// Counter for integration steps, useful for debugging or specific logic.


 public:
  /// Default constructor for serialization.
  PreintegratedGalileanMeasurements() :
    deltaUpsilon_(Gal3::Identity()),
    preintMeasCov_(Matrix20::Zero()),
    preintBiasJacobian_(Matrix20::Identity()) {}

  /// Constructor, initializes with parameters and optional bias.
  PreintegratedGalileanMeasurements(const std::shared_ptr<Params>& p,
                                   const Bias& biasHat = Bias());

  /// Virtual destructor.
  ~PreintegratedGalileanMeasurements() override {}

 public:
  /// @name Basic utilities
  /// @{

  /// Re-initialize preintegrated measurements to zero.
  void resetIntegration() override;

  /// Return shared_ptr to parameters, ensuring correct type.
  std::shared_ptr<const Params> galileanParams() const;

  /// @}

  /// @name Access instance variables needed by factors or users
  /// @{

  /// Access the preintegrated measurement mean as a Gal3 object
  const Gal3& deltaUpsilon() const { return deltaUpsilon_; }

  // --- Provide accessors matching the base class interface ---
  // --- extracting from deltaUpsilon_                 ---
  Rot3 deltaRij() const override { return deltaUpsilon_.rotation(); }
  Vector3 deltaPij() const override { return deltaUpsilon_.position(); }
  Vector3 deltaVij() const override { return deltaUpsilon_.velocity(); }
  // deltaTij() is inherited from Base and updated in integrateMeasurement

  /// Return the 9x9 covariance matrix for the NavState portion (R, p, v) error.
  Matrix9 preintegratedNavStateCovariance() const;

  NavState deltaXij() const override {
      return NavState(deltaRij(), deltaPij(), deltaVij());
  }

  /// Return the full 20x20 uncertainty covariance matrix Sigma_k
  const Matrix20& uncertaintyCovariance() const { return preintMeasCov_; }

  /// Return the full 20x20 bias Jacobian J_xi
  const Matrix20& biasJacobian() const { return preintBiasJacobian_; }
  /// @}


  /// @name Main functionality
  /// @{

  /**
   * Integrate a single IMU measurement using the Galilean propagation (Algorithm 1).
   * Updates deltaUpsilon_, preintMeasCov_, preintBiasJacobian_, and deltaTij_.
   * @param measuredAcc Measured acceleration in sensor frame
   * @param measuredOmega Measured angular velocity in sensor frame
   * @param dt Time interval for the measurement
   */
  void integrateMeasurement(const Vector3& measuredAcc,
                            const Vector3& measuredOmega, double dt) override;

  void update(const Vector3& measuredAcc, const Vector3& measuredOmega,
      const double dt, Matrix9* A = nullptr, Matrix93* B = nullptr, Matrix93* C = nullptr) override {
        if (A || B || C) {
             throw std::logic_error("PreintegratedGalileanMeasurements::update with Jacobians A, B, C is not supported. Use integrateMeasurement instead.");
        }
        integrateMeasurement(measuredAcc, measuredOmega, dt);
      }


  /**
   * Calculate the change in preintegrated measurements (deltaRij, deltaPij, deltaVij)
   * given a change in bias from the bias estimate used during integration (`biasHat_`)
   * to a new bias estimate (`bias_i`). Uses first-order approximation (Eq. 39).
   * Result is a 9D vector in the NavState tangent space [Log(R), p, v].
   * @param bias_i New estimate of the bias
   * @param H Optional Jacobian of the 9D correction vector wrt bias_i (9x6)
   */
  Vector9 biasCorrectedDelta(const imuBias::ConstantBias& bias_i,
                             OptionalJacobian<9, 6> H = {}) const override;


  /**
   * Predicts the state (NavState) at time j based on the state at time i and the
   * preintegrated measurements. Includes bias correction.
   * @param state_i NavState at time i
   * @param bias_i Bias estimate to use for correction
   * @param H1 Optional Jacobian wrt state_i (9x9)
   * @param H2 Optional Jacobian wrt bias_i (9x6)
   */
  NavState predict(const NavState& state_i, const imuBias::ConstantBias& bias_i,
                   OptionalJacobian<9, 9> H1 = {},
                   OptionalJacobian<9, 6> H2 = {}) const;


  /**
   * Compute the 9-DOF error vector between predicted and measured states and
   * optionally its Jacobians. This is the core computation needed by the GalileanImuFactor.
   * @param pose_i Pose3 at time i
   * @param vel_i Velocity at time i (Vector3)
   * @param pose_j Pose3 at time j
   * @param vel_j Velocity at time j (Vector3)
   * @param bias_i Bias estimate at time i (imuBias::ConstantBias)
   * @param H1 Optional Jacobian output matrix for error wrt pose_i (Pointer to 9x6 matrix)
   * @param H2 Optional Jacobian output matrix for error wrt vel_i  (Pointer to 9x3 matrix)
   * @param H3 Optional Jacobian output matrix for error wrt pose_j (Pointer to 9x6 matrix)
   * @param H4 Optional Jacobian output matrix for error wrt vel_j  (Pointer to 9x3 matrix)
   * @param H5 Optional Jacobian output matrix for error wrt bias_i (Pointer to 9x6 matrix)
   * @return 9-dimensional error vector (rotation, position, velocity)
   */
  Vector9 computeErrorAndJacobians(const Pose3& pose_i, const Vector3& vel_i,
                                   const Pose3& pose_j, const Vector3& vel_j,
                                   const imuBias::ConstantBias& bias_i,
                                   boost::optional<Matrix&> H1,
                                   boost::optional<Matrix&> H2,
                                   boost::optional<Matrix&> H3,
                                   boost::optional<Matrix&> H4,
                                   boost::optional<Matrix&> H5) const;

  /// @}

  /// @name Testable
  /// @{
  void print(const std::string& s = "PreintegratedGalileanMeasurements:") const override;
  bool equals(const PreintegratedGalileanMeasurements& other, double tol = 1e-9) const;
  /// @}

  // Made public for testing
  // Internal helper to map 6D bias vector (acc, gyro) to 10D bias tangent vector
  static Vector10 mapBias6ToTangent10(const Vector6& bias6D);

  // Internal helper to map 10D measurement vector to 10D tangent vector
  static Vector10 mapMeasurement10ToTangent10(const Vector10& measurement10D);


private:


  /** Serialization function */
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar & BOOST_SERIALIZATION_NVP(deltaUpsilon_);
    ar & BOOST_SERIALIZATION_NVP(preintMeasCov_);
    ar & BOOST_SERIALIZATION_NVP(preintBiasJacobian_);
  }
#endif

public:
 GTSAM_MAKE_ALIGNED_OPERATOR_NEW

}; // PreintegratedGalileanMeasurements


/**
 * A factor relating two Pose3 states, two Vector3 velocities (linear velocity),
 * and two ConstantBias states based on preintegrated IMU measurements
 * using the Galilean Preintegration formulation from Delama et al., RA-L 2024.
 *
 * This factor relies on the PreintegratedGalileanMeasurements class.
 * @author Original author: (From GalileanImuFactor.h)
 */
class GTSAM_EXPORT GalileanImuFactor : public NoiseModelFactorN<Pose3, Vector3, imuBias::ConstantBias,
                                                               Pose3, Vector3, imuBias::ConstantBias> {

public:
  // shorthand for base class and self
  typedef NoiseModelFactorN<Pose3, Vector3, imuBias::ConstantBias, Pose3, Vector3, imuBias::ConstantBias> Base;
  typedef GalileanImuFactor This;

private:
  PreintegratedGalileanMeasurements _PIM; ///< Preintegrated measurements instance.

public:

  // Provide standard typedefs needed by factor graph
  typedef std::shared_ptr<This> shared_ptr;

  /** Default constructor - only use for serialization */
  GalileanImuFactor() {}

  /**
   * Constructor. The noise model is automatically set using the covariance
   * from the PreintegratedGalileanMeasurements object.
   * @param key_pose_i      Key for state i Pose3 (X(i))
   * @param key_vel_i       Key for state i Velocity (Vector3, V(i))
   * @param key_bias_i      Key for state i IMU Bias (B(i))
   * @param key_pose_j      Key for state j Pose3 (X(j))
   * @param key_vel_j       Key for state j Velocity (Vector3, V(j))
   * @param key_bias_j      Key for state j IMU Bias (B(j))
   * @param pim             Preintegrated measurements from i to j (contains mean and covariance)
   */
  GalileanImuFactor(Key key_pose_i, Key key_vel_i, Key key_bias_i,
                    Key key_pose_j, Key key_vel_j, Key key_bias_j,
                    const PreintegratedGalileanMeasurements& pim);

  /// Default destructor
  ~GalileanImuFactor() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override;

  /// @name Testable
  /// @{

  /// Print the factor's details
  void print(const std::string& s = "GalileanImuFactor",
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override;

  /// Check equality with another factor
  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override;
  /// @}

  /// @name Factor interface
  /// @{

  /**
   * Evaluate the 9-dimensional error between predicted states and the measured state j.
   * This function calls PIM.computeErrorAndJacobians.
   * @param H1-H6 Optional Jacobians. Note H6 (wrt bias_j) is always zero.
   */
   Vector evaluateError(const Pose3& pose_i, const Vector3& vel_i, const imuBias::ConstantBias& bias_i,
                        const Pose3& pose_j, const Vector3& vel_j, const imuBias::ConstantBias& bias_j,
                        OptionalMatrixType H1 = nullptr,
                        OptionalMatrixType H2 = nullptr,
                        OptionalMatrixType H3 = nullptr,
                        OptionalMatrixType H4 = nullptr,
                        OptionalMatrixType H5 = nullptr,
                        OptionalMatrixType H6 = nullptr) const override;


  /// @}

  /// Access the PreintegratedGalileanMeasurements stored within the factor
  const PreintegratedGalileanMeasurements& preintegratedMeasurements() const {
    return _PIM;
  }

private:

#ifdef GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
    ar & boost::serialization::make_nvp("NoiseModelFactorN",
        boost::serialization::base_object<Base>(*this));
    ar & BOOST_SERIALIZATION_NVP(_PIM);
  }
#endif

public:
 GTSAM_MAKE_ALIGNED_OPERATOR_NEW

}; // class GalileanImuFactor

// Traits specializations
template <>
struct traits<PreintegratedGalileanMeasurements> : public Testable<PreintegratedGalileanMeasurements> {};

template <>
struct traits<GalileanImuFactor> : public Testable<GalileanImuFactor> {};


} // namespace gtsam
