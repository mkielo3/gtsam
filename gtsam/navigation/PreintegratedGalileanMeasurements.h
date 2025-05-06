/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file PreintegratedGalileanMeasurements.h
 * @brief Class for IMU preintegration on the manifold M = Gal(3) x R^10,
 * based on Delama et al., "Equivariant IMU Preintegration with Biases:
 * a Galilean Group Approach", RA-L 2024. Uses the Gal3 Lie group class.
 * This class computes the preintegrated measurement mean (deltaUpsilon_) and
 * the 20x20 covariance matrix (preintMeasCov_) according to Algorithm 1.
 * @author Your Name
 */

#pragma once

#include <gtsam/navigation/PreintegrationBase.h>
#include <gtsam/navigation/GalileanPreintegrationParams.h>
#include <gtsam/geometry/Gal3.h> // Include the Gal3 Lie group definition
#include <gtsam/navigation/NavState.h> // Include NavState for predict/error types
#include <gtsam/base/OptionalJacobian.h> // Include for OptionalJacobian

#include <boost/optional.hpp> // Include for boost::optional

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
using Matrix9 = Eigen::Matrix<double, 9, 9>; // NavState covariance

namespace gtsam {

/**
 * PreintegratedGalileanMeasurements integrates IMU measurements on the
 * manifold M = Gal(3) x R^10 according to Alg. 1 in Delama et al., RA-L 2024.
 * It stores the mean preintegrated measurement deltaUpsilon (as a Gal3 object)
 * and the 20x20 covariance matrix preintMeasCov_.
 *
 * The GalileanPreintegrationParams should be used.
 *
 * **CRITICAL NOTE:** This implementation currently relies on **numerical derivatives**
 * for the Gal3::ExpmapDerivative used within the covariance propagation step
 * (`integrateMeasurement`). For optimal performance, accuracy, and stability,
 * the analytical closed-form expressions for the Gal3 Jacobians ($J_L$, $Q_1$, $Q_2$, $U_1$)
 * derived in Appendix A of the paper **must be implemented** in `Gal3.cpp` and
 * used within `integrateMeasurement`. Similarly, the Jacobians calculated in
 * `computeErrorAndJacobians` are currently numerical and should ideally be analytical.
 */
class GTSAM_EXPORT PreintegratedGalileanMeasurements : public PreintegrationBase {
 public:
  typedef PreintegratedGalileanMeasurements This;
  typedef PreintegrationBase Base;
  typedef GalileanPreintegrationParams Params; // Specific parameters for this method

 protected: // Internal state representation
  /// Preintegrated measurement mean \hat{\Upsilon}_k (stores deltaR, deltaP, deltaV, deltaT)
  Gal3 deltaUpsilon_;

  /// Preintegrated measurement covariance Sigma_k (20x20 matrix). Eq 35.
  Matrix20 preintMeasCov_;

  /// Jacobian of preintegrated state w.r.t. initial bias J_xi (20x20 matrix). Eq 38.
  Matrix20 preintBiasJacobian_;

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
  std::shared_ptr<const Params> galileanParams() const; // Implementation moved to cpp

  /// @}

  /// @name Access instance variables needed by factors or users
  /// @{

  /// Access the preintegrated measurement mean as a Gal3 object
  const Gal3& deltaUpsilon() const { return deltaUpsilon_; }

  // --- Provide accessors matching the base class interface ---
  // --- extracting from deltaUpsilon_                 ---
  Rot3 deltaRij() const override { return deltaUpsilon_.rotation(); }
  // *** FIXED: Removed .vector() call. Point3 converts to Vector3. ***
  Vector3 deltaPij() const override { return deltaUpsilon_.position(); }
  Vector3 deltaVij() const override { return deltaUpsilon_.velocity(); }
  // deltaTij() is inherited from Base and updated in integrateMeasurement

  /// Return the 9x9 covariance matrix for the NavState portion (R, p, v) error.
  /// Extracts the relevant block from the full 20x20 preintMeasCov_.
  /// Order for NavState tangent space: [Log(R), p, v]
  Matrix9 preintegratedNavStateCovariance() const; // Implementation moved to cpp

  /// **NOTE:** This returns the *mean* delta computed by the PIM.
  /// Standard ImuFactor uses this with standard predict functions.
  /// Required by base class but less meaningful in the Galilean context if
  /// factor directly used the 20D error. Returns deltaXij computed from deltaUpsilon_.
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

  // NOTE: The `update` method with Jacobians A, B, C is part of the old
  //       TangentPreintegration interface. The Galilean PIM propagates its own
  //       internal state and covariance directly. We override it to avoid
  //       ambiguity but delegate to integrateMeasurement.
  void update(const Vector3& measuredAcc, const Vector3& measuredOmega,
      const double dt, Matrix9* A = nullptr, Matrix93* B = nullptr, Matrix93* C = nullptr) override {
        if (A || B || C) {
             // Provide a more informative error or warning
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
   * **Note:** This uses standard IMU forward prediction, adding gravity effects
   * to the bias-corrected PIM deltas. It does not use Gal3 group ops for prediction.
   * @param state_i NavState at time i
   * @param bias_i Bias estimate to use for correction
   * @param H1 Optional Jacobian wrt state_i (9x9)
   * @param H2 Optional Jacobian wrt bias_i (9x6)
   */
  NavState predict(const NavState& state_i, const imuBias::ConstantBias& bias_i,
                   OptionalJacobian<9, 9> H1 = {},
                   OptionalJacobian<9, 6> H2 = {}) const; // Implementation moved to cpp


  /**
   * Compute the 9-DOF error vector between predicted and measured states and
   * optionally its Jacobians. This is the core computation needed by the GalileanImuFactor.
   * Error = NavState(pose_j, vel_j).localCoordinates(predict(NavState(pose_i, vel_i), bias_i))
   * but implemented more directly by comparing state deltas to PIM deltas.
   * The PIM's internal deltaTij() is used to account for gravity when comparing states.
   *
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
   *
   * **NOTE:** This implementation currently uses **numerical derivatives** for Jacobians H1-H5.
   * For optimal performance, analytical Jacobians should be derived and implemented.
   */
  Vector9 computeErrorAndJacobians(const Pose3& pose_i, const Vector3& vel_i,
                                   const Pose3& pose_j, const Vector3& vel_j,
                                   const imuBias::ConstantBias& bias_i,
                                   boost::optional<Matrix&> H1, // Jacobian wrt pose_i
                                   boost::optional<Matrix&> H2, // Jacobian wrt vel_i
                                   boost::optional<Matrix&> H3, // Jacobian wrt pose_j
                                   boost::optional<Matrix&> H4, // Jacobian wrt vel_j
                                   boost::optional<Matrix&> H5) const; // Jacobian wrt bias_i

  /// @}

  /// @name Testable
  /// @{
  void print(const std::string& s = "PreintegratedGalileanMeasurements:") const override;
  bool equals(const PreintegratedGalileanMeasurements& other, double tol = 1e-9) const;
  /// @}


private:
  // Internal helper to map 6D bias vector (acc, gyro) to 10D bias tangent vector (b_omega, b_acc, 0, 0)
  static Vector10 mapBias6ToTangent10(const Vector6& bias6D); // Implementation moved to cpp

  // Internal helper to map 10D measurement vector (omega, acc, 0, 1) to 10D tangent vector (rho=0, nu=acc, theta=omega, t=1)
  static Vector10 mapMeasurement10ToTangent10(const Vector10& measurement10D); // Implementation moved to cpp


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

} // namespace gtsam
