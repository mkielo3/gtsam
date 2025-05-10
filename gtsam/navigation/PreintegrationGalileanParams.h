/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file GalileanPreintegrationParams.h
 * @brief Parameters for Galilean Preintegration.
 * @author (Your Name), based on Delama et al., RA-L 2024
 */

#pragma once

#include <gtsam/navigation/PreintegrationParams.h> // Use GTSAM's base class
#include <iostream> // For print statement

namespace gtsam {

// Useful constants
typedef Eigen::Matrix<double, 10, 10> Matrix10;
typedef Eigen::Matrix<double, 20, 20> Matrix20;

/**
 * Parameters for preintegration based on the Galilean group formulation
 * presented in Delama et al., "Equivariant IMU Preintegration with Biases:
 * a Galilean Group Approach", RA-L 2024.
 *
 * Extends the standard PreintegrationParams to include parameters specific
 * to the Galilean preintegration method, particularly bias random walk noise.
 * These correspond to sigma_d_tau_omega^2 and sigma_d_tau_a^2 in the paper.
 * Note: The paper uses dt * sigma_c^2 for discrete covariance, while GTSAM
 * typically uses sigma_c^2 / dt. Ensure consistency in usage. Here we store
 * the continuous-time variances sigma_c^2.
 */
struct GTSAM_EXPORT GalileanPreintegrationParams : PreintegrationParams {
  /// Continuous-time covariance matrix for gyroscope bias random walk (sigma_c_tau_omega^2)
  Matrix3 biasOmegaCovariance;
  /// Continuous-time covariance matrix for accelerometer bias random walk (sigma_c_tau_a^2)
  Matrix3 biasAccCovariance;
  /// Covariance matrix for the initial bias estimate uncertainty (Sigma_B in paper, for b_omega, b_acc).
  Matrix6 biasAccOmegaInt; // Covariance of (accel bias, gyro bias)

  /// Covariance for virtual velocity component
  Matrix3 virtualVelCovariance;
  /// Covariance for virtual time scale component
  double virtualTimeScaleCovariance;
  /// Random walk covariance for virtual velocity bias
  Matrix3 biasVirtualVelCovariance;
  /// Random walk covariance for virtual time scale bias
  double biasVirtualTimeCovariance;
  /// Initial covariance for all bias components (10x10)
  Matrix10 biasExtendedInit;

  /// Default constructor, initializes with identity matrices.
  GalileanPreintegrationParams()
      : PreintegrationParams(Vector3(0, 0, -9.81)), // Use default base constructor (Z-up gravity)
        biasOmegaCovariance(I_3x3),
        biasAccCovariance(I_3x3),
        biasAccOmegaInt(Matrix6::Zero()),  // Change from I_6x6 to Zero()
        virtualVelCovariance(Matrix3::Zero()),
        virtualTimeScaleCovariance(0),
        biasVirtualVelCovariance(Matrix3::Zero()),
        biasVirtualTimeCovariance(0) {
      // Initialize 10x10 bias covariance with zeros, then set the standard 6x6 part
      biasExtendedInit = Matrix10::Zero();
      // Now these lines won't set anything non-zero since biasAccOmegaInt is zero
      biasExtendedInit.block<3,3>(0,0) = biasAccOmegaInt.block<3,3>(0,0); // Acc bias
      biasExtendedInit.block<3,3>(3,3) = biasAccOmegaInt.block<3,3>(3,3); // Omega bias
  }

  /// Constructor initializes with gravity and optional bias random walk sigmas.
  GalileanPreintegrationParams(const Vector3& n_gravity_)
      : PreintegrationParams(n_gravity_), // Initialize base class (might normalize)
        biasOmegaCovariance(I_3x3),
        biasAccCovariance(I_3x3),
        biasAccOmegaInt(Matrix6::Zero()),
        virtualVelCovariance(Matrix3::Zero()),
        virtualTimeScaleCovariance(0),
        biasVirtualVelCovariance(Matrix3::Zero()),
        biasVirtualTimeCovariance(0) {

      // Override any normalization from base class
      n_gravity = n_gravity_;  // Force set to the actual input gravity

      // Initialize 10x10 bias covariance with zeros
      biasExtendedInit = Matrix10::Zero();
      biasExtendedInit.block<3,3>(0,0) = biasAccOmegaInt.block<3,3>(0,0); // Acc bias
      biasExtendedInit.block<3,3>(3,3) = biasAccOmegaInt.block<3,3>(3,3); // Omega bias
  }

  /// Named constructor for Z-down navigation frame (NED).
  static std::shared_ptr<GalileanPreintegrationParams> MakeSharedD(
      double g = 9.81) {
    return std::shared_ptr<GalileanPreintegrationParams>(
        new GalileanPreintegrationParams(Vector3(0, 0, g)));
  }

  /// Named constructor for Z-up navigation frame (ENU).
  static std::shared_ptr<GalileanPreintegrationParams> MakeSharedU(
      double g = 9.81) {
    Vector3 gravity(0, 0, -g);  // Create the actual gravity vector
    auto params = std::shared_ptr<GalileanPreintegrationParams>(
        new GalileanPreintegrationParams(gravity));

    // Ensure gravity is preserved
    params->n_gravity = gravity;

    return params;
  }

  /// Named constructor for Z-down navigation frame (NED) with Galilean-specific defaults
  static std::shared_ptr<GalileanPreintegrationParams> MakeSharedGalileanD(
      double g = 9.81, double omegaBiasSigma = 1e-4, double accBiasSigma = 1e-3) {
    auto params = MakeSharedD(g);
    params->setBiasOmegaCovariance(Matrix3::Identity() * omegaBiasSigma);
    params->setBiasAccCovariance(Matrix3::Identity() * accBiasSigma);
    params->setVirtualVelCovariance(Matrix3::Zero());  // Default: disabled
    params->setVirtualTimeScaleCovariance(0);          // Default: disabled
    return params;
  }

  /// Named constructor for Z-up navigation frame (ENU) with Galilean-specific defaults
  static std::shared_ptr<GalileanPreintegrationParams> MakeSharedGalileanU(
      double g = 9.81, double omegaBiasSigma = 1e-4, double accBiasSigma = 1e-3) {
    auto params = MakeSharedU(g);
    params->setBiasOmegaCovariance(Matrix3::Identity() * omegaBiasSigma);
    params->setBiasAccCovariance(Matrix3::Identity() * accBiasSigma);
    params->setVirtualVelCovariance(Matrix3::Zero());
    params->setVirtualTimeScaleCovariance(0);
    return params;
  }

  /// Print parameters.
  void print(const std::string& s = "GalileanPreintegrationParams") const override;

  /// Check equality. This overrides the correct virtual function from PreintegratedRotationParams.
  bool equals(const PreintegratedRotationParams& other, double tol = 1e-9) const override;

  // Standard Accessors
  const Matrix3& getBiasAccCovariance() const { return biasAccCovariance; }
  const Matrix3& getBiasOmegaCovariance() const { return biasOmegaCovariance; }
  const Matrix6& getBiasAccOmegaInit() const { return biasAccOmegaInt; }

  // Extended Accessors
  const Matrix3& getVirtualVelCovariance() const { return virtualVelCovariance; }
  double getVirtualTimeScaleCovariance() const { return virtualTimeScaleCovariance; }
  const Matrix3& getBiasVirtualVelCovariance() const { return biasVirtualVelCovariance; }
  double getBiasVirtualTimeCovariance() const { return biasVirtualTimeCovariance; }
  const Matrix10& getBiasExtendedInit() const { return biasExtendedInit; }

  // Standard Setters
  void setBiasAccCovariance(const Matrix3& cov) { biasAccCovariance = cov; }
  void setBiasOmegaCovariance(const Matrix3& cov) { biasOmegaCovariance = cov; }
  void setBiasAccOmegaInit(const Matrix6& cov) {
    biasAccOmegaInt = cov;
    // Also update the extended bias matrix for consistency
    biasExtendedInit.block<3,3>(0,0) = cov.block<3,3>(0,0); // Acc bias
    biasExtendedInit.block<3,3>(3,3) = cov.block<3,3>(3,3); // Omega bias
  }

  // Extended Setters
  void setVirtualVelCovariance(const Matrix3& cov) { virtualVelCovariance = cov; }
  void setVirtualTimeScaleCovariance(double cov) { virtualTimeScaleCovariance = cov; }
  void setBiasVirtualVelCovariance(const Matrix3& cov) { biasVirtualVelCovariance = cov; }
  void setBiasVirtualTimeCovariance(double cov) { biasVirtualTimeCovariance = cov; }
  void setBiasExtendedInit(const Matrix10& cov) {
    biasExtendedInit = cov;
    // Also update the standard 6x6 bias matrix for backward compatibility
    biasAccOmegaInt.block<3,3>(0,0) = cov.block<3,3>(0,0); // Acc bias
    biasAccOmegaInt.block<3,3>(3,3) = cov.block<3,3>(3,3); // Omega bias
  }

  /**
   * Creates the full 20x20 continuous-time noise covariance matrix
   * Follows block structure of the reference implementation
   * [0-2]: gyro, [3-5]: acc, [6-8]: virtual vel, [9]: virtual time
   * [10-12]: gyro bias, [13-15]: acc bias, [16-18]: vvel bias, [19]: vtime bias
   */
  Matrix20 createNoiseCovariance() const {
    Matrix20 Qc = Matrix20::Zero();
    // Measurement noise
    Qc.block<3,3>(0,0) = gyroscopeCovariance;
    Qc.block<3,3>(3,3) = accelerometerCovariance;
    Qc.block<3,3>(6,6) = virtualVelCovariance;
    Qc(9,9) = virtualTimeScaleCovariance;
    // Bias random walk noise
    Qc.block<3,3>(10,10) = biasOmegaCovariance;
    Qc.block<3,3>(13,13) = biasAccCovariance;
    Qc.block<3,3>(16,16) = biasVirtualVelCovariance;
    Qc(19,19) = biasVirtualTimeCovariance;
    return Qc;
  }

private:
#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /** Serialization function */
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE& ar, const unsigned int /*version*/) {
    namespace bs = ::boost::serialization;
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(PreintegrationParams);
    ar & BOOST_SERIALIZATION_NVP(biasOmegaCovariance);
    ar & BOOST_SERIALIZATION_NVP(biasAccCovariance);
    ar & BOOST_SERIALIZATION_NVP(biasAccOmegaInt);
    ar & BOOST_SERIALIZATION_NVP(virtualVelCovariance);
    ar & BOOST_SERIALIZATION_NVP(virtualTimeScaleCovariance);
    ar & BOOST_SERIALIZATION_NVP(biasVirtualVelCovariance);
    ar & BOOST_SERIALIZATION_NVP(biasVirtualTimeCovariance);
    ar & BOOST_SERIALIZATION_NVP(biasExtendedInit);
  }
#endif

public:
 GTSAM_MAKE_ALIGNED_OPERATOR_NEW
}; // struct GalileanPreintegrationParams

// Implementation of print/equals
inline void GalileanPreintegrationParams::print(const std::string& s) const {
    PreintegrationParams::print(s);
    std::cout << "  biasOmegaCovariance = \n[" << biasOmegaCovariance << "]" << std::endl;
    std::cout << "  biasAccCovariance = \n[" << biasAccCovariance << "]" << std::endl;
    std::cout << "  biasAccOmegaInt = \n[" << biasAccOmegaInt << "]" << std::endl;
    std::cout << "  virtualVelCovariance = \n[" << virtualVelCovariance << "]" << std::endl;
    std::cout << "  virtualTimeScaleCovariance = " << virtualTimeScaleCovariance << std::endl;
    std::cout << "  biasVirtualVelCovariance = \n[" << biasVirtualVelCovariance << "]" << std::endl;
    std::cout << "  biasVirtualTimeCovariance = " << biasVirtualTimeCovariance << std::endl;
    std::cout << "  biasExtendedInit = \n[" << biasExtendedInit << "]" << std::endl;
}

// This now correctly overrides the virtual function from PreintegratedRotationParams
inline bool GalileanPreintegrationParams::equals(const PreintegratedRotationParams& other, double tol) const {
    // Try to dynamic cast to the specific PreintegrationParams type first
    const auto* otherP = dynamic_cast<const PreintegrationParams*>(&other);
    if (otherP == nullptr) return false; // Cannot compare if it's not even a PreintegrationParams

    // Now try to dynamic cast to GalileanPreintegrationParams
    const auto* otherG = dynamic_cast<const GalileanPreintegrationParams*>(otherP);
    if (otherG == nullptr) return false; // Cannot compare if it's not a GalileanPreintegrationParams

    // Compare base class PreintegrationParams members and then Galilean specific members
    return PreintegrationParams::equals(other, tol) && // Use base class equals for its members
           equal_with_abs_tol(biasOmegaCovariance, otherG->biasOmegaCovariance, tol) &&
           equal_with_abs_tol(biasAccCovariance, otherG->biasAccCovariance, tol) &&
           equal_with_abs_tol(biasAccOmegaInt, otherG->biasAccOmegaInt, tol) &&
           equal_with_abs_tol(virtualVelCovariance, otherG->virtualVelCovariance, tol) &&
           std::abs(virtualTimeScaleCovariance - otherG->virtualTimeScaleCovariance) < tol &&
           equal_with_abs_tol(biasVirtualVelCovariance, otherG->biasVirtualVelCovariance, tol) &&
           std::abs(biasVirtualTimeCovariance - otherG->biasVirtualTimeCovariance) < tol &&
           equal_with_abs_tol(biasExtendedInit, otherG->biasExtendedInit, tol);
}

} // namespace gtsam
