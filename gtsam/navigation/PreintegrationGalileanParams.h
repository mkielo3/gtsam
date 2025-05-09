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

  /// Default constructor, initializes with identity matrices.
  GalileanPreintegrationParams()
      : PreintegrationParams(), // Use default base constructor (Z-up gravity)
        biasOmegaCovariance(I_3x3),
        biasAccCovariance(I_3x3),
        biasAccOmegaInt(I_6x6) {}

  /// Constructor initializes with gravity and optional bias random walk sigmas.
  GalileanPreintegrationParams(const Vector3& n_gravity_)
      : PreintegrationParams(n_gravity_), // Initialize base class
        biasOmegaCovariance(I_3x3),
        biasAccCovariance(I_3x3),
        biasAccOmegaInt(I_6x6) {}

  /// Named constructor for Z-down navigation frame (NED).
  static std::shared_ptr<GalileanPreintegrationParams> MakeSharedD(
      double g = 9.81) {
    return std::shared_ptr<GalileanPreintegrationParams>(
        new GalileanPreintegrationParams(Vector3(0, 0, g)));
  }

  /// Named constructor for Z-up navigation frame (ENU).
  static std::shared_ptr<GalileanPreintegrationParams> MakeSharedU(
      double g = 9.81) {
    return std::shared_ptr<GalileanPreintegrationParams>(
        new GalileanPreintegrationParams(Vector3(0, 0, -g)));
  }

  /// Print parameters.
  void print(const std::string& s = "GalileanPreintegrationParams") const override;

  /// Check equality. This overrides the correct virtual function from PreintegratedRotationParams.
  bool equals(const PreintegratedRotationParams& other, double tol = 1e-9) const override;

  // Accessors
  const Matrix3& getBiasAccCovariance() const { return biasAccCovariance; }
  const Matrix3& getBiasOmegaCovariance() const { return biasOmegaCovariance; }
  const Matrix6& getBiasAccOmegaInit() const { return biasAccOmegaInt; }

  // Setters
  void setBiasAccCovariance(const Matrix3& cov) { biasAccCovariance = cov; }
  void setBiasOmegaCovariance(const Matrix3& cov) { biasOmegaCovariance = cov; }
  void setBiasAccOmegaInit(const Matrix6& cov) { biasAccOmegaInt = cov; }

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
           equal_with_abs_tol(biasAccOmegaInt, otherG->biasAccOmegaInt, tol);
}


} // namespace gtsam
