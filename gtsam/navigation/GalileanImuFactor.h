/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   GalileanImuFactor.h
 * @brief  Factor for relating two states based on preintegrated Galilean IMU measurements.
 * @author Your Name, based on Delama et al., RA-L 2024 and ImuFactor.h
 */

#pragma once

#include <gtsam/navigation/PreintegratedGalileanMeasurements.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>          // Use Vector (specifically Vector3)
#include <gtsam/navigation/ImuBias.h> // Using ConstantBias
#include <gtsam/navigation/NavState.h> // Needed for PIM interface types
#include <gtsam/base/Matrix.h>         // Include for Matrix type

#include <boost/optional.hpp> // Include for boost::optional used by PIM

namespace gtsam {

/**
 * A factor relating two Pose3 states, two Vector3 velocities (linear velocity),
 * and two ConstantBias states based on preintegrated IMU measurements
 * using the Galilean Preintegration formulation from Delama et al., RA-L 2024.
 *
 * This factor relies on the PreintegratedGalileanMeasurements class, which
 * performs the actual integration according to Algorithm 1 of the paper.
 * The factor's error is a 9D vector representing the mismatch in rotation,
 * position, and velocity between the states, compared against the bias-corrected
 * preintegrated measurement.
 *
 * The noise model for this factor is derived from the 9x9
 * NavState covariance block provided by PreintegratedGalileanMeasurements::preintegratedNavStateCovariance().
 * This 9x9 block implicitly includes the effects of the full 20D Galilean covariance propagation.
 *
 * @param key_pose_i Key for state i Pose3 (X(i))
 * @param key_vel_i Key for state i Velocity (Vector3, V(i))
 * @param key_bias_i Key for state i IMU Bias (B(i))
 * @param key_pose_j Key for state j Pose3 (X(j))
 * @param key_vel_j Key for state j Velocity (Vector3, V(j))
 * @param key_bias_j Key for state j IMU Bias (B(j)) - Note: bias_j is not used in error calculation but included for compatibility/potential extensions.
 * @param pim Preintegrated measurements instance (PreintegratedGalileanMeasurements)
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
                    const PreintegratedGalileanMeasurements& pim); // Implementation in .cpp

  /// Default destructor
  ~GalileanImuFactor() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override; // Implementation in .cpp

  /// @name Testable
  /// @{

  /// Print the factor's details
  void print(const std::string& s = "GalileanImuFactor",
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override; // Implementation in .cpp

  /// Check equality with another factor
  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override; // Implementation in .cpp
  /// @}

  /// @name Factor interface
  /// @{

  /**
   * Evaluate the 9-dimensional error between predicted states (based on state i,
   * bias i, and PIM) and the measured state j.
   * This function calls PIM.computeErrorAndJacobians.
   * Error = [rotation error(3), position error(3), velocity error(3)]
   *
   * @param pose_i State i Pose3
   * @param vel_i State i Velocity (Vector3)
   * @param bias_i State i Bias (imuBias::ConstantBias)
   * @param pose_j State j Pose3
   * @param vel_j State j Velocity (Vector3)
   * @param bias_j State j Bias (imuBias::ConstantBias) - Not used in calculation.
   * @param H1 Optional Jacobian output matrix for error wrt pose_i (9x6)
   * @param H2 Optional Jacobian output matrix for error wrt vel_i  (9x3)
   * @param H3 Optional Jacobian output matrix for error wrt bias_i (9x6)
   * @param H4 Optional Jacobian output matrix for error wrt pose_j (9x6)
   * @param H5 Optional Jacobian output matrix for error wrt vel_j  (9x3)
   * @param H6 Optional Jacobian output matrix for error wrt bias_j (9x6) - Always Zero.
   * @return 9-dimensional error vector (rotation, position, velocity)
   *
   * **NOTE:** Jacobians H1-H5 are currently computed numerically within the PIM class.
   */
   Vector evaluateError(const Pose3& pose_i, const Vector3& vel_i, const imuBias::ConstantBias& bias_i,
                        const Pose3& pose_j, const Vector3& vel_j, const imuBias::ConstantBias& bias_j,
                        OptionalMatrixType H1 = nullptr, // Jacobian wrt pose_i
                        OptionalMatrixType H2 = nullptr, // Jacobian wrt vel_i
                        OptionalMatrixType H3 = nullptr, // Jacobian wrt bias_i
                        OptionalMatrixType H4 = nullptr, // Jacobian wrt pose_j
                        OptionalMatrixType H5 = nullptr, // Jacobian wrt vel_j
                        OptionalMatrixType H6 = nullptr) const override; // Jacobian wrt bias_j


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
    // Serialize the base class
    ar & boost::serialization::make_nvp("NoiseModelFactorN",
        boost::serialization::base_object<Base>(*this));
    // Serialize the PIM object
    ar & BOOST_SERIALIZATION_NVP(_PIM);
  }
#endif

public:
 // Macro required for classes with fixed-size Eigen matrices passed by value
 GTSAM_MAKE_ALIGNED_OPERATOR_NEW

}; // \class GalileanImuFactor

} /// namespace gtsam
