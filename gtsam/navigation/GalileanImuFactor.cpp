/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2025, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   GalileanImuFactor.cpp
 * @brief  Implementation file for GalileanImuFactor.
 * @author Your Name, based on Delama et al., RA-L 2024 and ImuFactor.cpp
 */

#include <gtsam/navigation/GalileanImuFactor.h>
#include <gtsam/base/Matrix.h> // Include for Matrix::Zero
#include <ostream>
#include <stdexcept> // For std::invalid_argument

namespace gtsam {

//------------------------------------------------------------------------------
// Constructor - Now takes PIM directly and constructs noise model from it.
GalileanImuFactor::GalileanImuFactor(Key key_pose_i, Key key_vel_i, Key key_bias_i,
                                     Key key_pose_j, Key key_vel_j, Key key_bias_j,
                                     const PreintegratedGalileanMeasurements& pim) :
    // Construct base class with the 9x9 covariance from the PIM.
    // Use 'true' for smart noise model sharing if appropriate.
    Base(noiseModel::Gaussian::Covariance(pim.preintegratedNavStateCovariance(), true /* smart */),
         key_pose_i, key_vel_i, key_bias_i, key_pose_j, key_vel_j, key_bias_j),
    _PIM(pim) // Store the PIM object itself
{
    // Noise model dimension check is implicitly handled by base class constructor
    // using the 9x9 covariance from PIM.
}

//------------------------------------------------------------------------------
// clone
gtsam::NonlinearFactor::shared_ptr GalileanImuFactor::clone() const {
  // Create a new object copy on the heap
  return std::static_pointer_cast<NonlinearFactor>(
      NonlinearFactor::shared_ptr(new This(*this)));
}

//------------------------------------------------------------------------------
// print
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

//------------------------------------------------------------------------------
// equals
bool GalileanImuFactor::equals(const NonlinearFactor& expected, double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  // Check base class equality (keys, noise model) and PIM equality
  return e != nullptr && Base::equals(*e, tol) && _PIM.equals(e->_PIM, tol);
}

//------------------------------------------------------------------------------
// evaluateError - Delegates computation to the PIM class.
Vector GalileanImuFactor::evaluateError(const Pose3& pose_i, const Vector3& vel_i, const imuBias::ConstantBias& bias_i,
                                       const Pose3& pose_j, const Vector3& vel_j, const imuBias::ConstantBias& bias_j,
                                       OptionalMatrixType H1, OptionalMatrixType H2, // H wrt pose_i, vel_i
                                       OptionalMatrixType H3,                       // H wrt bias_i
                                       OptionalMatrixType H4, OptionalMatrixType H5, // H wrt pose_j, vel_j
                                       OptionalMatrixType H6) const {              // H wrt bias_j (always zero)

    // Use boost::optional<Matrix&> to wrap the raw pointers for the PIM call
    boost::optional<Matrix&> pimH_pose_i = H1 ? boost::optional<Matrix&>(*H1) : boost::none;
    boost::optional<Matrix&> pimH_vel_i  = H2 ? boost::optional<Matrix&>(*H2) : boost::none;
    // Note the index swap: Factor H3 maps to PIM H5 (bias_i)
    boost::optional<Matrix&> pimH_bias_i = H3 ? boost::optional<Matrix&>(*H3) : boost::none;
    // Factor H4 maps to PIM H3 (pose_j)
    boost::optional<Matrix&> pimH_pose_j = H4 ? boost::optional<Matrix&>(*H4) : boost::none;
    // Factor H5 maps to PIM H4 (vel_j)
    boost::optional<Matrix&> pimH_vel_j  = H5 ? boost::optional<Matrix&>(*H5) : boost::none;

    // Call the PIM method which calculates the 9D error and Jacobians
    // Note the Jacobian mapping: H1->H1, H2->H2, H3->H5(bias_i), H4->H3(pose_j), H5->H4(vel_j)
    Vector9 error = _PIM.computeErrorAndJacobians(pose_i, vel_i, pose_j, vel_j, bias_i,
                                                pimH_pose_i, pimH_vel_i,
                                                pimH_pose_j, pimH_vel_j,
                                                pimH_bias_i);

    // Jacobian H6 (wrt bias_j) is known to be zero for this factor type.
    if (H6) {
        *H6 = Matrix::Zero(9, 6);
    }

    return error;
}


}
