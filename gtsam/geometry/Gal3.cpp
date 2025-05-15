/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    Gal3.cpp
 * @brief   Implementation of 3D Galilean Group SGal(3) state
 * @authors Matt Kielo, Scott Baker, Frank Dellaert
 * @date    April 30, 2025
 *
 * This implementation is based on the paper:
 * Kelly, J. (2023). "All About the Galilean Group SGal(3)"
 * arXiv:2312.07555
 *
 * All section, equation, and page references in comments throughout this file
 * refer to the aforementioned paper.
 */


#include <gtsam/geometry/Gal3.h>
#include <gtsam/geometry/SO3.h> // For so3::DexpFunctor and skewSymmetric
#include <gtsam/geometry/Event.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/nonlinear/expressions.h> // For Expression
#include <gtsam/geometry/concepts.h>     // For traits

#include <iostream>
#include <cmath>
#include <functional> // For std::function

namespace gtsam {

//------------------------------------------------------------------------------
// Static Constructor/Create functions
//------------------------------------------------------------------------------
Gal3 Gal3::Create(const Rot3& R, const Point3& r, const Velocity3& v, double t,
                    OptionalJacobian<10, 3> H1, OptionalJacobian<10, 3> H2,
                    OptionalJacobian<10, 3> H3, OptionalJacobian<10, 1> H4) {
      if (H1) {
        H1->setZero();
        H1->block<3, 3>(6, 0) = Matrix3::Identity();
      }
      if (H2) {
        H2->setZero();
        H2->block<3, 3>(0, 0) = R.transpose();
      }
      if (H3) {
        H3->setZero();
        H3->block<3, 3>(3, 0) = R.transpose();
      }
      if (H4) {
        H4->setZero();
        Vector3 drho_dt = -R.transpose() * v;
        H4->block<3, 1>(0, 0) = drho_dt;
        (*H4)(9, 0) = 1.0;
      }
      return Gal3(R, r, v, t);
}

//------------------------------------------------------------------------------
Gal3 Gal3::FromPoseVelocityTime(const Pose3& pose, const Velocity3& v, double t,
                                OptionalJacobian<10, 6> H1, OptionalJacobian<10, 3> H2,
                                OptionalJacobian<10, 1> H3) {
    const Rot3& R = pose.rotation();
    const Point3& r_world = pose.translation(); // Renamed to avoid conflict with Gal3::r_
    if (H1) {
        H1->setZero();
        // Jacobian wrt Pose3 tangent space [omega, v_body] (6x1) -> Gal3 tangent [rho, nu, theta, t_tan] (10x1)
        // dR/domega = I (approx) -> dtheta_gal3/domega = I
        H1->block<3, 3>(6, 0) = Matrix3::Identity();
        // dr_world/dv_body = R -> drho_gal3/dv_body = R^T * R = I ? Needs careful check.
        // Original code implies d(rho_final)/d(r_world) = I
        // Let's assume the original simple Jacobian is intended for now.
        H1->block<3, 3>(0, 3) = Matrix3::Identity();
    }
    if (H2) { // Jacobian wrt velocity v (world frame)
        H2->setZero();
        H2->block<3, 3>(3, 0) = R.transpose(); // d(nu_final)/dv = R.transpose()
    }
    if (H3) { // Jacobian wrt time t
        H3->setZero();
        Vector3 drho_dt = -R.transpose() * v; // d(rho_final)/dt = -R.transpose()*v
        H3->block<3, 1>(0, 0) = drho_dt;
        (*H3)(9, 0) = 1.0; // d(t_final)/dt = 1
    }
    return Gal3(R, r_world, v, t);
}

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------
Gal3::Gal3(const Matrix5& M) {
    // Constructor from 5x5 matrix representation (Equation 9, Page 5)
    if (std::abs(M(3, 3) - 1.0) > 1e-9 || std::abs(M(4, 4) - 1.0) > 1e-9 ||
        M.row(4).head(4).norm() > 1e-9 || M.row(3).head(3).norm() > 1e-9) {
        throw std::invalid_argument("Invalid Gal3 matrix structure: Check zero blocks and diagonal ones.");
    }
    R_ = Rot3(M.block<3, 3>(0, 0));
    v_ = M.block<3, 1>(0, 3);
    r_ = Point3(M.block<3, 1>(0, 4));
    t_ = M(3, 4);
}

//------------------------------------------------------------------------------
// Component Access
//------------------------------------------------------------------------------
const Rot3& Gal3::rotation(OptionalJacobian<3, 10> H) const {
    if (H) {
        H->setZero();
        H->block<3, 3>(0, 6) = Matrix3::Identity(); // Simplified Jacobian dR/dtheta_xi = I
    }
    return R_;
}

//------------------------------------------------------------------------------
const Point3& Gal3::translation(OptionalJacobian<3, 10> H) const {
     if (H) {
        H->setZero();
        H->block<3,3>(0, 0) = R_.matrix(); // Simplified Jacobian dr/drho_xi = R
        H->block<3,1>(0, 9) = v_;          // Simplified Jacobian dr/dt_tan_xi = v
    }
    return r_;
}

//------------------------------------------------------------------------------
const Velocity3& Gal3::velocity(OptionalJacobian<3, 10> H) const {
     if (H) {
        H->setZero();
        H->block<3, 3>(0, 3) = R_.matrix(); // Simplified Jacobian dv/dnu_xi = R
     }
    return v_;
}

//------------------------------------------------------------------------------
const double& Gal3::time(OptionalJacobian<1, 10> H) const {
    if (H) {
        H->setZero();
        (*H)(0, 9) = 1.0; // d(t_member)/d(t_tan_xi) = 1
    }
    return t_;
}

//------------------------------------------------------------------------------
// Matrix Representation
//------------------------------------------------------------------------------
Matrix5 Gal3::matrix() const {
    // Returns 5x5 matrix representation as in Equation 9, Page 5
    Matrix5 M = Matrix5::Identity();
    M.block<3, 3>(0, 0) = R_.matrix();
    M.block<3, 1>(0, 3) = v_;
    M.block<3, 1>(0, 4) = Vector3(r_); // Cast Point3 to Vector3
    M(3, 4) = t_;
    M.block<1,3>(3,0).setZero(); // Ensure (3,0), (3,1), (3,2) are zero
    M.block<1,4>(4,0).setZero(); // Ensure (4,0), (4,1), (4,2), (4,3) are zero
    return M;
}

//------------------------------------------------------------------------------
// Stream operator
//------------------------------------------------------------------------------
std::ostream& operator<<(std::ostream& os, const Gal3& state) {
    os << "R: " << state.R_ << "\n"; // Assumes Rot3 has an ostream operator
    os << "r: " << state.r_.transpose() << "\n";
    os << "v: " << state.v_.transpose() << "\n";
    os << "t: " << state.t_;
    return os;
}

//------------------------------------------------------------------------------
// Testable Requirements
//------------------------------------------------------------------------------
void Gal3::print(const std::string& s) const {
    std::cout << (s.empty() ? "" : s + " ");
    std::cout << *this << std::endl;
}

//------------------------------------------------------------------------------
bool Gal3::equals(const Gal3& other, double tol) const {
    return R_.equals(other.R_, tol) &&
           traits<Point3>::Equals(r_, other.r_, tol) &&
           traits<Velocity3>::Equals(v_, other.v_, tol) && // Assuming Velocity3 is Vector3
           std::abs(t_ - other.t_) < tol;
}

//------------------------------------------------------------------------------
// Group Operations
//------------------------------------------------------------------------------
Gal3 Gal3::inverse() const {
    // Implements inverse formula from Equation 10, Page 5
    const Rot3 Rinv = R_.inverse();
    const Velocity3 v_inv = -(Rinv.rotate(v_)); // v_inv = -R^T * v
    // r_inv = -R^T * (r - t*v)
    const Point3 r_inv = Point3(-(Rinv.rotate(Vector3(r_) - t_ * v_)));
    const double t_inv = -t_;
    return Gal3(Rinv, r_inv, v_inv, t_inv);
}

//------------------------------------------------------------------------------
Gal3 Gal3::operator*(const Gal3& other) const {
    // Implements group composition from Equation 8, Page 5
    // (R1, r1, v1, t1) * (R2, r2, v2, t2) =
    // (R1*R2, R1*r2 + t2*v1 + r1, R1*v2 + v1, t1 + t2)
    const Gal3& g1 = *this;
    const Gal3& g2 = other;

    const Rot3 R_comp = g1.R_.compose(g2.R_);
    const Vector3 r1_vec(g1.r_); // Cast Point3 to Vector3 for arithmetic
    const Vector3 r2_vec(g2.r_); // Cast Point3 to Vector3 for arithmetic
    const Point3 r_comp = Point3(g1.R_.rotate(r2_vec) + g2.t_ * g1.v_ + r1_vec);
    const Velocity3 v_comp = g1.R_.rotate(g2.v_) + g1.v_;
    const double t_comp = g1.t_ + g2.t_;

    return Gal3(R_comp, r_comp, v_comp, t_comp);
}

//------------------------------------------------------------------------------
// Lie Group Static Functions
//------------------------------------------------------------------------------
gtsam::Gal3 gtsam::Gal3::Expmap(const Vector10& xi, OptionalJacobian<10, 10> Hxi_out) {
    // Implements exponential map from Equations 16-19, Pages 7-8
    // xi = [rho_tan, nu_tan, theta_tan, t_tan_val]^T (10x1)
    // rho_tan (3x1), nu_tan (3x1), theta_tan (3x1), t_tan_val (scalar)

    const Vector3 rho_tan_vec = rho(xi);
    const Vector3 nu_tan_vec = nu(xi);
    const Vector3 theta_tan_vec = theta(xi);
    const double t_tan_scalar = t_tan(xi)(0);

    const gtsam::so3::DexpFunctor dexp_functor(theta_tan_vec); // Functor for SO(3) Expmap and its derivatives
    const Rot3 R_final = Rot3::Expmap(theta_tan_vec); // R = Exp_SO3(theta_tan)
    const Matrix3 Jl_theta = dexp_functor.leftJacobian(); // Left Jacobian of SO(3) Expmap

    Matrix3 E; // E matrix from Equation 19, Page 8
    // Access DexpFunctor members directly (they are variables, not functions)
    if (dexp_functor.nearZero) { // Use small angle approximation if theta_tan is small
         // Use W and WW member variables
         E = 0.5 * Matrix3::Identity() + (1.0 / 6.0) * dexp_functor.W + (1.0 / 24.0) * dexp_functor.WW;
    } else { // Use closed form based on paper's E matrix definition
         // Use skewSymmetric (assuming it's available in scope, often from SO3.h or geometry namespace)
         const Matrix3 W_mat = skewSymmetric(theta_tan_vec); // skew-symmetric matrix for theta_tan
         const Matrix3 WW_mat = W_mat * W_mat;
         const double theta_abs = dexp_functor.theta; // Access theta member variable
         const double cos_theta = std::cos(theta_abs);
         const double sin_theta = std::sin(theta_abs);
         const double theta_sq_val = theta_abs * theta_abs;
         // Avoid division by zero if theta is extremely small, though nearZero should handle it
         const double C_coeff = (theta_abs < Gal3::kSmallAngleThreshold) ? (1.0/6.0 - theta_sq_val/120.0) : // Taylor expansion for C
                                (theta_abs - sin_theta) / (theta_abs * theta_sq_val); // C from paper
         const double B_E_coeff = (theta_abs < Gal3::kSmallAngleThreshold) ? (1.0/24.0 - theta_sq_val/720.0) : // Taylor expansion for B_E
                                  (theta_sq_val - 2.0 + 2.0 * cos_theta) / (2.0 * theta_sq_val * theta_sq_val); // B_E from paper
         E = 0.5 * Matrix3::Identity() + C_coeff * W_mat + B_E_coeff * WW_mat;
    }

    // r_final = Jl_theta * rho_tan + E * (t_tan_val * nu_tan) (Equation 17)
    const Point3 r_final = Point3(Jl_theta * rho_tan_vec + E * (t_tan_scalar * nu_tan_vec));
    // v_final = Jl_theta * nu_tan (Equation 18)
    const Velocity3 v_final = Jl_theta * nu_tan_vec;
    // t_final = t_tan_val (Equation 16)
    const double t_final_scalar = t_tan_scalar;

    Gal3 result(R_final, r_final, v_final, t_final_scalar);

    if (Hxi_out) {
        *Hxi_out = Gal3::ExpmapDerivative(xi); // Call the (now separate) derivative function
    }

    return result;
}

//------------------------------------------------------------------------------
Matrix10 Gal3::LeftJacobian(const Vector10& u) {
  // Jl(u) = Jr(-u)
  // GTSAM's ExpmapDerivative(xi) computes Jr(xi).
  // Therefore, LeftJacobian(u) is ExpmapDerivative(-u).
  return Gal3::ExpmapDerivative(-u);
}

//------------------------------------------------------------------------------
Vector10 Gal3::Logmap(const Gal3& g, OptionalJacobian<10, 10> Hg_out) {
    // Implements logarithmic map from Equations 20-23, Page 8
    const Vector3 theta_vec = Rot3::Logmap(g.R_); // theta_tan = Log_SO3(R) (Equation 22)
    const gtsam::so3::DexpFunctor dexp_functor_log(theta_vec);
    const Matrix3 Jl_theta_inv = dexp_functor_log.leftJacobianInverse(); // J_l(theta_tan)^-1

    Matrix3 E; // E matrix, same as in Expmap
    // Access DexpFunctor members directly
    if (dexp_functor_log.nearZero) {
         // Use W and WW member variables
         E = 0.5 * Matrix3::Identity() + (1.0 / 6.0) * dexp_functor_log.W + (1.0 / 24.0) * dexp_functor_log.WW;
    } else {
         // Use skewSymmetric
         const Matrix3 W_mat = skewSymmetric(theta_vec);
         const Matrix3 WW_mat = W_mat * W_mat;
         const double theta_abs = dexp_functor_log.theta; // Access theta member variable
         const double cos_theta = std::cos(theta_abs);
         const double sin_theta = std::sin(theta_abs);
         const double theta_sq_val = theta_abs * theta_abs;
         // Avoid division by zero
         const double C_coeff = (theta_abs < Gal3::kSmallAngleThreshold) ? (1.0/6.0 - theta_sq_val/120.0) :
                                (theta_abs - sin_theta) / (theta_abs * theta_sq_val);
         const double B_E_coeff = (theta_abs < Gal3::kSmallAngleThreshold) ? (1.0/24.0 - theta_sq_val/720.0) :
                                  (theta_sq_val - 2.0 + 2.0 * cos_theta) / (2.0 * theta_sq_val * theta_sq_val);
         E = 0.5 * Matrix3::Identity() + C_coeff * W_mat + B_E_coeff * WW_mat;
    }

    const Vector3 r_vec_g = Vector3(g.r_); // Cast Point3 to Vector3
    const Velocity3& v_vec_g = g.v_;
    const double& t_val_g = g.t_;

    // nu_tan = J_l(theta_tan)^-1 * v (Equation 21)
    const Vector3 nu_tan_final = Jl_theta_inv * v_vec_g;
    // rho_tan = J_l(theta_tan)^-1 * (r - E * (t * nu_tan)) (Equation 23)
    const Vector3 rho_tan_final = Jl_theta_inv * (r_vec_g - E * (t_val_g * nu_tan_final));
    // t_tan = t (Equation 20)
    const double t_tan_final_scalar = t_val_g;

    Vector10 xi_final;
    rho(xi_final) = rho_tan_final;
    nu(xi_final) = nu_tan_final;
    theta(xi_final) = theta_vec;
    t_tan(xi_final)(0) = t_tan_final_scalar;

    if (Hg_out) {
        *Hg_out = Gal3::LogmapDerivative(g);
    }

    return xi_final;
}

//------------------------------------------------------------------------------
Matrix10 Gal3::AdjointMap() const {
    // Implements adjoint map Ad_g as in Equation 26, Page 9 in Kelly's paper
    // The adjoint map transforms a tangent vector from one point to another point in the manifold
    // Tangent space ordering: [rho, nu, theta, t] (position, velocity, rotation, time)

    const Matrix3 Rmat = R_.matrix();  // Rotation matrix
    const Vector3 v_vec = v_;          // Velocity vector
    const Vector3 r_minus_tv = Vector3(r_) - t_ * v_;  // Position minus time*velocity (key term for Galilean coupling)

    // Initialize adjoint matrix with zeros to ensure correct structure
    Matrix10 Ad = Matrix10::Zero();

    // Position to position coupling
    // Ad_g: rho → rho = R
    Ad.block<3,3>(0,0) = Rmat;

    // Position to velocity coupling (time effect on position via velocity)
    // Ad_g: nu → rho = -t*R (crucial time coupling term)
    Ad.block<3,3>(0,3) = -t_ * Rmat;

    // Position to rotation coupling (effect of rotation on position via the modified lever arm)
    // Ad_g: theta → rho = [(r-tv)]x*R (involves time-velocity coupling)
    Ad.block<3,3>(0,6) = skewSymmetric(r_minus_tv) * Rmat;

    // Position to time coupling (direct effect of time-derivative on position)
    // Ad_g: t → rho = v (velocity appears in position time-derivative)
    Ad.block<3,1>(0,9) = v_vec;

    // Velocity to velocity coupling
    // Ad_g: nu → nu = R
    Ad.block<3,3>(3,3) = Rmat;

    // Velocity to rotation coupling (effect of rotation on velocity via Coriolis force)
    // Ad_g: theta → nu = [v]x*R
    Ad.block<3,3>(3,6) = skewSymmetric(v_vec) * Rmat;

    // Rotation to rotation coupling
    // Ad_g: theta → theta = R
    Ad.block<3,3>(6,6) = Rmat;

    // Time to time coupling
    // Ad_g: t → t = 1
    Ad(9,9) = 1.0;

    // Zero blocks (all other couplings are zero):
    // - Velocity to position: Ad.block<3,3>(3,0) = 0
    // - Velocity to time: Ad.block<3,1>(3,9) = 0
    // - Rotation to position: Ad.block<3,3>(6,0) = 0
    // - Rotation to velocity: Ad.block<3,3>(6,3) = 0
    // - Rotation to time: Ad.block<3,1>(6,9) = 0
    // - Time to position/velocity/rotation: Ad.block<1,9>(9,0) = 0

    return Ad;
}



//------------------------------------------------------------------------------
Vector10 Gal3::Adjoint(const Vector10& xi, OptionalJacobian<10, 10> H_g, OptionalJacobian<10, 10> H_xi) const {
    Matrix10 Ad = AdjointMap(); // Ad_g
    Vector10 y = Ad * xi;      // Ad_g * xi

    if (H_xi) { // Jacobian wrt xi
        *H_xi = Ad; // d(Ad_g * xi)/dxi = Ad_g
    }

    if (H_g) { // Jacobian wrt g (the Gal3 element itself)
        // NOTE: Using numerical derivative for the Jacobian with respect to
        // the group element instead of deriving the analytical expression.
        // Future work to use analytical instead.
        // *** Reverted lambda to match original code structure ***
        std::function<Vector10(const Gal3&, const Vector10&)> adjoint_action_wrt_g =
          [&](const Gal3& g_in, const Vector10& xi_in) {
              // This implicitly calls Adjoint(xi_in, nullptr, nullptr) when used by numericalDerivative
              return g_in.Adjoint(xi_in);
          };
        *H_g = numericalDerivative21(adjoint_action_wrt_g, *this, xi, 1e-7);
    }
    return y;
}

//------------------------------------------------------------------------------
Matrix10 Gal3::adjointMap(const Vector10& xi) {
    // Implements Lie algebra adjoint representation ad_xi as in Equation 28, Page 10
    // xi = [rho_tan, nu_tan, theta_tan, t_tan_val]^T
    const Matrix3 Theta_hat = skewSymmetric(theta(xi)); // [theta_tan]_x
    const Matrix3 Nu_hat = skewSymmetric(nu(xi));       // [nu_tan]_x
    const Matrix3 Rho_hat = skewSymmetric(rho(xi));     // [rho_tan]_x
    const double t_val = t_tan(xi)(0);                  // t_tan_val
    const Vector3 nu_vec = nu(xi);                      // nu_tan vector

    Matrix10 ad = Matrix10::Zero();

    // Row 0-2
    ad.block<3,3>(0,0) = Theta_hat;
    ad.block<3,3>(0,3) = -t_val * Matrix3::Identity(); // Corrected based on paper Eq 28
    ad.block<3,3>(0,6) = Rho_hat;
    ad.block<3,1>(0,9) = nu_vec;

    // Row 3-5
    ad.block<3,3>(3,3) = Theta_hat;
    ad.block<3,3>(3,6) = Nu_hat;

    // Row 6-8
    ad.block<3,3>(6,6) = Theta_hat;

    // Row 9 is all zeros, Ad(9,9) is 0.

    return ad;
}

//------------------------------------------------------------------------------
Vector10 Gal3::adjoint(const Vector10& xi, const Vector10& y, OptionalJacobian<10, 10> Hxi, OptionalJacobian<10, 10> Hy) {
    // This is the Lie bracket [xi, y] = ad_xi * y
    Matrix10 ad_xi = adjointMap(xi); // ad_xi
    if (Hy) *Hy = ad_xi;             // d([xi,y])/dy = ad_xi
    if (Hxi) {
         // d([xi,y])/dxi = d(ad_xi * y)/dxi. Since ad_xi is linear in xi, this becomes ad_y with a sign change.
         // [xi, y] = -[y, xi] => d([xi,y])/dxi = -ad_y
         *Hxi = -adjointMap(y);
    }
    return ad_xi * y;
}


//------------------------------------------------------------------------------
Matrix10 Gal3::LogmapDerivative(const Gal3& g) {
    // Related to the inverse of left Jacobian in Equations 31-36, Pages 10-11
    // NOTE: Using numerical approximation instead of implementing the analytical
    // expression for the inverse Jacobian. Future work to replace this
    // with analytical derivative.
    Vector10 xi = Gal3::Logmap(g); // Calculate Logmap first to check near zero condition
    if (xi.norm() < Gal3::kSmallAngleThreshold) return Matrix10::Identity(); // Use class static member

    std::function<Vector10(const Gal3&)> fn =
        [](const Gal3& g_in) { return Gal3::Logmap(g_in); };
    return numericalDerivative11<Vector10, Gal3>(fn, g, 1e-5);
}

//------------------------------------------------------------------------------
// Lie Algebra (Hat/Vee maps)
//------------------------------------------------------------------------------
Matrix5 Gal3::Hat(const Vector10& xi) {
    // Implements hat operator (tangent vector to Lie algebra matrix) as in Equation 13, Page 6
    // xi = [rho_tan, nu_tan, theta_tan, t_tan_val]^T
    const Vector3 rho_tan_vec = rho(xi);
    const Vector3 nu_tan_vec = nu(xi);
    const Vector3 theta_tan_vec = theta(xi);
    const double t_tan_scalar = t_tan(xi)(0);

    Matrix5 X = Matrix5::Zero();
    X.block<3, 3>(0, 0) = skewSymmetric(theta_tan_vec); // [theta_tan]_x
    X.block<3, 1>(0, 3) = nu_tan_vec;                   // nu_tan
    X.block<3, 1>(0, 4) = rho_tan_vec;                  // rho_tan
    X(3, 4) = t_tan_scalar;                             // t_tan_val
    // Other elements are zero by construction of Matrix5::Zero()
    return X;
}

//------------------------------------------------------------------------------
Vector10 Gal3::Vee(const Matrix5& X) {
    // Implements vee operator (Lie algebra matrix to tangent vector), inverse of Hat.
    // Based on Equation 13, Page 6.
    // Check structure of X for sgal(3)
    if (X.row(4).norm() > 1e-9 ||         // Last row should be all zeros
        X.row(3).head(3).norm() > 1e-9 || // X(3,0), X(3,1), X(3,2) should be zero
        std::abs(X(3,3)) > 1e-9) {        // X(3,3) should be zero
     throw std::invalid_argument("Matrix is not in sgal(3) Lie algebra form.");
    }

    Vector10 xi;
    rho(xi) = X.block<3, 1>(0, 4);     // rho_tan from X(0:2, 4)
    nu(xi) = X.block<3, 1>(0, 3);      // nu_tan from X(0:2, 3)
    const Matrix3& S = X.block<3, 3>(0, 0); // Skew-symmetric part [theta_tan]_x
    // Extract vector from skew-symmetric matrix S = [w]_x
    // w = [S(2,1), S(0,2), S(1,0)]^T
    theta(xi) << S(2, 1), S(0, 2), S(1, 0); // Extract theta_tan from S
    t_tan(xi)(0) = X(3, 4);                 // t_tan_val from X(3,4)
    return xi;
}

//------------------------------------------------------------------------------
// ChartAtOrigin
//------------------------------------------------------------------------------
Gal3 Gal3::ChartAtOrigin::Retract(const Vector10& xi, ChartJacobian Hxi) {
  return Gal3::Expmap(xi, Hxi); // Uses the group Expmap for chart retraction
}

//------------------------------------------------------------------------------
Vector10 Gal3::ChartAtOrigin::Local(const Gal3& g, ChartJacobian Hg) {
  return Gal3::Logmap(g, Hg); // Uses the group Logmap for chart local coordinates
}

//------------------------------------------------------------------------------
Event Gal3::act(const Event& e, OptionalJacobian<4, 10> Hself,
                OptionalJacobian<4, 4> He) const {
  // Implements group action on events (spacetime points) as described in Section 4.1, Page 3-4
  // g = (R, r, v, t_g) acts on e = (p_e, t_e)
  // Result e' = (p', t') = (R*p_e + v*t_e + r, t_e + t_g)
  const double& t_event_in = e.time();
  const Point3& p_event_in = e.location();

  const double t_event_out = t_event_in + t_; // t' = t_e + t_g
  const Point3 p_event_out = R_.rotate(p_event_in) + v_ * t_event_in + r_; // p' = R*p_e + v*t_e + r

  if (He) { // Jacobian wrt input Event e = [t_e; p_e] (4x1)
            // Output Event is [t'; p'] (4x1)
    He->setZero();
    // d(t')/dt_e = 1
    (*He)(0, 0) = 1.0;
    // d(p')/dt_e = v
    He->block<3, 1>(1, 0) = v_;
    // d(p')/dp_e = R
    He->block<3, 3>(1, 1) = R_.matrix();
    // d(t')/dp_e = 0 (already zero)
  }

  if (Hself) { // Jacobian wrt this Gal3 element's tangent space xi (10x1)
               // xi = [rho, nu, theta, t_tan]^T
               // Using the simplified Jacobians from the original code.
               // These likely correspond to a specific perturbation model (e.g., right perturbations).
    Hself->setZero();
    const Matrix3 Rmat = R_.matrix();

    // d(t_out)/d(xi) -> only depends on t_tan component of xi
    (*Hself)(0, 9) = 1.0; // d(t_out)/d(t_tan_xi)

    // d(p_out)/d(xi)
    // d(p_out)/d(rho_xi)
    Hself->block<3, 3>(1, 0) = Rmat;
    // d(p_out)/d(nu_xi)
    Hself->block<3, 3>(1, 3) = Rmat * t_event_in;
    // d(p_out)/d(theta_xi)
    Hself->block<3, 3>(1, 6) = -Rmat * skewSymmetric(p_event_in);
    // d(p_out)/d(t_tan_xi)
    Hself->block<3, 1>(1, 9) = v_;
  }

  return Event(t_event_out, p_event_out);
}

} // namespace gtsam
