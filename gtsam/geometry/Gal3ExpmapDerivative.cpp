/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    Gal3ExpmapDerivative.cpp
 * @brief   Implementation of ExpmapDerivative for Gal3
 * @authors Matt Kielo, Scott Baker, Frank Dellaert
 * @date    April 30, 2025
 */

#include <gtsam/geometry/Gal3.h>
#include <gtsam/geometry/SO3.h> // For so3::DexpFunctor and skewSymmetric
#include <cmath>
// #include <iostream> // Removed as std::cout is no longer used

namespace gtsam {

//------------------------------------------------------------------------------
// Helper functions for Jacobian computation based on LiePlus++ implementation
//------------------------------------------------------------------------------
static Matrix3 computeGal3LeftJacobianQ1(const Vector3& w, const Vector3& v) {
    static constexpr double eps = 1.0e-9;  // Small angle threshold

    Matrix3 p = skewSymmetric(w);
    Matrix3 r = skewSymmetric(v);

    double ang = w.norm();
    if (ang < eps) {
        // Small angle approximation
        return 0.5 * r + (1.0/6.0) * (p * r + r * p + p * r * p);
    }

    double s = std::sin(ang);
    double c = std::cos(ang);

    double ang_p2 = ang * ang;
    double ang_p3 = ang_p2 * ang;
    double ang_p4 = ang_p3 * ang;
    double ang_p5 = ang_p4 * ang;

    double c1 = (ang - s) / ang_p3;
    double c2 = (0.5 * ang_p2 + c - 1.0) / ang_p4;
    double c3 = (ang * (1.0 + 0.5 * c) - 1.5 * s) / ang_p5;

    Matrix3 m1 = p * r + r * p + p * r * p;
    Matrix3 m2 = p * p * r + r * p * p - 3.0 * p * r * p;
    Matrix3 m3 = p * r * p * p + p * p * r * p;

    return 0.5 * r + c1 * m1 + c2 * m2 + c3 * m3;
}

//------------------------------------------------------------------------------
static Matrix3 computeGal3LeftJacobianQ2(const Vector3& w, const Vector3& v) {
    static constexpr double eps = 1.0e-9;  // Small angle threshold

    Matrix3 p = skewSymmetric(w);
    Matrix3 r = skewSymmetric(v);

    double ang = w.norm();
    if (ang < eps) {
        // Small angle approximation
        return (1.0/6.0) * r + (1.0/24.0) * (p * r + r * p);
    }

    double s = std::sin(ang);
    double c = std::cos(ang);

    double ang_p2 = pow(ang, 2);
    double ang_p3 = ang_p2 * ang;
    double ang_p4 = ang_p3 * ang;
    double ang_p5 = ang_p4 * ang;
    double ang_p6 = ang_p5 * ang;
    double ang_p7 = ang_p6 * ang;

    double c0 = 1.0 / 6.0;
    double c1 = (0.5 * ang_p2 + c - 1.0) / ang_p4;
    double c2 = (c0 * ang_p3 - ang + s) / ang_p5;
    double c3 = -(2.0 * c + ang * s - 2.0) / ang_p4;
    double c4 = (c0 * ang_p3 + ang * c + ang - 2.0 * s) / ang_p5;
    double c5 = -(0.75 * ang * c + (0.25 * ang_p2 - 0.75) * s) / ang_p5;
    double c6 = (0.25 * ang * c + 0.5 * ang - 0.75 * s) / ang_p5;
    double c7 = ((0.25 * ang_p2 - 2.0) * c - 1.25 * ang * s + 2.0) / ang_p6;
    double c8 = (c0 * ang_p3 + 1.25 * ang * c + (0.25 * ang_p2 - 1.25) * s) / ang_p7;

    Matrix3 m1 = r * p;
    Matrix3 m2 = r * p * p;
    Matrix3 m3 = p * r;
    Matrix3 m4 = p * p * r;
    Matrix3 m5 = p * r * p;
    Matrix3 m6 = p * p * r * p;
    Matrix3 m7 = p * r * p * p;
    Matrix3 m8 = p * p * r * p * p;

    return c0 * r + c1 * m1 + c2 * m2 + c3 * m3 + c4 * m4 + c5 * m5 + c6 * m6 + c7 * m7 + c8 * m8;
}

//------------------------------------------------------------------------------
static Matrix3 computeGal3LeftJacobianU1(const Vector3& w) {
    static constexpr double eps = 1.0e-9;  // Small angle threshold

    double ang = w.norm();
    if (ang < eps) {
        return 0.5 * Matrix3::Identity() + (1.0/3.0) * skewSymmetric(w);
    }

    Vector3 ax = w / ang;
    double ang_p2 = pow(ang, 2);
    double s = std::sin(ang);
    double c = std::cos(ang);
    double c1 = (ang * s + c - 1.0) / ang_p2;
    double c2 = (s - ang * c) / ang_p2;

    return c1 * Matrix3::Identity() + c2 * skewSymmetric(ax) + (0.5 - c1) * ax * ax.transpose();
}

//------------------------------------------------------------------------------
Matrix10 Gal3::ExpmapDerivative(const Vector10& xi) {
    // Implementation of right Jacobian Jr(xi) for Expmap.
    // Based on LiePlus++ implementation of leftJacobian(u) with u = -xi
    // Jr(xi) = Jl(-xi)

    // Check for small angle approximation
    // if (xi.norm() < Gal3::kSmallAngleThreshold) {
    //     return Matrix10::Identity();
    // }

    // Compute the right Jacobian Jr(xi) = Jl(-xi)
    const Vector10 minus_xi = -xi;

    // Extract components from -xi for the left Jacobian Jl(-xi)
    const Vector3 theta_vec = theta(minus_xi);  // theta component of -xi
    const Vector3 nu_vec = nu(minus_xi);        // nu component of -xi
    const Vector3 rho_vec = rho(minus_xi);      // rho component of -xi
    const double t_val = t_tan(minus_xi)(0);    // t_tan component of -xi

    // Create Jacobian matrix
    Matrix10 Jr = Matrix10::Zero();

    // Compute SO3 left Jacobian for theta_vec = theta(-xi)
    const gtsam::so3::DexpFunctor dexp_functor(theta_vec);
    const Matrix3 D = dexp_functor.leftJacobian();

    // Set D blocks for Jr(xi) = Jl(-xi) - these are the SO3 Jacobian blocks
    Jr.block<3, 3>(0, 0) = D;
    Jr.block<3, 3>(3, 3) = D;
    Jr.block<3, 3>(6, 6) = D;

    // Compute block (0,3) - interaction between rho and nu
    // In LiePlus++: J.template block<3, 3>(6, 3) = -s * Gal3leftJacobianU1(w);
    Matrix3 U1 = computeGal3LeftJacobianU1(theta_vec);
    Jr.block<3, 3>(0, 3) = -t_val * U1;

    // Compute block (3,6) - interaction between nu and theta
    // In LiePlus++: J.template block<3, 3>(3, 0) = Gal3leftJacobianQ1(w, v);
    Matrix3 Q1_nu_theta = computeGal3LeftJacobianQ1(theta_vec, nu_vec);
    Jr.block<3, 3>(3, 6) = Q1_nu_theta;

    // Compute block (0,6) - interaction between rho and theta
    // In LiePlus++: J.template block<3, 3>(6, 0) = Gal3leftJacobianQ1(w, p) - s * Gal3leftJacobianQ2(w, v);
    Matrix3 Q1_rho_theta = computeGal3LeftJacobianQ1(theta_vec, rho_vec);
    Matrix3 Q2_nu_theta = computeGal3LeftJacobianQ2(theta_vec, nu_vec);
    Jr.block<3, 3>(0, 6) = Q1_rho_theta - t_val * Q2_nu_theta;

    // Compute block (0,9) - interaction with time component
    // This part is critical - we need to use E matrix computation style from original
    // GTSAM code rather than Gamma2 computation
    const Matrix3 Theta_hat = skewSymmetric(theta_vec);
    const Matrix3 ThetaTheta = Theta_hat * Theta_hat;
    const double theta_sq = theta_vec.squaredNorm();
    const double theta_abs = std::sqrt(theta_sq);
    const double sin_theta = std::sin(theta_abs);
    const double cos_theta = std::cos(theta_abs);

    // Matrix E computation like in the original code
    Matrix3 E = Matrix3::Identity() * 0.5;
    double A_E_coeff = 0.0, B_E_coeff = 0.0;

    if (theta_sq > Gal3::kSmallAngleThreshold) {
        if (std::abs(theta_abs * theta_sq) < Gal3::kSmallAngleThreshold) {
            A_E_coeff = 1.0/6.0 - theta_sq/120.0;
        } else {
            A_E_coeff = (theta_abs - sin_theta) / (theta_abs * theta_sq);
        }

        if (std::abs(2.0 * theta_sq * theta_sq) < Gal3::kSmallAngleThreshold) {
            B_E_coeff = 1.0/24.0 - theta_sq/720.0;
        } else {
            B_E_coeff = (theta_sq + 2.0 * cos_theta - 2.0) / (2.0 * theta_sq * theta_sq);
        }

        E += A_E_coeff * Theta_hat + B_E_coeff * ThetaTheta;
    } else {
        A_E_coeff = 1.0/6.0;
        B_E_coeff = 1.0/24.0;
        E += A_E_coeff * Theta_hat + B_E_coeff * ThetaTheta;
    }

    Jr.block<3, 1>(0, 9) = E * nu_vec;

    // Block (9,9) for t_tan component
    Jr(9, 9) = 1.0;

    return Jr;
}

} // namespace gtsam
