#include <iostream>
#include <iomanip>
#include <gtsam/geometry/Gal3.h>
using namespace gtsam;

void printMatrix(const Eigen::MatrixXd& mat, const std::string& name) {
    std::cout << name << ":" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < mat.rows(); ++i) {
        std::cout << "  ";
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << std::setw(10) << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Create a test Galilean group element with same parameters for all libraries
    gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(0.0, 0.0, M_PI/6); // 30° rotation around Z
    gtsam::Point3 translation(1.0, 2.0, 3.0);
    gtsam::Vector3 velocity(0.1, 0.2, 0.3);
    double time = 0.5;

    gtsam::Gal3 g(rotation, translation, velocity, time);

    // Compute and print the adjoint map
    gtsam::Matrix10 adjoint = g.AdjointMap();
    printMatrix(adjoint, "GTSAM Adjoint Map");

    // Print component values for verification
    std::cout << "Components for verification:" << std::endl;
    std::cout << "Rotation matrix:" << std::endl << rotation.matrix() << std::endl;
    std::cout << "Translation: " << translation.transpose() << std::endl;
    std::cout << "Velocity: " << velocity.transpose() << std::endl;
    std::cout << "Time: " << time << std::endl;

    return 0;
}
