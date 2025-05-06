/**
 * @file GalileanEuRocDataExample_WithBaseline_Debug.cpp
 * @brief Example evaluating Galilean and Combined IMU Preintegration consistency (15-DOF) on EuRoC dataset.
 * Includes debug prints for covariance diagonals.
 * Mimics the evaluation methodology from Section V-B of
 * Fornasier et al., "Equivariant IMU Preintegration with Biases:
 * a Galilean Group Approach" (arXiv:2411.05548v4), calculating
 * 15-DOF NEES for NavState + Bias using both PreintegratedGalileanMeasurements
 * and PreintegratedCombinedMeasurements for comparison.
 * @author Your Name (based on GTSAM examples and user request)
 */


// GTSAM related includes
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/ImuBias.h>
// Galilean Preintegration
#include <gtsam/navigation/GalileanImuFactor.h> // Include for completeness
#include <gtsam/navigation/PreintegratedGalileanMeasurements.h>
#include <gtsam/navigation/GalileanPreintegrationParams.h>
// Combined (Baseline) Preintegration
#include <gtsam/navigation/CombinedImuFactor.h> // For baseline comparison
#include <gtsam/navigation/PreintegrationCombinedParams.h> // For baseline comparison

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

// Standard C++ includes
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm> // For std::sort, std::lower_bound, std::nth_element
#include <limits>    // For numeric_limits
#include <stdexcept> // For exception handling

// Define custom types needed for this example
namespace gtsam {
    typedef Eigen::Matrix<double, 15, 1> Vector15;
    // Define Matrix15x15 type for convenience, avoiding conflict with GTSAM's Matrix15 (1x5)
    typedef Eigen::Matrix<double, 15, 15> Matrix15x15;
}

using namespace gtsam;
using namespace std;

// --- Configuration ---
// Note: For accurate comparison with paper's Table 1, this path should eventually point
// to the 'mav0' directory INSIDE a specific EuRoC sequence (e.g., MH_01_easy/mav0).
// Using the path structure from user's previous run for now.
const string euroc_dataset_path = "/home/gildroid/workspace2025/gtsam-py"; // Base path
const string imu_csv_path = euroc_dataset_path + "/mav0/imu0/data.csv";
const string ground_truth_csv_path = euroc_dataset_path + "/mav0/state_groundtruth_estimate0/data.csv";

// Preintegration interval duration (seconds) - match paper's columns
const double deltaTij = 0.5; // Options: 0.2, 0.5, 1.0

// --- EuRoC Noise Parameters (VI Sensor From Dataset Spec) ---
// These are the *NOISE DENSITIES* from the EuRoC VI sensor spec (sensor.yaml)
// Units are per sqrt(Hz). Verify these against the sensor.yaml for your specific sequence.
const double gyro_noise_density = 1.6968e-04; // [rad/s/sqrt(Hz)]
const double accel_noise_density = 2.0000e-03; // [m/s^2/sqrt(Hz)]
// These are the *RANDOM WALK* densities
const double gyro_bias_rw_density = 1.9393e-05; // [rad/s^2/sqrt(Hz)] = rad/s/sqrt(Hz)/s
const double accel_bias_rw_density = 3.0000e-03; // [m/s^3/sqrt(Hz)] = m/s^2/sqrt(Hz)/s

// Integration noise sigma (usually very small or zero)
const double integration_noise_sigma = 1e-8;

// Gravity vector in navigation frame (ENU convention: Z-up, gravity is negative Z)
const Vector3 gravity_n(0, 0, -9.81);

// --- Data Structures ---

struct ImuData {
    double timestamp; // seconds
    Vector3 omega;    // rad/s (gyro)
    Vector3 acc;      // m/s^2 (accel)
};

struct GroundTruthData {
    double timestamp;  // seconds
    NavState navState;
    imuBias::ConstantBias bias;
};

// --- Helper Functions ---

// Function to parse IMU CSV line
bool parseImuLine(const string& line, ImuData& data) {
    stringstream ss(line);
    string segment;
    vector<string> seglist;

    while (getline(ss, segment, ',')) {
        seglist.push_back(segment);
    }

    if (seglist.size() < 7) return false; // Expect at least 7 columns

    try {
        data.timestamp = stod(seglist[0]) * 1e-9; // ns to s
        data.omega = Vector3(stod(seglist[1]), stod(seglist[2]), stod(seglist[3]));
        data.acc = Vector3(stod(seglist[4]), stod(seglist[5]), stod(seglist[6]));
    } catch (const std::invalid_argument& e) {
        cerr << "Error parsing IMU line (invalid_argument): " << line << endl;
        return false;
    } catch (const std::out_of_range& e) {
        cerr << "Error parsing IMU line (out_of_range): " << line << endl;
        return false;
    }
    return true;
}

// Function to parse Ground Truth CSV line
bool parseGroundTruthLine(const string& line, GroundTruthData& data) {
    stringstream ss(line);
    string segment;
    vector<string> seglist;

    while (getline(ss, segment, ',')) {
        seglist.push_back(segment);
    }

    if (seglist.size() < 17) return false; // Expect at least 17 columns

    try {
        data.timestamp = stod(seglist[0]) * 1e-9; // ns to s
        Point3 pos(stod(seglist[1]), stod(seglist[2]), stod(seglist[3]));
        // Quaternion order in EuRoC: w, x, y, z
        Quaternion q(stod(seglist[4]), stod(seglist[5]), stod(seglist[6]), stod(seglist[7]));
        Rot3 rot = Rot3(q);
        Velocity3 vel(stod(seglist[8]), stod(seglist[9]), stod(seglist[10]));
        data.navState = NavState(rot, pos, vel);

        Vector3 bias_gyro(stod(seglist[11]), stod(seglist[12]), stod(seglist[13]));
        Vector3 bias_acc(stod(seglist[14]), stod(seglist[15]), stod(seglist[16]));
        data.bias = imuBias::ConstantBias(bias_acc, bias_gyro);
    } catch (const std::invalid_argument& e) {
        cerr << "Error parsing Ground Truth line (invalid_argument): " << line << endl;
        return false;
    } catch (const std::out_of_range& e) {
        cerr << "Error parsing Ground Truth line (out_of_range): " << line << endl;
        return false;
    }
    return true;
}

// Function to load data from CSV
template <typename T>
bool loadData(const string& filename, vector<T>& data, bool (*parseFunc)(const string&, T&)) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return false;
    }

    string line;
    // Skip header line
    if (!getline(file, line)) {
        cerr << "Error reading header or empty file: " << filename << endl;
        return false;
    }

    while (getline(file, line)) {
        // Skip empty lines or lines starting with '#' (sometimes present)
        if (line.empty() || line[0] == '#') continue;

        T entry;
        if (parseFunc(line, entry)) {
            data.push_back(entry);
        } else {
            // Don't print every skipped line, could be many
            // cerr << "Skipping invalid line in " << filename << ": " << line << endl;
        }
    }
    cout << "Loaded " << data.size() << " valid entries from " << filename << endl;
    if (data.empty()) {
        cerr << "Warning: No valid data loaded from " << filename << endl;
    }
    return !data.empty();
}

// Find index of the data point at or just before the given timestamp
template <typename T>
size_t findIndexBefore(const vector<T>& sorted_data, double timestamp) {
    // Find the first element NOT less than timestamp
    auto it = lower_bound(sorted_data.begin(), sorted_data.end(), timestamp,
                          [](const T& element, double value) {
                              return element.timestamp < value;
                          });

    // If iterator points to the beginning, it means timestamp is before or equal to the first element.
    if (it == sorted_data.begin()) {
         // If the first element's timestamp is >= target, return 0. Otherwise, timestamp is before all data.
         if (it != sorted_data.end() && it->timestamp >= timestamp) {
            return 0;
         } else {
            cerr << "Warning: Timestamp " << fixed << setprecision(9) << timestamp
                 << " is before the first data point " << sorted_data.front().timestamp << endl;
            return 0; // Return 0, but be aware it might be before the range
         }
    } else {
         // Iterator points to the first element >= timestamp. We want the one before it.
         return distance(sorted_data.begin(), it) - 1;
    }
}


// Find index of the first data point at or after the given timestamp
template <typename T>
size_t findIndexAtOrAfter(const vector<T>& sorted_data, double timestamp) {
     // Find the first element NOT less than timestamp
     auto it = lower_bound(sorted_data.begin(), sorted_data.end(), timestamp,
                           [](const T& element, double value) {
                               return element.timestamp < value;
                           });
     // Return the index of this element
     return distance(sorted_data.begin(), it);
}

// Function to calculate the median of a vector
double calculateMedian(vector<double>& values) {
    if (values.empty()) {
        return numeric_limits<double>::quiet_NaN(); // Return NaN for empty vector
    }
    // Use nth_element for efficient median calculation without full sort
    size_t n = values.size();
    size_t mid = n / 2;
    nth_element(values.begin(), values.begin() + mid, values.end());

    if (n % 2 != 0) { // Odd size
        return values[mid];
    } else { // Even size
        // Find element just before mid for even case
        // Need the value at mid and the max value in the lower half
        double mid_val1 = values[mid];
        nth_element(values.begin(), values.begin() + mid - 1, values.end());
        return (values[mid - 1] + mid_val1) / 2.0;
    }
}

// --- Main Evaluation Logic ---
int main(int argc, char* argv[]) {

    cout << "Starting EuRoC Galilean vs Combined Preintegration Evaluation (15-DOF NEES)..." << endl;
    cout << "Dataset path: " << euroc_dataset_path << endl; // Prints the base path
    cout << "Preintegration Interval (deltaTij): " << deltaTij << " s" << endl;

    // 1. Load Data
    vector<ImuData> imu_data;
    vector<GroundTruthData> gt_data;

    if (!loadData(imu_csv_path, imu_data, parseImuLine)) return 1; // Uses constructed path
    if (!loadData(ground_truth_csv_path, gt_data, parseGroundTruthLine)) return 1; // Uses constructed path

    if (imu_data.empty() || gt_data.empty()) {
        cerr << "IMU or Ground Truth data is empty after loading." << endl;
        return 1;
    }

    // 2. Setup Preintegration Parameters
    // These are calculated from densities assuming a nominal IMU dt.
    const double nominal_imu_dt = 0.005; // Assuming 200 Hz (Verified by python script)

    // Parameters for Galilean Preintegration
    auto p_galilean = GalileanPreintegrationParams::MakeSharedU(gravity_n.norm());
    p_galilean->gyroscopeCovariance = pow(gyro_noise_density, 2) / nominal_imu_dt * I_3x3;
    p_galilean->accelerometerCovariance = pow(accel_noise_density, 2) / nominal_imu_dt * I_3x3;
    p_galilean->integrationCovariance = pow(integration_noise_sigma, 2) * I_3x3;
    p_galilean->biasAccCovariance = pow(accel_bias_rw_density, 2) * nominal_imu_dt * I_3x3;
    p_galilean->biasOmegaCovariance = pow(gyro_bias_rw_density, 2) * nominal_imu_dt * I_3x3;
    p_galilean->biasAccOmegaInt = I_6x6 * 1e-5; // Example value

    // Parameters for Combined Preintegration (Baseline)
    auto p_combined = PreintegrationCombinedParams::MakeSharedU(gravity_n.norm());
    p_combined->gyroscopeCovariance = p_galilean->gyroscopeCovariance;
    p_combined->accelerometerCovariance = p_galilean->accelerometerCovariance;
    p_combined->integrationCovariance = p_galilean->integrationCovariance;
    p_combined->biasAccCovariance = p_galilean->biasAccCovariance;
    p_combined->biasOmegaCovariance = p_galilean->biasOmegaCovariance;
    p_combined->biasAccOmegaInt = p_galilean->biasAccOmegaInt;
    p_combined->n_gravity = gravity_n;

    cout << "\nUsing Galilean Preintegration Parameters:" << endl;
    p_galilean->print("Params_Galilean");
    cout << "\nUsing Combined Preintegration Parameters:" << endl;
    p_combined->print("Params_Combined");


    // 3. Process Data in Intervals
    vector<double> nees_results_galilean;
    vector<double> nees_results_combined; // For baseline
    vector<double> time_diffs; // Store actual integrated time intervals

    double min_time = max(imu_data.front().timestamp, gt_data.front().timestamp);
    double max_time = min(imu_data.back().timestamp, gt_data.back().timestamp);

    // Adjust start time slightly to ensure first GT lookup is valid
    size_t first_valid_gt_idx = findIndexAtOrAfter(gt_data, min_time);
    if (first_valid_gt_idx >= gt_data.size()) {
         cerr << "Error: No ground truth data found at or after minimum start time." << endl;
         return 1;
    }
    min_time = gt_data[first_valid_gt_idx].timestamp;


    cout << fixed << setprecision(9); // Use higher precision for timestamps
    cout << "Processing data from " << min_time << " s to " << max_time << " s" << endl;

    double current_t_i = min_time;
    int skipped_intervals_gt = 0;
    int skipped_intervals_imu = 0;
    int numerical_issue_count = 0;
    int processed_intervals = 0;
    const int max_debug_prints = 5; // How many intervals to print debug info for

    while (current_t_i + deltaTij <= max_time) {
        double t_j_target = current_t_i + deltaTij;

        // --- Get Ground Truth Data at t_i ---
        size_t gt_idx_i = findIndexAtOrAfter(gt_data, current_t_i);
        if (gt_idx_i >= gt_data.size() || gt_data[gt_idx_i].timestamp > current_t_i + 0.01) {
            current_t_i += deltaTij;
            skipped_intervals_gt++;
            continue;
        }
        const GroundTruthData& gt_i = gt_data[gt_idx_i];
        current_t_i = gt_i.timestamp;
        t_j_target = current_t_i + deltaTij;

        // --- Get Ground Truth Data at t_j ---
        size_t gt_idx_j = findIndexBefore(gt_data, t_j_target);
        if (gt_idx_j <= gt_idx_i || gt_idx_j >= gt_data.size() || gt_data[gt_idx_j].timestamp < current_t_i) {
            current_t_i += deltaTij;
            skipped_intervals_gt++;
            continue;
        }
        const GroundTruthData& gt_j = gt_data[gt_idx_j];
        double actual_t_j = gt_j.timestamp;

        // --- Initialize Preintegrators ---
        PreintegratedGalileanMeasurements pim_galilean(p_galilean, gt_i.bias);
        PreintegratedCombinedMeasurements pim_combined(p_combined, gt_i.bias); // Baseline

        // --- Integrate IMU Measurements ---
        size_t imu_idx_start = findIndexAtOrAfter(imu_data, current_t_i);
        size_t imu_idx_end = findIndexAtOrAfter(imu_data, actual_t_j);

        if (imu_idx_start >= imu_data.size() || imu_idx_start >= imu_idx_end ) {
             current_t_i += deltaTij;
             skipped_intervals_imu++;
             continue;
        }

        double previous_imu_t = current_t_i;
        for (size_t k = imu_idx_start; k < imu_idx_end; ++k) {
            const ImuData& imu = imu_data[k];
            double current_imu_t = imu.timestamp;
            double dt = current_imu_t - previous_imu_t;

            if (dt <= 1e-9) {
                 previous_imu_t = current_imu_t;
                 continue;
            }
            if (dt > 0.1) { /* cout << "Warning: Large IMU dt=" << dt << endl; */ }

            // Integrate for both PIMs
            pim_galilean.integrateMeasurement(imu.acc, imu.omega, dt);
            pim_combined.integrateMeasurement(imu.acc, imu.omega, dt); // Integrate baseline

            previous_imu_t = current_imu_t;
        }

        double final_dt = actual_t_j - previous_imu_t;
        if (final_dt > 1e-9 && imu_idx_end > imu_idx_start) {
             const ImuData& last_imu = imu_data[imu_idx_end - 1];
             pim_galilean.integrateMeasurement(last_imu.acc, last_imu.omega, final_dt);
             pim_combined.integrateMeasurement(last_imu.acc, last_imu.omega, final_dt); // Integrate baseline
        }

        // Store the actual integrated time duration (should be same for both)
        time_diffs.push_back(pim_galilean.deltaTij());

        if (pim_galilean.deltaTij() < 1e-6 || pim_combined.deltaTij() < 1e-6) {
             current_t_i += deltaTij;
             skipped_intervals_imu++;
             continue;
        }

        // --- Predict State and Calculate Errors ---
        NavState estimated_state_j_galilean = pim_galilean.predict(gt_i.navState, gt_i.bias);
        NavState estimated_state_j_combined = pim_combined.predict(gt_i.navState, gt_i.bias);

        Vector9 error_nav_galilean = gt_j.navState.localCoordinates(estimated_state_j_galilean);
        Vector9 error_nav_combined = gt_j.navState.localCoordinates(estimated_state_j_combined);

        Vector6 error_bias = gt_j.bias.vector() - gt_i.bias.vector();

        Vector15 error15_galilean;
        error15_galilean << error_nav_galilean, error_bias;
        Vector15 error15_combined;
        error15_combined << error_nav_combined, error_bias;

        // --- Get Covariances ---
        Matrix P20 = pim_galilean.uncertaintyCovariance();
        Matrix15x15 P15_galilean = Matrix15x15::Zero(15, 15);
        Matrix15x15 P15_combined = pim_combined.preintMeasCov();

        // Extract Galilean Covariance
        bool extraction_ok = true;
        if (P20.rows() != 20 || P20.cols() != 20) {
            cerr << "Error: Expected 20x20 Galilean covariance matrix, got "
                 << P20.rows() << "x" << P20.cols() << " at t_i=" << current_t_i << endl;
            numerical_issue_count++;
            extraction_ok = false;
        } else {
            try {
                // Extract NavState block (9x9) -> Top-Left of P15
                P15_galilean.block<3, 3>(0, 0) = P20.block<3, 3>(6, 6); // Rot-Rot (theta-theta)
                P15_galilean.block<3, 3>(3, 3) = P20.block<3, 3>(0, 0); // Pos-Pos (rho-rho)
                P15_galilean.block<3, 3>(6, 6) = P20.block<3, 3>(3, 3); // Vel-Vel (nu-nu)
                P15_galilean.block<3, 3>(0, 3) = P20.block<3, 3>(6, 0); // Rot-Pos (theta-rho)
                P15_galilean.block<3, 3>(3, 0) = P20.block<3, 3>(0, 6); // Pos-Rot (rho-theta)
                P15_galilean.block<3, 3>(0, 6) = P20.block<3, 3>(6, 3); // Rot-Vel (theta-nu)
                P15_galilean.block<3, 3>(6, 0) = P20.block<3, 3>(3, 6); // Vel-Rot (nu-theta)
                P15_galilean.block<3, 3>(3, 6) = P20.block<3, 3>(0, 3); // Pos-Vel (rho-nu)
                P15_galilean.block<3, 3>(6, 3) = P20.block<3, 3>(3, 0); // Vel-Pos (nu-rho)
                // Extract Bias block (6x6) -> Bottom-Right of P15
                P15_galilean.block<3, 3>(9, 9)   = P20.block<3, 3>(13, 13); // Acc-Acc (b_acc-b_acc)
                P15_galilean.block<3, 3>(12, 12) = P20.block<3, 3>(10, 10); // Gyro-Gyro (b_omega-b_omega)
                P15_galilean.block<3, 3>(9, 12)  = P20.block<3, 3>(13, 10); // Acc-Gyro (b_acc-b_omega)
                P15_galilean.block<3, 3>(12, 9)  = P20.block<3, 3>(10, 13); // Gyro-Acc (b_omega-b_acc)
                // Extract NavState-Bias cross-covariance (9x6) -> Top-Right of P15
                P15_galilean.block<3, 3>(0, 9) = P20.block<3, 3>(6, 13); // Rot-Acc (theta-b_acc)
                P15_galilean.block<3, 3>(0, 12) = P20.block<3, 3>(6, 10); // Rot-Gyro (theta-b_omega)
                P15_galilean.block<3, 3>(3, 9) = P20.block<3, 3>(0, 13); // Pos-Acc (rho-b_acc)
                P15_galilean.block<3, 3>(3, 12) = P20.block<3, 3>(0, 10); // Pos-Gyro (rho-b_omega)
                P15_galilean.block<3, 3>(6, 9) = P20.block<3, 3>(3, 13); // Vel-Acc (nu-b_acc)
                P15_galilean.block<3, 3>(6, 12) = P20.block<3, 3>(3, 10); // Vel-Gyro (nu-b_omega)
                // Extract Bias-NavState cross-covariance (6x9) -> Bottom-Left of P15
                P15_galilean.block<3, 3>(9, 0) = P20.block<3, 3>(13, 6); // Acc-Rot (b_acc-theta)
                P15_galilean.block<3, 3>(9, 3) = P20.block<3, 3>(13, 0); // Acc-Pos (b_acc-rho)
                P15_galilean.block<3, 3>(9, 6) = P20.block<3, 3>(13, 3); // Acc-Vel (b_acc-nu)
                P15_galilean.block<3, 3>(12, 0) = P20.block<3, 3>(10, 6); // Gyro-Rot (b_omega-theta)
                P15_galilean.block<3, 3>(12, 3) = P20.block<3, 3>(10, 0); // Gyro-Pos (b_omega-rho)
                P15_galilean.block<3, 3>(12, 6) = P20.block<3, 3>(10, 3); // Gyro-Vel (b_omega-nu)
            } catch (const std::exception& e) {
                cerr << "Error during Galilean covariance block extraction at t_i=" << current_t_i << ": " << e.what() << endl;
                numerical_issue_count++;
                extraction_ok = false;
            }
        }

        // Check Combined Covariance size
        if (P15_combined.rows() != 15 || P15_combined.cols() != 15) {
             cerr << "Error: Expected 15x15 Combined covariance matrix, got "
                 << P15_combined.rows() << "x" << P15_combined.cols() << " at t_i=" << current_t_i << endl;
             numerical_issue_count++;
             extraction_ok = false; // Skip NEES calculation if either cov is bad
        }

        // --- DEBUG: Print Covariance Diagonals and Errors for first few intervals ---
        if (processed_intervals < max_debug_prints && extraction_ok) {
            cout << "\n--- DEBUG: Interval " << processed_intervals << " (t_i=" << current_t_i << ") ---" << endl;
            cout << scientific << setprecision(6); // Use scientific notation for clarity
            cout << "P15_galilean Diag:\n" << P15_galilean.diagonal().transpose() << endl;
            cout << "P15_combined Diag:\n" << P15_combined.diagonal().transpose() << endl;
            cout << "Error15_galilean: " << error15_galilean.transpose() << endl;
            cout << "Error15_combined: " << error15_combined.transpose() << endl;
            cout << defaultfloat << setprecision(9); // Reset precision
        }
        // --- END DEBUG ---


        // --- Calculate 15-DOF NEES if covariances are valid ---
        if (extraction_ok) {
            try {
                // Regularize and Invert Galilean Covariance
                Matrix15x15 P15_galilean_reg = P15_galilean + Matrix15x15::Identity() * 1e-9;
                Matrix15x15 P15_galilean_inv = P15_galilean_reg.inverse();
                double nees15_galilean = error15_galilean.transpose() * P15_galilean_inv * error15_galilean;

                if (!isnan(nees15_galilean) && !isinf(nees15_galilean) && nees15_galilean >= 0) {
                    nees_results_galilean.push_back(nees15_galilean);
                } else {
                    numerical_issue_count++;
                }
            } catch (const std::exception& e) {
                 numerical_issue_count++;
            }

            try {
                // Regularize and Invert Combined Covariance
                Matrix15x15 P15_combined_reg = P15_combined + Matrix15x15::Identity() * 1e-9;
                Matrix15x15 P15_combined_inv = P15_combined_reg.inverse();
                double nees15_combined = error15_combined.transpose() * P15_combined_inv * error15_combined;

                if (!isnan(nees15_combined) && !isinf(nees15_combined) && nees15_combined >= 0) {
                    nees_results_combined.push_back(nees15_combined);
                } else {
                    numerical_issue_count++; // Count issues for either calculation
                }
            } catch (const std::exception& e) {
                 numerical_issue_count++; // Count issues for either calculation
            }

            // Only increment processed intervals if both NEES calculations were attempted without prior errors
             processed_intervals++;

        } else {
            // Covariance extraction failed, skip NEES and move to next interval
             current_t_i += deltaTij;
             continue;
        }

        // Move to the start of the next non-overlapping interval
        current_t_i += deltaTij;

    } // end while loop over intervals

    // 4. Calculate Median NEES and other stats
    cout << "\nProcessed " << processed_intervals << " valid intervals." << endl;
    cout << "Skipped " << skipped_intervals_gt << " intervals due to GT data gaps." << endl;
    cout << "Skipped " << skipped_intervals_imu << " intervals due to IMU data gaps." << endl;
    cout << "Skipped " << numerical_issue_count << " NEES calculations due to numerical/extraction issues." << endl;

    if (nees_results_galilean.empty() || nees_results_combined.empty()) {
        cerr << "No valid NEES results were calculated for one or both methods." << endl;
        // Optionally print counts for debugging
        cerr << "Galilean NEES count: " << nees_results_galilean.size() << endl;
        cerr << "Combined NEES count: " << nees_results_combined.size() << endl;
        return 1;
    }

    // Calculate Galilean Stats
    double median_nees_galilean = calculateMedian(nees_results_galilean);
    double mean_nees_galilean = 0;
    double nees_sum_sq_galilean = 0;
    int valid_nees_count_galilean = 0;
    for(double n : nees_results_galilean) {
        if (isfinite(n)) {
            mean_nees_galilean += n;
            nees_sum_sq_galilean += n*n;
            valid_nees_count_galilean++;
        }
    }
    double nees_std_dev_galilean = 0;
    if (valid_nees_count_galilean > 0) {
        mean_nees_galilean /= valid_nees_count_galilean;
        if (valid_nees_count_galilean > 1) {
             nees_std_dev_galilean = sqrt((nees_sum_sq_galilean - (mean_nees_galilean * mean_nees_galilean * valid_nees_count_galilean)) / (valid_nees_count_galilean - 1));
        }
    }

    // Calculate Combined (Baseline) Stats
    double median_nees_combined = calculateMedian(nees_results_combined);
    double mean_nees_combined = 0;
    double nees_sum_sq_combined = 0;
    int valid_nees_count_combined = 0;
    for(double n : nees_results_combined) {
        if (isfinite(n)) {
            mean_nees_combined += n;
            nees_sum_sq_combined += n*n;
            valid_nees_count_combined++;
        }
    }
    double nees_std_dev_combined = 0;
    if (valid_nees_count_combined > 0) {
        mean_nees_combined /= valid_nees_count_combined;
        if (valid_nees_count_combined > 1) {
             nees_std_dev_combined = sqrt((nees_sum_sq_combined - (mean_nees_combined * mean_nees_combined * valid_nees_count_combined)) / (valid_nees_count_combined - 1));
        }
    }


    // Calculate Time Stats
    double median_dt = calculateMedian(time_diffs);
    double mean_dt = 0;
    int valid_dt_count = 0;
     for(double d : time_diffs) {
        if (isfinite(d)) {
            mean_dt += d;
            valid_dt_count++;
        }
     }
    if (valid_dt_count > 0) mean_dt /= valid_dt_count;


    cout << "\n--- Results (15-DOF NEES) ---" << endl;
    cout << "Dataset: " << euroc_dataset_path << endl;
    cout << "Target Preintegration Interval (deltaTij): " << fixed << setprecision(5) << deltaTij << " s" << endl;
    cout << "Actual Median Integrated Interval: " << fixed << setprecision(5) << median_dt << " s" << endl;
    cout << "Actual Mean Integrated Interval:   " << fixed << setprecision(5) << mean_dt << " s" << endl;

    cout << "\n--- Galilean Preintegration ---" << endl;
    cout << "Number of Valid NEES Intervals: " << valid_nees_count_galilean << endl;
    cout << "Median 15-DOF NEES: " << fixed << setprecision(3) << median_nees_galilean << endl;
    cout << "Mean 15-DOF NEES:   " << fixed << setprecision(3) << mean_nees_galilean << endl;
    cout << "StdDev 15-DOF NEES: " << fixed << setprecision(3) << nees_std_dev_galilean << endl;

    cout << "\n--- Combined Preintegration (Baseline) ---" << endl;
    cout << "Number of Valid NEES Intervals: " << valid_nees_count_combined << endl;
    cout << "Median 15-DOF NEES: " << fixed << setprecision(3) << median_nees_combined << endl;
    cout << "Mean 15-DOF NEES:   " << fixed << setprecision(3) << mean_nees_combined << endl;
    cout << "StdDev 15-DOF NEES: " << fixed << setprecision(3) << nees_std_dev_combined << endl;
    cout << "--------------------------------------------" << endl;

    cout << "\nNote: Expected Median/Mean NEES for 15-DOF is approx 15.0 for a consistent system." << endl;
    cout << "Compare Galilean results to Table I (Gal(3) x gal(3) column) in Fornasier et al." << endl;
    cout << "Compare Combined results to Table I (e.g., [18], [22], [23] columns)." << endl;
    cout << "Ensure correct EuRoC noise parameters and dataset path (pointing to a specific sequence) are used." << endl;

    return 0;
}
