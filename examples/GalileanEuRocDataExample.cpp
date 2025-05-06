/**
 * @file evaluatePreintegrationNavStateBias.cpp // Renamed for clarity
 * @brief Example evaluating IMU Preintegration consistency (15-DOF) on EuRoC dataset.
 * Mimics the evaluation methodology from Section V-B of
 * Fornasier et al., "Equivariant IMU Preintegration with Biases:
 * a Galilean Group Approach" (arXiv:2411.05548v4), calculating
 * 15-DOF NEES for NavState + Bias using PreintegratedCombinedMeasurements.
 * @author Your Name (based on GTSAM examples and user request)
 */


// GTSAM related includes
#include <gtsam/navigation/NavState.h>
// #include <gtsam/navigation/ImuFactor.h> // Defines PreintegratedImuMeasurements
#include <gtsam/navigation/CombinedImuFactor.h> // Defines PreintegratedCombinedMeasurements
#include <gtsam/navigation/ImuBias.h>

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
#include <algorithm> // For std::sort, std::lower_bound
#include <limits>    // For numeric_limits

// Define the Vector15 type
namespace gtsam {
    typedef Eigen::Matrix<double, 15, 1> Vector15;
}

using namespace gtsam;
using namespace std;

// --- Configuration ---
const string euroc_dataset_path = "/home/gildroid/workspace2025/gtsam-py"; // Example path
const string imu_csv_path = euroc_dataset_path + "/mav0/imu0/data.csv";
const string ground_truth_csv_path = euroc_dataset_path + "/mav0/state_groundtruth_estimate0/data.csv";

// Preintegration interval duration (seconds) - match paper's columns
const double deltaTij = 0.5; // Options: 0.2, 0.5, 1.0

// --- EuRoC Noise Parameters (VI Sensor From Dataset Spec) ---
// NOTE 1: GTSAM uses continuous-time noise densities
// NOTE 2: These values likely need significant tuning for 15-DOF NEES consistency!
const double acc_noise_sigma = 2.0000e-03 * 1800.0;  // Now 3.6
const double gyro_noise_sigma = 1.6968e-04 * 1800.0; // Now 0.305424

// --- Keep others at original/low values for now ---
const double acc_bias_rw_sigma = 3.0000e-03;
const double gyro_bias_rw_sigma = 1.9393e-05;
const double integration_noise_sigma = 1e-8; // Keep low

const Vector3 gravity_n(0, 0, 9.81); // Gravity vector in navigation frame (assuming Z-up)

// --- Data Structures ---

struct ImuData {
    double timestamp; // seconds
    Vector3 omega;    // rad/s
    Vector3 acc;      // m/s^2
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

    if (seglist.size() != 7) return false; // Expect 7 columns

    try {
        data.timestamp = stod(seglist[0]) * 1e-9; // ns to s
        data.omega = Vector3(stod(seglist[1]), stod(seglist[2]), stod(seglist[3]));
        data.acc = Vector3(stod(seglist[4]), stod(seglist[5]), stod(seglist[6]));
    } catch (const std::invalid_argument& e) {
        cerr << "Error parsing IMU line: " << line << endl;
        return false;
    } catch (const std::out_of_range& e) {
        cerr << "Out of range error parsing IMU line: " << line << endl;
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

    if (seglist.size() != 17) return false; // Expect 17 columns

    try {
        data.timestamp = stod(seglist[0]) * 1e-9; // ns to s
        Point3 pos(stod(seglist[1]), stod(seglist[2]), stod(seglist[3]));
        // Quaternion order in EuRoC: w, x, y, z
        Quaternion q(stod(seglist[4]), stod(seglist[5]), stod(seglist[6]), stod(seglist[7]));
        Rot3 rot = Rot3(q);
        Velocity3 vel(stod(seglist[8]), stod(seglist[9]), stod(seglist[10]));
        data.navState = NavState(rot, pos, vel);

        Vector3 bias_acc(stod(seglist[14]), stod(seglist[15]), stod(seglist[16]));
        Vector3 bias_gyro(stod(seglist[11]), stod(seglist[12]), stod(seglist[13]));
        data.bias = imuBias::ConstantBias(bias_acc, bias_gyro);
    } catch (const std::invalid_argument& e) {
        cerr << "Error parsing Ground Truth line: " << line << endl;
        return false;
    } catch (const std::out_of_range& e) {
        cerr << "Out of range error parsing Ground Truth line: " << line << endl;
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
            cerr << "Skipping invalid line in " << filename << ": " << line << endl;
        }
    }
    cout << "Loaded " << data.size() << " entries from " << filename << endl;
    return !data.empty();
}

// Find index of the data point at or just before the given timestamp
template <typename T>
size_t findIndexBefore(const vector<T>& sorted_data, double timestamp) {
    auto it = lower_bound(sorted_data.begin(), sorted_data.end(), timestamp,
                          [](const T& element, double value) {
                              return element.timestamp < value;
                          });

    if (it == sorted_data.begin()) {
         if (it != sorted_data.end() && it->timestamp >= timestamp) return 0;
         cerr << "Warning: Timestamp " << timestamp << " is before the first data point " << sorted_data.front().timestamp << endl;
         return 0; // Or throw? Consider edge case carefully.
    } else {
         return distance(sorted_data.begin(), it) - 1;
    }
}

// Find index of the first data point at or after the given timestamp
template <typename T>
size_t findIndexAtOrAfter(const vector<T>& sorted_data, double timestamp) {
     auto it = lower_bound(sorted_data.begin(), sorted_data.end(), timestamp,
                           [](const T& element, double value) {
                               return element.timestamp < value;
                           });
     return distance(sorted_data.begin(), it);
}

// Function to calculate the median of a vector
double calculateMedian(vector<double>& values) {
    if (values.empty()) {
        return 0.0; // Or NaN or throw an exception
    }
    // Use nth_element for efficient median calculation without full sort
    size_t n = values.size();
    size_t mid = n / 2;
    nth_element(values.begin(), values.begin() + mid, values.end());
    if (n % 2 != 0) { // Odd size
        return values[mid];
    } else { // Even size
        double mid_val1 = values[mid];
        // Find element just before mid for even case
        nth_element(values.begin(), values.begin() + mid - 1, values.end());
        return (values[mid - 1] + mid_val1) / 2.0;
    }
}

// --- Main Evaluation Logic ---
int main(int argc, char* argv[]) {

    cout << "Starting EuRoC Preintegration Evaluation (15-DOF NEES)..." << endl;
    cout << "Dataset path: " << euroc_dataset_path << endl;
    cout << "Preintegration Interval (deltaTij): " << deltaTij << " s" << endl;

    // 1. Load Data
    vector<ImuData> imu_data;
    vector<GroundTruthData> gt_data;

    if (!loadData(imu_csv_path, imu_data, parseImuLine)) return 1;
    if (!loadData(ground_truth_csv_path, gt_data, parseGroundTruthLine)) return 1;

    if (imu_data.empty() || gt_data.empty()) {
        cerr << "IMU or Ground Truth data is empty after loading." << endl;
        return 1;
    }

    // 2. Setup Preintegration Parameters using PreintegratedCombinedMeasurements::Params
    const Vector3 gravity_n(0, 0, 9.81); // Using ENU convention (+Z gravity)

    // Use PreintegrationCombinedParams and MakeSharedU for ENU gravity convention
    auto p = PreintegrationCombinedParams::MakeSharedU(gravity_n.norm()); // Use magnitude for constructor

    // Set common parameters (measurement noises)
    // ** These likely need significant tuning for 15-DOF NEES **
    p->gyroscopeCovariance = pow(gyro_noise_sigma, 2) * I_3x3;
    p->accelerometerCovariance = pow(acc_noise_sigma, 2) * I_3x3;
    p->integrationCovariance = pow(integration_noise_sigma, 2) * I_3x3;

    // Set bias random walk covariances (required for Combined)
    // ** These also likely need tuning for 15-DOF NEES **
    p->biasAccCovariance = pow(acc_bias_rw_sigma, 2) * I_3x3;
    p->biasOmegaCovariance = pow(gyro_bias_rw_sigma, 2) * I_3x3;
    p->biasAccOmegaInt = I_6x6 * 1e-5; // Initial correlation between bias random walks (common value from examples)


    // Set sensor pose if necessary (default is identity)
    // p->body_P_sensor = Pose3(...)

    // Explicitly set gravity vector in params (MakeSharedU should do this, but belt-and-suspenders)
    p->n_gravity = gravity_n;


    // 3. Process Data in Intervals
    vector<double> nees_results;
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


    cout << fixed << setprecision(5);
    cout << "Processing data from " << min_time << " s to " << max_time << " s" << endl;

    double current_t_i = min_time;
    int skipped_intervals = 0;
    int numerical_issue_count = 0;

    while (current_t_i + deltaTij <= max_time) {
        double t_j = current_t_i + deltaTij;

        // --- Get Ground Truth Data ---
        size_t gt_idx_i = findIndexAtOrAfter(gt_data, current_t_i);
        if (gt_idx_i >= gt_data.size() || gt_data[gt_idx_i].timestamp > current_t_i + 0.01) {
            // cout << "Warning: GT data gap near t_i = " << current_t_i << ". Skipping interval." << endl;
            current_t_i += deltaTij; // Skip to next potential interval
            skipped_intervals++;
            continue;
        }
        const GroundTruthData& gt_i = gt_data[gt_idx_i];
        current_t_i = gt_i.timestamp; // Use actual GT time as start
        t_j = current_t_i + deltaTij; // Recalculate t_j

        size_t gt_idx_j = findIndexBefore(gt_data, t_j);
        if (gt_idx_j <= gt_idx_i || gt_idx_j >= gt_data.size()) {
            // cout << "Warning: Cannot find suitable GT data before end time t_j = " << t_j << ". Skipping interval." << endl;
            current_t_i += deltaTij; // Try next interval start
            skipped_intervals++;
            continue;
        }
        const GroundTruthData& gt_j = gt_data[gt_idx_j]; // Store full GT at j

        // --- Initialize Preintegrator ---
        // Use PreintegratedCombinedMeasurements
        PreintegratedCombinedMeasurements pim(p, gt_i.bias); // Initialize with GT bias at t_i

        // --- Integrate IMU Measurements ---
        size_t imu_idx_start = findIndexAtOrAfter(imu_data, current_t_i);
        size_t imu_idx_end = findIndexAtOrAfter(imu_data, t_j);

        if (imu_idx_start >= imu_data.size() || imu_idx_start >= imu_idx_end ) {
            current_t_i += deltaTij;
            skipped_intervals++;
            continue;
        }

        double previous_imu_t = (imu_idx_start > 0) ? imu_data[imu_idx_start - 1].timestamp : current_t_i;
        if (previous_imu_t > current_t_i) { previous_imu_t = current_t_i; }

        for (size_t k = imu_idx_start; k < imu_idx_end; ++k) {
            const ImuData& imu = imu_data[k];
            double current_imu_t = imu.timestamp;
            double dt = (k == imu_idx_start) ? (current_imu_t - current_t_i) : (current_imu_t - previous_imu_t);

            if (k == imu_idx_start && dt < 1e-9) {
                 if (imu_idx_start > 0) dt = current_imu_t - imu_data[imu_idx_start-1].timestamp;
                 else dt = 0;
            }

            if (dt <= 1e-9 || dt > 0.1) {
                 previous_imu_t = current_imu_t;
                 continue;
            }
            pim.integrateMeasurement(imu.acc, imu.omega, dt);
            previous_imu_t = current_imu_t;
        }

        time_diffs.push_back(pim.deltaTij());

        if (pim.deltaTij() < 1e-6) {
             current_t_i += deltaTij;
             skipped_intervals++;
             continue;
        }

        // --- Predict State and Calculate Errors ---
        // Predict NavState using CombinedIMU (may only predict NavState part directly)
        NavState estimated_state_j = pim.predict(gt_i.navState, gt_i.bias);

        // Calculate 9-DOF NavState error
        Vector9 error_nav = gt_j.navState.localCoordinates(estimated_state_j);

        // Calculate 6-DOF Bias error (GT Bias at j vs GT Bias at i)
        Vector6 error_bias = gt_j.bias.vector() - gt_i.bias.vector();

        // Stack errors (Pose, Vel, Bias order assumed based on common GTSAM factors)
        Vector15 error15;
        error15 << error_nav, error_bias;

        // --- Get 15x15 Covariance ---
        Matrix P15 = pim.preintMeasCov();

        if (P15.rows() != 15 || P15.cols() != 15) {
             cerr << "Error: Expected 15x15 covariance matrix, but got "
                  << P15.rows() << "x" << P15.cols() << " at t_i=" << current_t_i << endl;
             numerical_issue_count++;
             current_t_i += deltaTij;
             continue;
        }

        // --- Calculate 15-DOF NEES ---
        try {
            // Consider using pseudoInverse or adding diagonal regularization if inversion fails often
            Matrix P15_inv = P15.inverse();
            double nees15 = error15.transpose() * P15_inv * error15;

            if (!isnan(nees15) && !isinf(nees15) && nees15 >= 0) {
                nees_results.push_back(nees15);
            } else {
                // cout << "Warning: Skipping NEES calculation due to NaN/Inf/Negative value=" << nees15 << " at t_i=" << current_t_i << endl;
                numerical_issue_count++;
            }
        } catch (const std::exception& e) {
            // cout << "Warning: Skipping NEES calculation due to exception (likely matrix inversion) at t_i=" << current_t_i << ": " << e.what() << endl;
             numerical_issue_count++;
        }

        // Move to the start of the next non-overlapping interval
        current_t_i += deltaTij;

    } // end while loop over intervals

    // 4. Calculate Median NEES and other stats
    cout << "\nProcessed " << nees_results.size() << " valid intervals." << endl;
    cout << "Skipped " << skipped_intervals << " intervals due to data gaps." << endl;
    cout << "Skipped " << numerical_issue_count << " NEES calculations due to numerical issues." << endl;

    if (nees_results.empty()) {
        cerr << "No valid NEES results were calculated." << endl;
        return 1;
    }

    double median_nees = calculateMedian(nees_results);
    double mean_nees = 0;
    for(double n : nees_results) mean_nees += n;
    mean_nees /= nees_results.size();

    double median_dt = calculateMedian(time_diffs);
    double mean_dt = 0;
     for(double d : time_diffs) mean_dt += d;
    if (!time_diffs.empty()) mean_dt /= time_diffs.size();


    cout << "\n--- Results (15-DOF) ---" << endl;
    cout << "Dataset: " << euroc_dataset_path << endl;
    cout << "Target Preintegration Interval (deltaTij): " << fixed << setprecision(5) << deltaTij << " s" << endl;
    cout << "Actual Median Integrated Interval: " << fixed << setprecision(5) << median_dt << " s" << endl;
    cout << "Actual Mean Integrated Interval:   " << fixed << setprecision(5) << mean_dt << " s" << endl;
    cout << "Number of Valid NEES Intervals: " << nees_results.size() << endl;
    cout << "Median 15-DOF NEES: " << fixed << setprecision(3) << median_nees << endl;
    cout << "Mean 15-DOF NEES:   " << fixed << setprecision(3) << mean_nees << endl;
    cout << "------------------------" << endl;
    cout << "\nNote: This calculates 15-DOF NEES (NavState + Bias)." << endl;
    cout << "Expected Median/Mean NEES for 15-DOF is approx 15.0 for a consistent system." << endl;
    cout << "Compare results to Table I in Fornasier et al. (arXiv:2411.05548v4)." << endl;
    cout << "Noise parameters likely require significant tuning!" << endl;

    return 0;
}
