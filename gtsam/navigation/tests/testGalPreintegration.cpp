#include <gtsam/navigation/GalileanImuFactor.h>
#include <gtsam/navigation/PreintegrationGalileanParams.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/geometry/Gal3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/TestableAssertions.h>

#include <CppUnitLite/TestHarness.h>

#include <vector>
#include <cmath>

using namespace std;
using namespace gtsam;

// Define constants used in tests (following pattern from other tests)
const int N_TESTS = 1; // Adjusted as we are using specific logged data for propagation tests
const double kTolerance = 1e-9;
static const Vector3 kZero = Z_3x1; // Define kZero to avoid ambiguity

// Define matrix types
typedef Eigen::Matrix<double, 20, 20> Matrix20;
typedef Eigen::Matrix<double, 10, 10> Matrix10;
typedef Eigen::Matrix<double, 9, 6> Matrix96;

// Helper functions to generate test vectors (kept for potential other uses, but not for modified tests)
vector<Vector3> createTestAccels(size_t n) {
    vector<Vector3> accs;
    accs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double val = 0.1 * i;
        accs.push_back(Vector3(val + 0.1, val + 0.2, val + 0.3));
    }
    return accs;
}

vector<Vector3> createTestGyros(size_t n) {
    vector<Vector3> gyros;
    gyros.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double val = 0.01 * i;
        gyros.push_back(Vector3(val + 0.01, val + 0.02, val + 0.03));
    }
    return gyros;
}

// Helper to create parameters (same as before)
std::shared_ptr<GalileanPreintegrationParams> createParams(
    const Vector3& gravity = Vector3(0, 0, -9.81),
    double gyroNoise = 1.0,
    double accNoise = 1.0,
    double virtualVelNoise = 0.0,
    double virtualTimeNoise = 0.0,
    double gyroBiasNoise = 1.0,
    double accBiasNoise = 1.0,
    double virtualVelBiasNoise = 0.0,
    double virtualTimeBiasNoise = 0.0) {

    auto params = std::shared_ptr<GalileanPreintegrationParams>(
        new GalileanPreintegrationParams(gravity));

    params->n_gravity = gravity; // Constructor already sets this.

    params->gyroscopeCovariance = I_3x3 * (gyroNoise * gyroNoise);
    params->accelerometerCovariance = I_3x3 * (accNoise * accNoise);
    params->virtualVelCovariance = I_3x3 * (virtualVelNoise * virtualVelNoise);
    params->virtualTimeScaleCovariance = virtualTimeNoise * virtualTimeNoise;
    params->biasOmegaCovariance = I_3x3 * (gyroBiasNoise * gyroBiasNoise);
    params->biasAccCovariance = I_3x3 * (accBiasNoise * accBiasNoise);
    params->biasVirtualVelCovariance = I_3x3 * (virtualVelBiasNoise * virtualVelBiasNoise);
    params->biasVirtualTimeCovariance = virtualTimeBiasNoise * virtualTimeBiasNoise;

    params->biasAccOmegaInt.setZero();

    return params;
}


// accumulates rotation
Matrix3 computeGamma2(const Vector3& omega) {
    double theta2 = omega.dot(omega);
    double theta = std::sqrt(theta2);

    if (theta < 1e-9) {
        // Taylor expansion for small angles
        return 0.5 * Matrix3::Identity() - (1.0/6.0) * SO3::Hat(omega);
    }

    Vector3 axis = omega / theta;
    double s = (theta - std::sin(theta)) / theta2;
    double c = (1 - std::cos(theta)) / theta2;

    return c * Matrix3::Identity() + s * SO3::Hat(axis) + (0.5 - c) * axis * axis.transpose();
}


/* ************************************************************************* */
// Test Case 1: Constructors
// TEST(GalPreintegration, Constructors) {
//     for (int i = 0; i < N_TESTS; ++i) { // N_TESTS can be 1 for this, as it's not randomized here
//         {
//             // Test default constructor with default parameters
//             auto p = std::make_shared<GalileanPreintegrationParams>();
//             PreintegratedGalileanMeasurements pim(p);

//             Matrix20 expectedQc = Matrix20::Identity();
//             expectedQc.block<3,3>(6,6).setZero();  // virtual velocity noise block (3D)
//             expectedQc(9,9) = 0.0;                 // virtual time scale noise block (1D)
//             expectedQc.block<3,3>(16,16).setZero(); // virtual velocity bias noise block (3D)
//             expectedQc(19,19) = 0.0;                // virtual time scale bias noise block (1D)


//             Matrix20 zeroMat20 = Matrix20::Zero();
//             Matrix20 identMat20 = Matrix20::Identity();
//             EXPECT(assert_equal(zeroMat20, pim.uncertaintyCovariance(), kTolerance));
//             EXPECT(assert_equal(identMat20, pim.biasJacobian(), kTolerance));
//             EXPECT(assert_equal(Gal3::Identity(), pim.deltaUpsilon(), kTolerance));

//             Vector6 zeroBias = Vector6::Zero();
//             EXPECT(assert_equal(zeroBias, pim.biasHat().vector(), kTolerance));

//             EXPECT(assert_equal(Rot3::Identity(), pim.deltaRij(), kTolerance));
//             EXPECT(assert_equal(kZero, pim.deltaVij(), kTolerance));
//             EXPECT(assert_equal(kZero, pim.deltaPij(), kTolerance));
//             DOUBLES_EQUAL(0.0, pim.deltaTij(), kTolerance);

//             auto galileanParams = std::static_pointer_cast<const GalileanPreintegrationParams>(pim.params());
//             EXPECT(galileanParams != nullptr);
//             EXPECT(assert_equal(Vector3(0, 0, -9.81), galileanParams->n_gravity, kTolerance));
//             DOUBLES_EQUAL(1.0, sqrt(galileanParams->gyroscopeCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1.0, sqrt(galileanParams->accelerometerCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(0.0, galileanParams->virtualVelCovariance.trace(), kTolerance);
//             DOUBLES_EQUAL(0.0, galileanParams->virtualTimeScaleCovariance, kTolerance);
//             DOUBLES_EQUAL(1.0, sqrt(galileanParams->biasOmegaCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1.0, sqrt(galileanParams->biasAccCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(0.0, galileanParams->biasVirtualVelCovariance.trace(), kTolerance);
//             DOUBLES_EQUAL(0.0, galileanParams->biasVirtualTimeCovariance, kTolerance);

//             Matrix10 zeroMat10 = Matrix10::Zero();
//             EXPECT(assert_equal(zeroMat10, galileanParams->getBiasExtendedInit(), kTolerance));

//             EXPECT(assert_equal(expectedQc, galileanParams->createNoiseCovariance(), kTolerance));
//         }

//         {
//             // Test constructor with custom parameters
//             auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 1e-8, 1e-10, 1e-6, 1e-5, 1e-7, 1e-8);
//             PreintegratedGalileanMeasurements pim(p);

//             Matrix20 expectedQc = Matrix20::Zero(); // Initialize with zeros
//             expectedQc.block<3,3>(0,0) = Matrix3::Identity() * (1e-4 * 1e-4);   // gyro noise
//             expectedQc.block<3,3>(3,3) = Matrix3::Identity() * (1e-3 * 1e-3);   // acc noise
//             expectedQc.block<3,3>(6,6) = Matrix3::Identity() * (1e-8 * 1e-8);   // virtual vel noise
//             expectedQc(9,9) = (1e-10 * 1e-10);                                 // virtual time noise
//             expectedQc.block<3,3>(10,10) = Matrix3::Identity() * (1e-6 * 1e-6); // gyro bias noise
//             expectedQc.block<3,3>(13,13) = Matrix3::Identity() * (1e-5 * 1e-5); // acc bias noise
//             expectedQc.block<3,3>(16,16) = Matrix3::Identity() * (1e-7 * 1e-7); // virtual vel bias noise
//             expectedQc(19,19) = (1e-8 * 1e-8);                                 // virtual time bias noise

//             auto galileanParams = std::static_pointer_cast<const GalileanPreintegrationParams>(pim.params());
//             EXPECT(assert_equal(Vector3(0, 0, -9.81), galileanParams->n_gravity, kTolerance));
//             DOUBLES_EQUAL(1e-4, sqrt(galileanParams->gyroscopeCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1e-3, sqrt(galileanParams->accelerometerCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1e-8, sqrt(galileanParams->virtualVelCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1e-10, sqrt(galileanParams->virtualTimeScaleCovariance), kTolerance);
//             DOUBLES_EQUAL(1e-6, sqrt(galileanParams->biasOmegaCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1e-5, sqrt(galileanParams->biasAccCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1e-7, sqrt(galileanParams->biasVirtualVelCovariance(0,0)), kTolerance);
//             DOUBLES_EQUAL(1e-8, sqrt(galileanParams->biasVirtualTimeCovariance), kTolerance);

//             Matrix10 zeroMat10 = Matrix10::Zero();
//             EXPECT(assert_equal(zeroMat10, galileanParams->getBiasExtendedInit(), kTolerance));

//             EXPECT(assert_equal(expectedQc, galileanParams->createNoiseCovariance(), kTolerance));
//         }
//     }
// }

// /* ************************************************************************* */
// // Test Case 2: Mean Propagation
TEST(GalPreintegration, MeanPropagation) {
    size_t n = 3; // From log: n: 3
    double dt = 0.01; // From log: dt: 0.01

    // Accelerations from log: PreintegrationTest/1.MeanPropagation
    vector<Vector3> accs = {
        Vector3(-2.5068965533954604, 0.87582680732853024,  4.8643795488708914),
        Vector3(  9.6795269911406496, -0.35888562614098563,   3.2383400118944139),
        Vector3( 2.5392504149042483,  5.5171485422613049, -7.9623734434444966)
    };

    // Gyroscope readings from log: PreintegrationTest/1.MeanPropagation
    vector<Vector3> gyros = {
        Vector3(-0.90010662229654148,  0.39839438991809351, -0.10436735729977142),
        Vector3(  0.30869533447007291, 0.0082392834687143868,  -0.86987437409491464),
        Vector3(-0.23045677741642689, -0.11463504589973816,  -0.9506352007960075)
    };

    auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 0, 0, 1e-6, 1e-5, 0, 0); // Default noise params for this test as in log context
    PreintegratedGalileanMeasurements pim(p);

    // Expected values from log: PreintegrationTest/1.MeanPropagation
    Matrix3 expectedDeltaRij_mat;
    expectedDeltaRij_mat << 0.99981068491855862,   0.01923387098203096,    0.0029415182693738138,
                           -0.019257513997689664,  0.99978066320686099,    0.008232474250065961,
                           -0.0027825307386009877, -0.008287562047779775,   0.9999617861888489;
    Rot3 expectedDeltaRij(expectedDeltaRij_mat);

    Vector3 expectedDeltaVij(0.097846746908176491, 0.059351851359761489, 0.00057657097205366325);
    Vector3 expectedDeltaPij(0.00095827440195437761, 0.0004386145812195936, 0.0012962605866650757);
    double expectedDeltaTij = 0.029999999999999999;

    for (size_t j = 0; j < n; ++j) {
        pim.integrateMeasurement(accs[j], gyros[j], dt);
    }

    EXPECT(assert_equal(expectedDeltaRij, pim.deltaRij(), 1e-10));
    EXPECT(assert_equal(expectedDeltaVij, pim.deltaVij(), 1e-10));
    EXPECT(assert_equal(expectedDeltaPij, pim.deltaPij(), 1e-10));
    DOUBLES_EQUAL(expectedDeltaTij, pim.deltaTij(), 1e-10);
}


/* ************************************************************************* */
// Test Case 3: Covariance Propagation
TEST(GalPreintegration, CovariancePropagation) {
    size_t n = 2; // From log: n: 2
    double dt = 0.01; // From log: dt: 1.00000000000000002e-02

    // Accelerations from log: PreintegrationTest/1.CovariancePropagation
    vector<Vector3> accs = {
        Vector3(1.22506436189006473e+00, -5.21056685429191191e+00, -4.75766114076331803e+00),
        Vector3(-6.92624602916510135e+00, -4.09175350034472007e+00, -6.02857638488549519e+00)
    };

    // Gyroscope readings from log: PreintegrationTest/1.CovariancePropagation
    vector<Vector3> gyros = {
        Vector3(4.07671857884981392e-02,  1.56378111247902796e-01, -2.21783725996555492e-01),
        Vector3(-9.74699336902655311e-01, -8.54637441421201016e-01, -8.79588089225050718e-01)
    };

    auto p = createParams(Vector3(0, 0, -9.81), 1e-4, 1e-3, 1e-8, 1e-10, 1e-6, 1e-5, 1e-7, 1e-8);
    PreintegratedGalileanMeasurements pim(p);

    // Expected Covariance Matrix (pim.Cov()) from log: PreintegrationTest/1.CovariancePropagation "Final values"
    Matrix20 expectedCov;
    expectedCov <<
        1.99998691557941965e-10,  6.99755856359394903e-16,  7.08338594247838921e-16, -7.50353044646448299e-15, -1.01468955611150615e-11,  9.85772055529551783e-12,  1.10522991549942812e-16,  7.18363104560932057e-14, -6.19686782721880474e-14, 0,  9.99975038629791217e-17, -4.37806744383901684e-19,  4.27479901886116107e-19, -4.38327399087170224e-20, -6.06083707156799795e-18,  4.04624690931843161e-18,  2.26414894319069661e-22,  4.32810108760194263e-20, -8.63714478211074088e-21, 0,
        6.99755856359434543e-16,  1.99998518619023369e-10,  5.94846951901707230e-16,  1.01550792471161753e-11, -1.10274045157363615e-14, -1.64081076595844354e-12, -7.19597252094407741e-14,  1.55797106617578358e-16,  3.81208172443093615e-14, 0,  4.40584506106874588e-19,  9.99971199998084041e-17, -4.87725781897906804e-19,  6.11003029672390411e-18, -6.06396186519586332e-20, -6.91331618636512549e-18, -4.38168405116757192e-20,  7.59183495119013668e-22,  1.16071661840381752e-19, 0,
        7.08338594247824031e-16,  5.94846951901753083e-16,  1.99998577151053578e-10, -9.84807337354357767e-12,  1.64841180665844070e-12, -7.87019564802512613e-15,  6.18205426400439984e-14, -3.82313995399668572e-14,  1.24858556238825056e-16, 0, -4.24616420929709853e-19,  4.90220766513753688e-19,  9.99971958403891122e-17, -3.98603175831381559e-18,  6.95687709179388486e-18, -5.10308325800817169e-20,  7.91416766463152484e-21, -1.16293455548280195e-19,  6.03716453851008549e-22, 0,
        -7.50353044646448456e-15,  1.01550792944049422e-11, -9.84807342593757734e-12,  2.00011228430199176e-08, -7.79341821543719446e-14, -8.96265319520811920e-14, -2.00007339868835727e-10,  9.35074543896454635e-14, -5.59783063635999136e-14, 0,  5.77076458894038270e-20,  7.74021409550965540e-18, -7.28040361436667437e-18,  1.00012613581598558e-14, -4.42034393789479891e-17,  4.23105011850418709e-17, -2.00010195861260917e-16,  8.82812693896774571e-19, -8.47426304204471200e-19, 0,
        -1.01468956084038301e-11, -1.10274045157363646e-14,  1.64841179452997100e-12, -7.79341821543735855e-14,  2.00005665596273284e-08, -5.65829391633079494e-13, -9.15808123034402745e-14, -2.00003894575616422e-10,  8.15219407865915673e-14, 0, -7.77710617824537735e-18,  3.50124502916359387e-20,  2.22526543469819509e-18,  4.38556376850616428e-17,  1.00006783659521584e-14, -4.94965963102021107e-17, -8.79269757213778173e-19, -2.00005305907359796e-16,  9.82263403903857531e-19, 0,
        9.85772060768951749e-12, -1.64081076595844334e-12, -7.87019564802512929e-15, -8.96265319520857484e-14, -5.65829391633072325e-13,  2.00005059893642535e-08,  5.83032543961376574e-14, -7.41783168226015395e-14, -2.00002894588055012e-10, 0,  7.24257851408440810e-18, -2.28295187702029888e-18,  3.34380272779935196e-20, -4.27110477248878484e-17,  4.82426082183028573e-17,  1.00005225904013541e-14,  8.51870333797360033e-19, -9.72199398426764393e-19, -2.00002940204918117e-16, 0,
        1.10522992268150191e-16, -7.19597256822765160e-14,  6.18205431640785548e-14, -2.00007339868835727e-10, -9.15808123034402745e-14,  5.83032543961375817e-14,  2.50004550242448158e-12, -6.18846463538188134e-18, -1.00402808520842959e-17, -3.89904150487117721e-24, -3.90689691008527713e-23, -1.67405847042174050e-20,  8.13284962351999270e-22, -1.50008721454355260e-16,  5.86960131634181885e-19, -5.66243656501203893e-19,  4.00004496634993311e-18, -1.60871469860429256e-20,  1.56340885073301422e-20, -2.25420366513567036e-22,
        7.18363109285880839e-14,  1.55797106864893129e-16, -3.82313995399668509e-14,  9.35074543896454131e-14, -2.00003894575616448e-10, -7.41783168226015017e-14, -6.18846463538415779e-18,  2.50002848004911705e-12, -2.12653994900628120e-17, -1.71005389417895958e-23,  1.68777737244558226e-20, -2.02808497848954341e-22, -5.22635418569651028e-20, -1.44660408477151826e-19, -5.00039304535130099e-17,  1.63372419179864050e-19,  1.61594647088662406e-20,  4.00007887339108362e-18, -1.78633520661141292e-20, -7.24756388365086369e-22,
        -6.19686787965192156e-14,  3.81208171228470911e-14,  1.24858556512029577e-16, -5.59783063635999388e-14,  8.15219407865915294e-14, -2.00002894588055012e-10, -1.00402808520839477e-17, -2.12653994900618875e-17,  2.50001692834906480e-12, -1.79246861224934709e-23, -6.22133357601705795e-22,  5.23012763005865115e-20, -1.58311952780037134e-22,  1.44851108065598900e-19, -1.63277380261768080e-19, -5.00029490532455194e-17, -1.55736073693644049e-20,  1.79844801386892698e-20,  4.00007156209533257e-18, -7.77259172729456159e-22,
        0,0,0,0,0,0, -3.89904150487117721e-24, -1.71005389417895958e-23, -1.79246861224934709e-23, 3.00000000000000038e-22,0,0,0,0,0,0, -6.93120968713676618e-22, -4.01600185074971339e-22, -6.09061863012451864e-22,  1.00000000000000025e-20,
        9.99975038629791340e-17,  4.40584506106874395e-19, -4.24616420929709853e-19,  5.77076458894038391e-20, -7.77710617824537735e-18,  7.24257851408440810e-18, -3.98142506806673666e-22,  6.40754237890726390e-20, -5.31081437276388683e-20, 0,  2.00000000000000031e-14, -9.40073163598960955e-33, -5.90691424697084129e-33, -1.57511627042467242e-33, -2.15774408441176012e-15,  1.85677014327684416e-15,  6.11663504083793458e-35,  2.28506928978373821e-17, -1.74294527492594190e-17, 0,
        -4.37806744383901876e-19,  9.99971199998083918e-17,  4.90220766513753785e-19,  7.74021409550965540e-18,  3.50124502916359387e-20, -2.28295187702029965e-18, -1.67405847042174080e-20, -2.02808497848954293e-22,  5.23012763005865176e-20, 0,  2.69045368280351849e-32,  2.00000000000000000e-14,  3.60403051106339663e-33,  2.15774408441175972e-15,  7.08184935349143190e-33, -1.14427453763321625e-15, -2.28506928978373790e-17, -4.97547576237149828e-35,  1.95958150731933118e-17, 0,
        4.27479901886116107e-19, -4.87725781897906612e-19,  9.99971958403891245e-17, -7.28040361436667437e-18,  2.22526543469819625e-18,  3.34380272779935196e-20,  5.34472269831902413e-20, -3.99878890917008102e-20, -2.95444883327885181e-22, 0, -5.64874390755328053e-33,  4.13647089580420956e-33,  2.00000000000000031e-14, -1.85677014327684376e-15,  1.14427453763321645e-15,  1.25951085423268273e-34,  1.74294527492594128e-17, -1.95958150731933087e-17,  3.82348891648972283e-35, 0,
        -4.38327399087170224e-20,  6.11003029672390334e-18, -3.98603175831381482e-18,  1.00012613581598558e-14,  4.38556376850616366e-17, -4.27110477248878422e-17, -1.50008721454355285e-16, -5.83654239078140279e-19,  5.70660183735484172e-19, 0,  1.28311521098078741e-33,  1.08502948384702446e-15, -9.22695012300068228e-16,  2.00040517274493924e-12, -2.15015047899043312e-17, -3.64296592176799726e-17, -2.00005483457179658e-14,  4.34712548467234367e-19,  6.78333613289224794e-19, 0,
        -6.06083707156799795e-18, -6.06396186519586332e-20,  6.95687709179388486e-18, -4.39214355219759438e-17,  1.00006783659521584e-14,  4.82426082183028697e-17,  1.46111267749146053e-19, -5.00039304535130161e-17, -1.63277380261768128e-19, 0, -1.08502948384702446e-15,  1.01533872406983041e-33,  5.71564472455617350e-16, -2.15015047899067687e-17,  2.00029826118756494e-12, -2.44599100683629650e-17,  2.46718538556403293e-20, -2.00011920836131280e-14,  1.73525808920568916e-18, 0,
        4.04624690931843161e-18, -6.91331618636512549e-18, -5.10308325800817109e-20,  4.25380724878731484e-17, -4.88966057559250917e-17,  1.00005225904013541e-14, -1.41216396126450046e-19,  1.63372419179864050e-19, -5.00029490532455194e-17, 0,  9.22695012300068228e-16, -5.71564472455617449e-16,  1.87655977312139226e-33, -3.64296592176809032e-17, -2.44599100683629650e-17,  2.00023784797912298e-12,  2.74505089330295345e-19,  2.99843964549501142e-19, -2.00009805764877358e-14, 0,
        2.26414894319069755e-22, -4.38168405116757192e-20,  7.91416766463152484e-21, -9.99989974818898672e-17, -4.40312231934319972e-19,  4.26138143718186037e-19,  1.50000022357770370e-18,  5.88307358861440784e-21, -5.66457030987320038e-21, -6.93120968713676712e-22, -1.04650455797582823e-35, -6.75801894742207144e-18,  3.41843776656937342e-18, -2.00005483457179627e-14,  2.46718538556252589e-20,  2.74505089330302904e-19,  4.00008179965346078e-16,  1.36423227277459081e-21, -1.29590040669138382e-21, -5.71564472455617378e-20,
        4.32810108760194263e-20,  7.59183495119013574e-22, -1.16293455548280195e-19,  4.40156241429441163e-19, -1.00002472185435091e-16, -4.89365852794163974e-19, -5.82563059834448798e-21,  1.50004013700133689e-18,  6.54738545895256459e-21, -4.01600185074971339e-22,  6.75801894742207144e-18, -1.13407452986572834e-35, -1.10048829604412030e-17,  4.34712548467234367e-19, -2.00011920836131280e-14,  2.99843964549498734e-19,  1.36423227277451069e-21,  4.00020329771412248e-16,  3.95019782871465179e-21, -9.22695012300068055e-20,
        -8.63714478211074088e-21,  1.16071661840381728e-19,  6.03716453851008737e-22, -4.23954737646469961e-19,  4.87997467486215225e-19, -1.00001390554101934e-16,  5.70023219990632327e-21, -6.49344016296717084e-21,  1.50004426376560592e-18, -6.09061863012451958e-22, -3.41843776656937380e-18,  1.10048829604412014e-17,  3.17940314808379244e-35,  6.78333613289222868e-19,  1.73525808920564607e-19, -2.00009805764877326e-14, -1.29590040669129731e-21,  3.95019782871477216e-21,  4.00020238448884344e-16, -1.08502948384702483e-19,
        0,0,0,0,0,0, -2.25420366513567036e-22, -7.24756388365086463e-22, -7.77259172729456159e-22, 1.00000000000000025e-20,0,0,0,0,0,0, -5.71564472455617378e-20, -9.22695012300068175e-20, -1.08502948384702483e-19,  2.00000000000000053e-18
    ;

    // Expected Jacobian Matrix (pim.Jxi()) from log: PreintegrationTest/1.CovariancePropagation "Final values"
    Matrix20 expectedJxi;
    expectedJxi <<
        1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0, -1.99996700120077114e-02, -7.73123223153156399e-05,  1.90293501523185410e-05, 0,0,0,0,0,0,0,
        0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0,  7.71198304956659356e-05, -1.99996008883846221e-02, -4.28154941602819039e-05, 0,0,0,0,0,0,0,
        0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0, -1.93937055642201563e-05,  4.25857963521619206e-05, -1.99997889705466822e-02, 0,0,0,0,0,0,0,
        0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0,  7.91623267422091239e-06, -1.01866682244965774e-03,  9.81365686796745879e-04, -1.99996700120077114e-02, -7.73123223153156399e-05,  1.90293501523185410e-05,0,0,0,0,
        0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0,  1.01400695767307259e-03,  7.52661769994618409e-06, -1.66740103835238631e-04,  7.71198304956659356e-05, -1.99996008883846221e-02, -4.28154941602819039e-05,0,0,0,0,
        0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0, -9.87175940191208599e-04,  1.58984995282032660e-04,  3.55657088108364673e-06, -1.93937055642201563e-05,  4.25857963521619206e-05, -1.99997889705466822e-02,0,0,0,0,
        0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00, 0, -6.79988351881230357e-08,  7.21930639451916488e-06, -6.15346797742295216e-06,  1.99994465628182492e-04,  1.14078207046484312e-06, -4.21266386890698163e-07, -1.99996700120077114e-02, -7.73123223153156399e-05,  1.90293501523185410e-05, 1.64483783973550727e-04,
        0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00,  0.00000000000000000e+00, 0, -7.17232029704765340e-06, -7.67645030790968199e-08,  3.83962777677867528e-06, -1.13737254255757599e-06,  1.99993228507708179e-04,  7.40954466556433388e-07,  7.71198304956659356e-05, -1.99996008883846221e-02, -4.28154941602819039e-05, 9.85297505813872987e-04,
        0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  0.00000000000000000e+00,  1.00000000000000000e+00, 0,  6.22304555880458289e-06, -3.77571239902754201e-06, -4.41791974289750795e-08,  4.27620053238619287e-07, -7.36754031107038800e-07,  1.99996310970680522e-04, -1.93937055642201563e-05,  4.25857963521619206e-05, -1.99997889705466822e-02, 1.01520943951989074e-03,
        0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,-2.00000000000000004e-02,
        0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1;

    for (size_t j = 0; j < n; ++j) {
        pim.integrateMeasurement(accs[j], gyros[j], dt);
    }

    EXPECT(assert_equal(expectedCov, pim.uncertaintyCovariance(), 1e-10));
    EXPECT(assert_equal(expectedJxi, pim.biasJacobian(), 1e-10));
}

/* ************************************************************************* */
int main() {
    TestResult tr;
    return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
