// The simplest useful Kalman filter: track a car's position and velocity from
// noisy GPS readings of position only. Prints the filter's numbers step by step
// so you can watch the gain and uncertainty settle.
//
//   ./gps_car_1d            first 15 steps + summary
//   ./gps_car_1d 100        print all 100 steps

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "kftrack/kalman_filter.hpp"

using namespace kf;

int main(int argc, char** argv) {
  const int nSteps = 100;
  const int nPrint = argc > 1 ? std::atoi(argv[1]) : 15;
  const double dt = 1.0, accelSigma = 0.5, gpsSigma = 8.0;

  // ---- simulate the truth and the noisy GPS ----
  std::mt19937 rng(0);
  std::normal_distribution<double> accel(0.0, accelSigma), gpsNoise(0.0, gpsSigma);
  const Matrix<2, 2> F{1, dt,
                       0, 1};
  const Matrix<2, 1> G{0.5 * dt * dt,
                       dt};
  Vector<2> truth{0.0, 10.0};  // starts at 0 m, 10 m/s
  std::vector<Vector<2>> truths;
  std::vector<double> gps;
  for (int k = 0; k < nSteps; ++k) {
    truth = F * truth + G * accel(rng);
    truths.push_back(truth);
    gps.push_back(truth[0] + gpsNoise(rng));
  }

  // ---- the Kalman filter ----
  KalmanFilter<2, 1> kf;
  kf.F = F;
  kf.Q = G * G.transpose() * (accelSigma * accelSigma);
  kf.H = Matrix<1, 2>{1, 0};
  kf.R = Matrix<1, 1>{gpsSigma * gpsSigma};
  kf.x = Vector<2>{0.0, 0.0};                    // deliberately poor initial guess
  kf.P = Matrix<2, 2>::Diagonal({500.0, 500.0});  // ...so start very uncertain

  std::printf("%4s %9s | %9s %8s | %6s %6s | %9s %8s %8s | %8s %8s\n", "k", "GPS z", "pred pos", "pred sd",
              "K_pos", "K_vel", "est pos", "est vel", "est sd", "true pos", "true vel");
  std::printf("%s\n", std::string(110, '-').c_str());

  double seGps = 0, seKf = 0;
  int nErr = 0;
  for (int k = 0; k < nSteps; ++k) {
    kf.predict();
    const double predPos = kf.x[0], predSd = std::sqrt(kf.P(0, 0));
    kf.update(Vector<1>{gps[k]});

    if (k < nPrint)
      std::printf("%4d %9.1f | %9.1f %8.2f | %6.3f %6.3f | %9.1f %8.2f %8.2f | %8.1f %8.2f\n", k, gps[k], predPos,
                  predSd, kf.lastGain(0, 0), kf.lastGain(1, 0), kf.x[0], kf.x[1], std::sqrt(kf.P(0, 0)),
                  truths[k][0], truths[k][1]);
    if (k >= 10) {  // skip the start-up transient
      seGps += std::pow(gps[k] - truths[k][0], 2);
      seKf += std::pow(kf.x[0] - truths[k][0], 2);
      ++nErr;
    }
  }

  const double rmseGps = std::sqrt(seGps / nErr), rmseKf = std::sqrt(seKf / nErr);
  std::printf("\nGPS RMSE:    %.2f m\nKalman RMSE: %.2f m  (%.0f%% better)\n", rmseGps, rmseKf,
              100 * (1 - rmseKf / rmseGps));
  std::printf("Steady-state gain: K_pos = %.3f, K_vel = %.3f\n", kf.lastGain(0, 0), kf.lastGain(1, 0));
  return 0;
}
