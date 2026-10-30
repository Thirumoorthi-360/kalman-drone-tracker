#pragma once
// Linear Kalman filter with N states and M measurements.
//
//   Model:    x_k = F x_{k-1} + w,   w ~ N(0, Q)
//             z_k = H x_k     + v,   v ~ N(0, R)
//
// Each method below is one line of the textbook equations; the comments show which.

#include "kftrack/matrix.hpp"

namespace kf {

template <int N, int M>
class KalmanFilter {
 public:
  using StateVector = Vector<N>;
  using StateMatrix = Matrix<N, N>;
  using MeasVector = Vector<M>;
  using MeasMatrix = Matrix<M, M>;
  using ObsMatrix = Matrix<M, N>;
  using GainMatrix = Matrix<N, M>;

  StateVector x;  // state estimate          x̂
  StateMatrix P;  // estimate covariance     P
  StateMatrix F;  // state transition        F
  StateMatrix Q;  // process noise           Q
  ObsMatrix H;    // observation model       H
  MeasMatrix R;   // measurement noise       R

  // Predict (time update)
  //   x̂⁻ = F x̂
  //   P⁻ = F P Fᵀ + Q
  void predict() {
    x = F * x;
    P = F * P * F.transpose() + Q;
  }

  // Innovation  ν = z − H x̂⁻
  MeasVector innovation(const MeasVector& z) const { return z - H * x; }

  // Innovation covariance  S = H P⁻ Hᵀ + R
  MeasMatrix innovationCovariance() const { return H * P * H.transpose() + R; }

  // Squared Mahalanobis distance  d² = νᵀ S⁻¹ ν  (used for gating)
  double mahalanobis2(const MeasVector& z) const {
    const MeasVector nu = innovation(z);
    return (nu.transpose() * inverse(innovationCovariance()) * nu)(0, 0);
  }

  // Update (measurement update)
  //   K  = P⁻ Hᵀ S⁻¹
  //   x̂  = x̂⁻ + K (z − H x̂⁻)
  //   P  = (I − K H) P⁻
  void update(const MeasVector& z) {
    const MeasMatrix S = innovationCovariance();
    const GainMatrix K = P * H.transpose() * inverse(S);
    x = x + K * innovation(z);
    P = (StateMatrix::Identity() - K * H) * P;
    lastGain = K;
  }

  GainMatrix lastGain;  // K from the most recent update (for inspection / plotting)
};

}  // namespace kf
