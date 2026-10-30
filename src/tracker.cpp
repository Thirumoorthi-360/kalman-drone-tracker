#include "kftrack/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include "kftrack/hungarian.hpp"

namespace kf {

namespace {
constexpr double kInfeasible = 1e6;  // cost for pairs outside the gate
}

MultiTargetTracker::MultiTargetTracker(const TrackerConfig& cfg) : cfg_(cfg) {
  const double dt = cfg_.dt;

  // Constant-velocity motion model
  F_ = Matrix<4, 4>{1, 0, dt, 0,
                    0, 1, 0,  dt,
                    0, 0, 1,  0,
                    0, 0, 0,  1};

  // Q = G Gᵀ σa²  — a random acceleration a moves position by a·dt²/2 and velocity by a·dt
  const Matrix<4, 2> G{dt * dt / 2, 0,
                       0,           dt * dt / 2,
                       dt,          0,
                       0,           dt};
  Q_ = G * G.transpose() * (cfg_.accelSigma * cfg_.accelSigma);

  // The camera measures position only
  H_ = Matrix<2, 4>{1, 0, 0, 0,
                    0, 1, 0, 0};
  R_ = Matrix<2, 2>::Identity() * (cfg_.measSigma * cfg_.measSigma);
}

Track MultiTargetTracker::makeTrack(const Vec2& z) {
  Track t;
  t.uid = nextUid_++;
  t.kf.F = F_;
  t.kf.Q = Q_;
  t.kf.H = H_;
  t.kf.R = R_;
  // Position from the first detection; velocity unknown (0 with a large variance)
  t.kf.x = Vector<4>{z[0], z[1], 0.0, 0.0};
  const double s2 = cfg_.measSigma * cfg_.measSigma;
  const double v2 = cfg_.initVelSigma * cfg_.initVelSigma;
  t.kf.P = Matrix<4, 4>::Diagonal({s2, s2, v2, v2});
  return t;
}

std::vector<TrackSnapshot> MultiTargetTracker::step(const std::vector<Vec2>& detections, const Box& fov) {
  const int nT = static_cast<int>(tracks_.size());
  const int nD = static_cast<int>(detections.size());

  // 1. Predict every track forward one frame
  for (Track& t : tracks_) t.kf.predict();

  // 2. Gating and cost. A pair is allowed only if the Mahalanobis distance
  //      d² = νᵀ S⁻¹ ν  <  gate
  //    The cost used to choose between allowed pairs is the Gaussian negative
  //    log-likelihood  d² + ln|S|. Plain d² would favour vague tracks (a huge S makes
  //    every detection look close); the ln|S| term penalises that vagueness.
  std::vector<std::vector<double>> cost(nT, std::vector<double>(nD, kInfeasible));
  for (int i = 0; i < nT; ++i) {
    const TrackFilter& f = tracks_[i].kf;
    const Matrix<2, 2> S = f.innovationCovariance();
    const Matrix<2, 2> Sinv = inverse(S);
    const double logDetS = std::log(S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0));
    for (int j = 0; j < nD; ++j) {
      const Vec2 nu = f.innovation(detections[j]);
      const double d2 = (nu.transpose() * Sinv * nu)(0, 0);
      if (d2 < cfg_.gate) cost[i][j] = d2 + logDetS;
    }
  }
  auto feasible = [](double c) { return c < kInfeasible / 2; };

  // 3. Assignment as a matching cascade: confirmed tracks seen last frame choose
  //    first, then tracks that have been coasting longer, then all tentative tracks
  //    together. This stops a stale track stealing a healthy track's detection.
  std::vector<bool> trackMatched(nT, false), detMatched(nD, false);
  auto levelOf = [](const Track& t) { return t.confirmed ? std::make_pair(0, t.misses) : std::make_pair(1, 0); };
  std::set<std::pair<int, int>> levels;
  for (const Track& t : tracks_) levels.insert(levelOf(t));

  for (const auto& level : levels) {
    std::vector<int> idx, cand;
    for (int i = 0; i < nT; ++i)
      if (levelOf(tracks_[i]) == level) idx.push_back(i);
    for (int j = 0; j < nD; ++j)
      if (!detMatched[j]) cand.push_back(j);
    if (cand.empty()) break;

    std::vector<std::vector<double>> sub(idx.size(), std::vector<double>(cand.size()));
    for (size_t a = 0; a < idx.size(); ++a)
      for (size_t b = 0; b < cand.size(); ++b) sub[a][b] = cost[idx[a]][cand[b]];

    const std::vector<int> assign = solveAssignment(sub);
    for (size_t a = 0; a < idx.size(); ++a) {
      const int b = assign[a];
      if (b < 0 || !feasible(sub[a][b])) continue;
      const int i = idx[a], j = cand[b];
      trackMatched[i] = true;
      detMatched[j] = true;
      // 4. Kalman update with the assigned detection
      Track& t = tracks_[i];
      t.kf.update(detections[j]);
      t.hits += 1;
      t.misses = 0;
    }
  }

  // 5. Track management
  for (int i = 0; i < nT; ++i) {
    Track& t = tracks_[i];
    t.updated = trackMatched[i];
    if (!trackMatched[i]) t.misses += 1;  // coast: keep the prediction, no update
    if (!t.confirmed && t.hits >= cfg_.confirmHits) {
      t.confirmed = true;
      t.label = nextLabel_++;
    }
  }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                               [&](const Track& t) {
                                 // Young tracks have earned less trust, so they may coast for less time
                                 int limit = t.confirmed ? cfg_.maxCoast : 2;
                                 if (t.confirmed && cfg_.coastByHits && t.hits < cfg_.matureHits)
                                   limit = std::min(limit, std::max(2, t.hits));
                                 const double posSigma = std::sqrt(0.5 * (t.kf.P(0, 0) + t.kf.P(1, 1)));
                                 return t.misses >= limit || posSigma > cfg_.maxPosSigma ||
                                        !fov.contains(t.kf.x[0], t.kf.x[1], cfg_.fovMargin);
                               }),
                tracks_.end());

  // Unmatched detections start new tentative tracks
  for (int j = 0; j < nD; ++j)
    if (!detMatched[j]) tracks_.push_back(makeTrack(detections[j]));

  std::vector<TrackSnapshot> out;
  out.reserve(tracks_.size());
  for (const Track& t : tracks_) out.push_back({t.uid, t.label, t.confirmed, t.updated, t.kf.x, t.kf.P});
  return out;
}

}  // namespace kf
