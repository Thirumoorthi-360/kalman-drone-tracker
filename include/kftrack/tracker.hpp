#pragma once
// Multi-target tracker: one constant-velocity Kalman filter per vehicle, plus
// gating, Hungarian assignment (as a matching cascade) and track management.

#include <vector>

#include "kftrack/kalman_filter.hpp"
#include "kftrack/types.hpp"

namespace kf {

struct TrackerConfig {
  double dt = 0.5;            // seconds between frames
  double measSigma = 1.5;     // detector position noise (m)      -> R
  double accelSigma = 1.5;    // assumed random acceleration (m/s²) -> Q
  double initVelSigma = 15.0; // velocity uncertainty of a brand-new track (m/s)
  double gate = 9.21;         // chi-square 99% threshold, 2 dof
  int confirmHits = 3;        // hits before a track is confirmed and shown
  int maxCoast = 14;          // max frames a mature track survives without detections
  bool coastByHits = true;    // a young track may coast at most as many frames as it has been seen
  int matureHits = 6;         // ...where "young" means fewer than this many hits
  double maxPosSigma = 12.0;  // delete a track once its position uncertainty (1σ, m) exceeds this
  double fovMargin = 10.0;    // tracks are deleted once this far outside the camera view
};

// State x = [px, py, vx, vy]
using TrackFilter = KalmanFilter<4, 2>;

struct Track {
  int uid = -1;          // internal id (every track, including tentative ones)
  int label = -1;        // public id shown to the operator, assigned on confirmation
  bool confirmed = false;
  bool updated = true;   // false while coasting
  int hits = 1;
  int misses = 0;
  TrackFilter kf;
};

struct TrackSnapshot {
  int uid = -1;
  int label = -1;
  bool confirmed = false;
  bool updated = false;
  Vector<4> x;
  Matrix<4, 4> P;
};

class MultiTargetTracker {
 public:
  explicit MultiTargetTracker(const TrackerConfig& cfg = TrackerConfig());

  // Process one frame of detections. Returns the state of every live track.
  std::vector<TrackSnapshot> step(const std::vector<Vec2>& detections, const Box& fov);

  const std::vector<Track>& tracks() const { return tracks_; }
  const TrackerConfig& config() const { return cfg_; }

 private:
  Track makeTrack(const Vec2& z);

  TrackerConfig cfg_;
  Matrix<4, 4> F_, Q_;
  Matrix<2, 4> H_;
  Matrix<2, 2> R_;
  std::vector<Track> tracks_;
  int nextLabel_ = 1;
  int nextUid_ = 0;
};

}  // namespace kf
