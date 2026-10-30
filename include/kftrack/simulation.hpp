#pragma once
// Simulated drone surveillance scene: vehicles driving through a crossroads,
// observed by a drifting camera with a noisy, imperfect detector.

#include <string>
#include <vector>

#include "kftrack/types.hpp"

namespace kf {

struct Waypoint {
  double x = 0, y = 0;
  double dwell = 0.0;        // seconds to wait at this point (e.g. a stop line)
  double speedLimit = -1.0;  // slow to this speed when passing (e.g. a turn); <0 = no limit
};

struct Vehicle {
  std::string name;
  double speed = 10.0;  // cruise speed (m/s)
  double start = 0.0;   // time it enters the map (s)
  std::vector<Waypoint> path;

  // Ground-truth trajectory sampled finely, filled by Scene::finalize()
  std::vector<double> ts;
  std::vector<Vec2> positions, velocities;
};

struct TruthState {
  int vid = -1;
  Vec2 pos, vel;
  bool inView = false;
  bool hidden = false;  // under tree cover
};

struct Frame {
  int index = 0;
  double t = 0.0;
  Box fov;
  std::vector<TruthState> truth;
  std::vector<Vec2> detections;
  std::vector<bool> isClutter;  // ground-truth label, NOT given to the tracker
};

struct SensorConfig {
  double pDetect = 0.9;      // chance a visible vehicle is detected each frame
  double measSigma = 1.5;    // detection position noise (m)
  double clutterRate = 1.5;  // mean false alarms per frame
  unsigned seed = 1;
};

struct Scene {
  double dt = 0.5;
  double tEnd = 60.0;
  Box area{0, 400, 0, 300};
  Box trees{250, 305, 132, 168};
  double fovWidth = 300, fovHeight = 230;
  double accel = 2.0, brake = 3.0;  // vehicle acceleration / braking (m/s²)
  std::vector<Vehicle> vehicles;

  int numFrames() const { return static_cast<int>(tEnd / dt); }
  Box droneFov(double t) const;
  bool vehicleState(int vid, double t, Vec2& pos, Vec2& vel) const;
  void finalize();  // builds every vehicle's trajectory
};

// The five-vehicle crossroads used in the notebook version.
Scene makeCrossroadsScene();

std::vector<Frame> simulateDrone(const Scene& scene, const SensorConfig& cfg);

}  // namespace kf
