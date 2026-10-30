#include "kftrack/simulation.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

namespace kf {

namespace {
constexpr double kPi = 3.14159265358979323846;

void buildTrajectory(Vehicle& v, const Scene& scene, double fine = 0.02) {
  const size_t n = v.path.size();
  std::vector<double> segLen(n - 1), cum(n, 0.0);
  for (size_t i = 0; i + 1 < n; ++i) {
    segLen[i] = std::hypot(v.path[i + 1].x - v.path[i].x, v.path[i + 1].y - v.path[i].y);
    cum[i + 1] = cum[i] + segLen[i];
  }
  std::vector<double> limit(n);
  for (size_t j = 0; j < n; ++j) {
    const Waypoint& w = v.path[j];
    limit[j] = w.dwell > 0 ? 0.0 : (w.speedLimit >= 0 ? w.speedLimit : v.speed);
  }

  double s = 0.0, spd = v.speed, t = v.start, wait = 0.0;
  std::set<size_t> done;
  while (s < cum.back() && t <= scene.tEnd + 1.0) {
    size_t i = static_cast<size_t>(std::upper_bound(cum.begin(), cum.end(), s) - cum.begin()) - 1;
    i = std::min(i, n - 2);
    const double ux = (v.path[i + 1].x - v.path[i].x) / segLen[i];
    const double uy = (v.path[i + 1].y - v.path[i].y) / segLen[i];
    v.ts.push_back(t);
    v.positions.push_back(Vec2{v.path[i].x + ux * (s - cum[i]), v.path[i].y + uy * (s - cum[i])});
    v.velocities.push_back(Vec2{ux * spd, uy * spd});
    t += fine;

    if (wait > 0) {
      wait -= fine;
      continue;
    }
    for (size_t j = 1; j < n; ++j) {
      if (v.path[j].dwell > 0 && !done.count(j) && s >= cum[j] - 0.05) {
        done.insert(j);
        wait = v.path[j].dwell;
        spd = 0.0;
      }
    }
    // Brake early enough to meet every upcoming speed limit / stop line
    double vmax = v.speed;
    for (size_t j = 1; j < n; ++j) {
      const double d = cum[j] - s;
      if (d > 0 && !done.count(j)) vmax = std::min(vmax, std::sqrt(limit[j] * limit[j] + 2 * scene.brake * d));
    }
    spd = wait <= 0 ? std::min(spd + scene.accel * fine, vmax) : 0.0;
    s += spd * fine;
  }
}
}  // namespace

Box Scene::droneFov(double t) const {
  // The drone drifts slowly, so the camera footprint moves
  const double cx = 200 + 45 * std::sin(2 * kPi * t / tEnd);
  const double cy = 150 + 15 * std::sin(4 * kPi * t / tEnd);
  return Box{cx - fovWidth / 2, cx + fovWidth / 2, cy - fovHeight / 2, cy + fovHeight / 2};
}

bool Scene::vehicleState(int vid, double t, Vec2& pos, Vec2& vel) const {
  const Vehicle& v = vehicles[vid];
  if (v.ts.empty() || t < v.ts.front() || t > v.ts.back()) return false;
  size_t k = static_cast<size_t>(std::lower_bound(v.ts.begin(), v.ts.end(), t) - v.ts.begin());
  k = std::min(k, v.ts.size() - 1);
  pos = v.positions[k];
  vel = v.velocities[k];
  return true;
}

void Scene::finalize() {
  for (Vehicle& v : vehicles) {
    v.ts.clear();
    v.positions.clear();
    v.velocities.clear();
    buildTrajectory(v, *this);
  }
}

Scene makeCrossroadsScene() {
  Scene s;
  s.vehicles = {
      {"Car A", 12, 0, {{0, 145}, {400, 145}}, {}, {}, {}},
      {"Car B", 10, 0, {{400, 155}, {0, 155}}, {}, {}, {}},
      {"Car C (turns)", 9, 3,
       {{0, 145}, {180, 145, 0, 5}, {190, 148, 0, 5}, {195, 157, 0, 5}, {195, 300}}, {}, {}, {}},
      {"Truck D (stops)", 7, 10, {{205, 0}, {205, 110, 6}, {205, 300}}, {}, {}, {}},
      {"Car E (overtakes)", 16, 5, {{0, 141}, {400, 141}}, {}, {}, {}},
  };
  s.finalize();
  return s;
}

std::vector<Frame> simulateDrone(const Scene& scene, const SensorConfig& cfg) {
  std::mt19937 rng(cfg.seed);
  std::normal_distribution<double> noise(0.0, cfg.measSigma);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::poisson_distribution<int> clutterCount(cfg.clutterRate);

  std::vector<Frame> frames;
  frames.reserve(scene.numFrames());
  for (int k = 0; k < scene.numFrames(); ++k) {
    Frame f;
    f.index = k;
    f.t = k * scene.dt;
    f.fov = scene.droneFov(f.t);

    for (int vid = 0; vid < static_cast<int>(scene.vehicles.size()); ++vid) {
      TruthState ts;
      if (!scene.vehicleState(vid, f.t, ts.pos, ts.vel)) continue;
      ts.vid = vid;
      ts.inView = f.fov.contains(ts.pos);
      ts.hidden = scene.trees.contains(ts.pos);
      f.truth.push_back(ts);
      if (ts.inView && !ts.hidden && unit(rng) < cfg.pDetect) {
        const double nx = noise(rng);
        const double ny = noise(rng);
        f.detections.push_back(Vec2{ts.pos[0] + nx, ts.pos[1] + ny});
        f.isClutter.push_back(false);
      }
    }
    const int nClutter = cfg.clutterRate > 0 ? clutterCount(rng) : 0;
    for (int c = 0; c < nClutter; ++c) {
      const double x = f.fov.xmin + unit(rng) * (f.fov.xmax - f.fov.xmin);
      const double y = f.fov.ymin + unit(rng) * (f.fov.ymax - f.fov.ymin);
      f.detections.push_back(Vec2{x, y});
      f.isClutter.push_back(true);
    }
    frames.push_back(std::move(f));
  }
  return frames;
}

}  // namespace kf
