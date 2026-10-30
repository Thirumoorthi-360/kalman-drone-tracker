// Unit tests (no framework needed). Run with `ctest` or directly: ./kf_tests

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "kftrack/hungarian.hpp"
#include "kftrack/kalman_filter.hpp"
#include "kftrack/metrics.hpp"
#include "kftrack/simulation.hpp"
#include "kftrack/tracker.hpp"

using namespace kf;

namespace {
int gFailures = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::cerr << "  FAILED: " #cond "  (" << __FILE__ << ":" << __LINE__ << ")\n"; \
      ++gFailures;                                                                    \
    }                                                                                 \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

void run(const std::string& name, const std::function<void()>& fn) {
  const int before = gFailures;
  fn();
  std::cout << (gFailures == before ? "[pass] " : "[FAIL] ") << name << "\n";
}

// ---------------------------------------------------------------------------

void testMatrixBasics() {
  const Matrix<2, 3> A{1, 2, 3,
                       4, 5, 6};
  const Matrix<3, 2> B = A.transpose();
  const Matrix<2, 2> C = A * B;  // [[14, 32], [32, 77]]
  CHECK_NEAR(C(0, 0), 14, 1e-12);
  CHECK_NEAR(C(0, 1), 32, 1e-12);
  CHECK_NEAR(C(1, 1), 77, 1e-12);
  CHECK_NEAR((Matrix<3, 3>::Identity() * 2.0).trace(), 6, 1e-12);
}

void testInverse() {
  const Matrix<4, 4> A{4, 1, 0, 2,
                       1, 3, 1, 0,
                       0, 1, 5, 1,
                       2, 0, 1, 6};
  const Matrix<4, 4> I = A * inverse(A);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) CHECK_NEAR(I(r, c), r == c ? 1.0 : 0.0, 1e-12);

  // Needs pivoting: zero on the diagonal
  const Matrix<2, 2> Z{0, 1,
                       1, 0};
  const Matrix<2, 2> Zi = inverse(Z);
  CHECK_NEAR(Zi(0, 1), 1, 1e-12);
  CHECK_NEAR(Zi(1, 0), 1, 1e-12);

  bool threw = false;
  try {
    inverse(Matrix<2, 2>{1, 2, 2, 4});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

// Static scalar (F = H = 1, Q = 0): the covariance has a closed form,
//   P_n = 1 / (1/P0 + n/R)
// and the estimate converges to the true value.
void testScalarFilterMatchesTheory() {
  KalmanFilter<1, 1> f;
  f.F = Matrix<1, 1>{1};
  f.H = Matrix<1, 1>{1};
  f.Q = Matrix<1, 1>{0};
  f.R = Matrix<1, 1>{4.0};
  f.x = Vector<1>{0.0};
  f.P = Matrix<1, 1>{100.0};

  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, 2.0);
  double prevP = f.P(0, 0);
  for (int n = 1; n <= 400; ++n) {
    f.predict();
    f.update(Vector<1>{5.0 + noise(rng)});
    CHECK_NEAR(f.P(0, 0), 1.0 / (1.0 / 100.0 + n / 4.0), 1e-9);
    CHECK(f.P(0, 0) < prevP);  // every measurement reduces uncertainty
    prevP = f.P(0, 0);
  }
  CHECK_NEAR(f.x[0], 5.0, 0.3);
}

// A constant-velocity filter should learn velocity from position-only measurements.
void testConstantVelocityRecoversSpeed() {
  const double dt = 1.0;
  KalmanFilter<2, 1> f;
  f.F = Matrix<2, 2>{1, dt, 0, 1};
  f.H = Matrix<1, 2>{1, 0};
  f.Q = Matrix<2, 2>::Diagonal({1e-4, 1e-4});
  f.R = Matrix<1, 1>{1.0};
  f.x = Vector<2>{0, 0};
  f.P = Matrix<2, 2>::Diagonal({100, 100});

  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 1.0);
  for (int k = 1; k <= 60; ++k) {
    f.predict();
    f.update(Vector<1>{3.0 * k + noise(rng)});
  }
  CHECK_NEAR(f.x[1], 3.0, 0.1);
}

void testHungarian() {
  // Optimal assignment: row0->col1, row1->col0, row2->col2  (total 1+2+2 = 5)
  const std::vector<std::vector<double>> c{{4, 1, 3},
                                           {2, 0, 5},
                                           {3, 2, 2}};
  const auto a = solveAssignment(c);
  CHECK(a.size() == 3);
  CHECK(a[0] == 1 && a[1] == 0 && a[2] == 2);

  // More columns than rows
  const auto w = solveAssignment({{9, 1, 9, 9}, {9, 9, 9, 2}});
  CHECK(w[0] == 1 && w[1] == 3);

  // More rows than columns: one row stays unassigned
  const auto t = solveAssignment({{5, 9}, {1, 9}, {9, 1}});
  CHECK(t[0] == -1 && t[1] == 0 && t[2] == 1);

  CHECK(solveAssignment({}).empty());
}

// Two vehicles crossing paths with clean detections must keep their own IDs.
void testTrackerKeepsIdsThroughCrossing() {
  TrackerConfig cfg;
  MultiTargetTracker trk(cfg);
  const Box fov{-50, 250, -50, 250};
  std::vector<std::vector<TrackSnapshot>> hist;
  for (int k = 0; k < 40; ++k) {
    const double t = k * cfg.dt;
    const std::vector<Vec2> dets{Vec2{10 + 8 * t, 100}, Vec2{100, 10 + 8 * t}};
    hist.push_back(trk.step(dets, fov));
  }
  const auto& last = hist.back();
  int confirmed = 0, labelA = -1, labelB = -1;
  for (const auto& s : last) {
    if (!s.confirmed) continue;
    ++confirmed;
    if (std::abs(s.x[1] - 100) < 5 && s.x[0] > 150) labelA = s.label;  // eastbound
    if (std::abs(s.x[0] - 100) < 5 && s.x[1] > 150) labelB = s.label;  // northbound
  }
  CHECK(confirmed == 2);
  CHECK(labelA == 1 || labelA == 2);
  CHECK(labelB == 1 || labelB == 2);
  CHECK(labelA != labelB);
  for (const auto& s : last)
    if (s.confirmed) CHECK_NEAR(std::hypot(s.x[2], s.x[3]), 8.0, 0.5);  // speed recovered
}

// A track with no detections coasts: position keeps moving, uncertainty grows.
void testCoastingGrowsUncertainty() {
  TrackerConfig cfg;
  MultiTargetTracker trk(cfg);
  const Box fov{-100, 500, -100, 500};
  for (int k = 0; k < 10; ++k) trk.step({Vec2{10.0 * k * cfg.dt, 0}}, fov);
  double prevVar = trk.tracks().front().kf.P(0, 0);
  double prevX = trk.tracks().front().kf.x[0];
  for (int k = 0; k < 5; ++k) {
    trk.step({}, fov);
    const Track& t = trk.tracks().front();
    CHECK(!t.updated);
    CHECK(t.kf.P(0, 0) > prevVar);
    CHECK(t.kf.x[0] > prevX);
    prevVar = t.kf.P(0, 0);
    prevX = t.kf.x[0];
  }
}

// End-to-end regression on the crossroads scene, averaged over several seeds.
void testCrossroadsScenario() {
  const Scene scene = makeCrossroadsScene();
  double coverage = 0, rmseGain = 0;
  double precision = 0;
  int switches = 0, falseTracks = 0;
  const int nSeeds = 20;
  for (unsigned seed = 1; seed <= static_cast<unsigned>(nSeeds); ++seed) {
    SensorConfig sc;
    sc.seed = seed;
    const auto frames = simulateDrone(scene, sc);
    MultiTargetTracker trk;
    TrackHistory hist;
    for (const auto& f : frames) hist.push_back(trk.step(f.detections, f.fov));
    const Scorecard s = evaluate(scene, frames, hist);
    coverage += s.meanCoverage;
    precision += s.trackPrecision;
    switches += s.idSwitches;
    falseTracks += s.falseTracks;
    rmseGain += s.rawRmse - s.rmseVisible;
  }
  std::cout << "       avg coverage " << coverage / nSeeds << "%, precision " << precision / nSeeds
            << "%, ID switches/run " << double(switches) / nSeeds
            << ", false tracks/run " << double(falseTracks) / nSeeds << ", RMSE gain " << rmseGain / nSeeds
            << " m\n";
  CHECK(coverage / nSeeds > 90.0);
  CHECK(precision / nSeeds > 92.0);
  CHECK(double(switches) / nSeeds < 3.5);
  CHECK(double(falseTracks) / nSeeds < 1.0);
  CHECK(rmseGain / nSeeds > 0.0);  // tracker beats the raw detector
}

}  // namespace

int main() {
  run("matrix basics", testMatrixBasics);
  run("matrix inverse", testInverse);
  run("scalar filter matches closed-form covariance", testScalarFilterMatchesTheory);
  run("constant-velocity filter recovers speed", testConstantVelocityRecoversSpeed);
  run("hungarian assignment", testHungarian);
  run("tracker keeps IDs through a crossing", testTrackerKeepsIdsThroughCrossing);
  run("coasting grows uncertainty", testCoastingGrowsUncertainty);
  run("crossroads scenario regression", testCrossroadsScenario);
  std::cout << (gFailures ? "\nSOME TESTS FAILED\n" : "\nAll tests passed\n");
  return gFailures ? 1 : 0;
}
