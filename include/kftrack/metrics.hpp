#pragma once
// Scoring the tracker against ground truth.

#include <ostream>
#include <string>
#include <vector>

#include "kftrack/simulation.hpp"
#include "kftrack/tracker.hpp"

namespace kf {

struct VehicleScore {
  std::string name;
  double coverage = 0.0;  // % of in-view frames with a confirmed track on it
  double rmse = 0.0;      // tracker position error (m)
  int idSwitches = 0;     // times its track label changed
};

struct Scorecard {
  std::vector<VehicleScore> vehicles;
  double meanCoverage = 0.0;
  double rmseVisible = 0.0;  // tracker error while the vehicle is visible
  double rmseHidden = 0.0;   // tracker error while coasting under the trees (NaN if never)
  double rawRmse = 0.0;      // raw detector error
  int idSwitches = 0;
  int falseTracks = 0;       // confirmed tracks never matched to a real vehicle
  double trackPrecision = 0.0;  // % of displayed (confirmed) track-frames that sit on a real vehicle
};

using TrackHistory = std::vector<std::vector<TrackSnapshot>>;

Scorecard evaluate(const Scene& scene, const std::vector<Frame>& frames, const TrackHistory& history,
                   double maxMatchDist = 10.0);

void printScorecard(std::ostream& os, const Scorecard& sc);

// One line per vehicle showing which track ID followed it in each frame.
//   ' ' not on map   '.' outside camera view   '-' visible but untracked
//   '*' hidden under trees and untracked   1-9/A-Z track label
void printIdTimeline(std::ostream& os, const Scene& scene, const std::vector<Frame>& frames,
                     const TrackHistory& history, double maxMatchDist = 10.0);

}  // namespace kf
