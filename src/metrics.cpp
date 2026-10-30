#include "kftrack/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>

#include "kftrack/hungarian.hpp"

namespace kf {

namespace {

struct Pair {
  int vid;
  int label;
  bool hidden;
  double dist;
};

double dist(const Vec2& a, double x, double y) { return std::hypot(a[0] - x, a[1] - y); }

// Match confirmed tracks to visible vehicles in one frame (Hungarian on distance).
// inViewOnly=false also counts vehicles that just drove out of the camera view (used for precision).
std::vector<Pair> matchFrame(const Frame& f, const std::vector<TrackSnapshot>& snap, double maxDist,
                             bool inViewOnly = true) {
  std::vector<const TruthState*> tv;
  for (const TruthState& t : f.truth)
    if (t.inView || !inViewOnly) tv.push_back(&t);
  std::vector<const TrackSnapshot*> ct;
  for (const TrackSnapshot& s : snap)
    if (s.confirmed) ct.push_back(&s);
  if (tv.empty() || ct.empty()) return {};

  std::vector<std::vector<double>> D(tv.size(), std::vector<double>(ct.size()));
  for (size_t i = 0; i < tv.size(); ++i)
    for (size_t j = 0; j < ct.size(); ++j) D[i][j] = dist(tv[i]->pos, ct[j]->x[0], ct[j]->x[1]);

  std::vector<Pair> out;
  const std::vector<int> assign = solveAssignment(D);
  for (size_t i = 0; i < tv.size(); ++i) {
    const int j = assign[i];
    if (j >= 0 && D[i][j] < maxDist) out.push_back({tv[i]->vid, ct[j]->label, tv[i]->hidden, D[i][j]});
  }
  return out;
}

double rms(const std::vector<double>& e) {
  if (e.empty()) return std::numeric_limits<double>::quiet_NaN();
  double s = 0;
  for (double v : e) s += v * v;
  return std::sqrt(s / static_cast<double>(e.size()));
}

char labelChar(int label) {
  const char* chars = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  return label >= 1 && label <= 35 ? chars[label - 1] : '#';
}

}  // namespace

Scorecard evaluate(const Scene& scene, const std::vector<Frame>& frames, const TrackHistory& history,
                   double maxMatchDist) {
  std::vector<std::vector<Pair>> matches;
  for (size_t k = 0; k < frames.size(); ++k) matches.push_back(matchFrame(frames[k], history[k], maxMatchDist));

  Scorecard sc;
  std::vector<double> errVisible, errHidden;
  for (int vid = 0; vid < static_cast<int>(scene.vehicles.size()); ++vid) {
    int present = 0, covered = 0, switches = 0, lastLabel = -1;
    std::vector<double> errs;
    for (size_t k = 0; k < frames.size(); ++k) {
      for (const TruthState& t : frames[k].truth)
        if (t.vid == vid && t.inView) ++present;
      for (const Pair& p : matches[k]) {
        if (p.vid != vid) continue;
        ++covered;
        errs.push_back(p.dist);
        (p.hidden ? errHidden : errVisible).push_back(p.dist);
        if (lastLabel != -1 && p.label != lastLabel) ++switches;
        lastLabel = p.label;
      }
    }
    if (present == 0) continue;
    sc.vehicles.push_back({scene.vehicles[vid].name, 100.0 * covered / present, rms(errs), switches});
    sc.idSwitches += switches;
  }

  double covSum = 0;
  for (const VehicleScore& v : sc.vehicles) covSum += v.coverage;
  sc.meanCoverage = sc.vehicles.empty() ? 0.0 : covSum / static_cast<double>(sc.vehicles.size());
  sc.rmseVisible = rms(errVisible);
  sc.rmseHidden = rms(errHidden);

  std::set<int> realLabels, allLabels;
  for (const auto& m : matches)
    for (const Pair& p : m) realLabels.insert(p.label);
  for (const auto& snap : history)
    for (const TrackSnapshot& s : snap)
      if (s.confirmed) allLabels.insert(s.label);
  for (int l : allLabels)
    if (!realLabels.count(l)) ++sc.falseTracks;

  int shown = 0, onVehicle = 0;
  for (size_t k = 0; k < frames.size(); ++k) {
    for (const TrackSnapshot& s : history[k]) shown += s.confirmed ? 1 : 0;
    onVehicle += static_cast<int>(matchFrame(frames[k], history[k], maxMatchDist, false).size());
  }
  sc.trackPrecision = shown ? 100.0 * onVehicle / shown : 100.0;

  std::vector<double> raw;
  for (const Frame& f : frames) {
    for (size_t j = 0; j < f.detections.size(); ++j) {
      if (f.isClutter[j]) continue;
      double best = std::numeric_limits<double>::infinity();
      for (const TruthState& t : f.truth) best = std::min(best, dist(t.pos, f.detections[j][0], f.detections[j][1]));
      raw.push_back(best);
    }
  }
  sc.rawRmse = rms(raw);
  return sc;
}

void printScorecard(std::ostream& os, const Scorecard& sc) {
  os << std::fixed;
  os << std::left << std::setw(20) << "Vehicle" << std::right << std::setw(10) << "Coverage" << std::setw(11)
     << "RMSE (m)" << std::setw(13) << "ID switches" << "\n";
  os << std::string(54, '-') << "\n";
  for (const VehicleScore& v : sc.vehicles) {
    os << std::left << std::setw(20) << v.name << std::right << std::setw(9) << std::setprecision(0) << v.coverage
       << "%" << std::setw(11) << std::setprecision(2) << v.rmse << std::setw(13) << v.idSwitches << "\n";
  }
  os << std::string(54, '-') << "\n" << std::setprecision(2);
  os << "Position error  raw detector: " << sc.rawRmse << " m  |  tracker (visible): " << sc.rmseVisible
     << " m  |  tracker coasting under trees: " << sc.rmseHidden << " m\n";
  os << std::setprecision(1) << "Track precision: " << sc.trackPrecision
     << "% of displayed tracks are on a real vehicle  |  False tracks: " << sc.falseTracks
     << "  |  Total ID switches: " << sc.idSwitches << "\n";
}

void printIdTimeline(std::ostream& os, const Scene& scene, const std::vector<Frame>& frames,
                     const TrackHistory& history, double maxMatchDist) {
  for (int vid = 0; vid < static_cast<int>(scene.vehicles.size()); ++vid) {
    std::string line;
    for (size_t k = 0; k < frames.size(); ++k) {
      const TruthState* tv = nullptr;
      for (const TruthState& t : frames[k].truth)
        if (t.vid == vid) tv = &t;
      if (!tv) {
        line += ' ';
        continue;
      }
      if (!tv->inView) {
        line += '.';
        continue;
      }
      char c = tv->hidden ? '*' : '-';
      for (const Pair& p : matchFrame(frames[k], history[k], maxMatchDist))
        if (p.vid == vid) c = labelChar(p.label);
      line += c;
    }
    while (!line.empty() && line.back() == ' ') line.pop_back();
    os << std::left << std::setw(20) << scene.vehicles[vid].name << line << "\n";
  }
}

}  // namespace kf
