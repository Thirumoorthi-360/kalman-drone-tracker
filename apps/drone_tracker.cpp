// Drone surveillance demo: simulate a crossroads, track every vehicle with
// Kalman filters, print a scorecard and write CSV files for plotting.
//
//   ./drone_tracker                         default run, writes ./output/*.csv
//   ./drone_tracker --clutter 8 --seed 3    harder conditions
//   ./drone_tracker --help

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kftrack/metrics.hpp"
#include "kftrack/simulation.hpp"
#include "kftrack/tracker.hpp"

using namespace kf;

namespace {

struct Options {
  SensorConfig sensor;
  TrackerConfig tracker;
  std::string outDir = "output";
  bool writeCsv = true;
  bool timeline = true;
};

void printHelp() {
  std::cout << "Usage: drone_tracker [options]\n\n"
               "Sensor (simulated drone camera):\n"
               "  --seed N            random seed (default 1)\n"
               "  --pdetect P         detection probability 0..1 (default 0.9)\n"
               "  --meas-sigma S      detector position noise, m (default 1.5)\n"
               "  --clutter C         mean false alarms per frame (default 1.5)\n\n"
               "Tracker:\n"
               "  --accel-sigma A     Kalman process noise, m/s^2 (default 1.5)\n"
               "  --confirm-hits N    hits before a track is confirmed (default 3)\n"
               "  --max-coast N       frames a track survives without detections (default 14)\n"
               "  --max-pos-sigma S   delete tracks whose position uncertainty exceeds S m (default 12)\n"
               "  --no-coast-by-hits  let young tracks coast as long as mature ones\n\n"
               "Output:\n"
               "  --out DIR           CSV output folder (default ./output)\n"
               "  --no-csv            do not write CSV files\n"
               "  --no-timeline       do not print the ID timeline\n";
}

Options parseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      printHelp();
      std::exit(0);
    } else if (a == "--seed") o.sensor.seed = static_cast<unsigned>(std::stoul(next()));
    else if (a == "--pdetect") o.sensor.pDetect = std::stod(next());
    else if (a == "--meas-sigma") o.sensor.measSigma = std::stod(next());
    else if (a == "--clutter") o.sensor.clutterRate = std::stod(next());
    else if (a == "--accel-sigma") o.tracker.accelSigma = std::stod(next());
    else if (a == "--confirm-hits") o.tracker.confirmHits = std::stoi(next());
    else if (a == "--max-coast") o.tracker.maxCoast = std::stoi(next());
    else if (a == "--max-pos-sigma") o.tracker.maxPosSigma = std::stod(next());
    else if (a == "--no-coast-by-hits") o.tracker.coastByHits = false;
    else if (a == "--out") o.outDir = next();
    else if (a == "--no-csv") o.writeCsv = false;
    else if (a == "--no-timeline") o.timeline = false;
    else {
      std::cerr << "Unknown option: " << a << "  (try --help)\n";
      std::exit(2);
    }
  }
  // The tracker is told the detector's true noise level
  o.tracker.measSigma = o.sensor.measSigma;
  return o;
}

void writeCsv(const std::string& dir, const Scene& scene, const std::vector<Frame>& frames,
              const TrackHistory& history) {
  std::filesystem::create_directories(dir);

  std::ofstream sc(dir + "/scene.csv");
  sc << "key,xmin,xmax,ymin,ymax\n";
  sc << "area," << scene.area.xmin << "," << scene.area.xmax << "," << scene.area.ymin << "," << scene.area.ymax
     << "\n";
  sc << "trees," << scene.trees.xmin << "," << scene.trees.xmax << "," << scene.trees.ymin << ","
     << scene.trees.ymax << "\n";

  std::ofstream tr(dir + "/truth.csv");
  tr << "frame,t,vid,name,x,y,vx,vy,in_view,hidden\n";
  std::ofstream de(dir + "/detections.csv");
  de << "frame,t,x,y,is_clutter\n";
  std::ofstream fv(dir + "/fov.csv");
  fv << "frame,t,xmin,xmax,ymin,ymax\n";
  std::ofstream tk(dir + "/tracks.csv");
  tk << "frame,t,uid,label,confirmed,updated,x,y,vx,vy,pxx,pxy,pyy\n";

  for (size_t k = 0; k < frames.size(); ++k) {
    const Frame& f = frames[k];
    fv << k << "," << f.t << "," << f.fov.xmin << "," << f.fov.xmax << "," << f.fov.ymin << "," << f.fov.ymax
       << "\n";
    for (const TruthState& t : f.truth)
      tr << k << "," << f.t << "," << t.vid << ",\"" << scene.vehicles[t.vid].name << "\"," << t.pos[0] << ","
         << t.pos[1] << "," << t.vel[0] << "," << t.vel[1] << "," << t.inView << "," << t.hidden << "\n";
    for (size_t j = 0; j < f.detections.size(); ++j)
      de << k << "," << f.t << "," << f.detections[j][0] << "," << f.detections[j][1] << "," << f.isClutter[j]
         << "\n";
    for (const TrackSnapshot& s : history[k])
      tk << k << "," << f.t << "," << s.uid << "," << s.label << "," << s.confirmed << "," << s.updated << ","
         << s.x[0] << "," << s.x[1] << "," << s.x[2] << "," << s.x[3] << "," << s.P(0, 0) << "," << s.P(0, 1)
         << "," << s.P(1, 1) << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parseArgs(argc, argv);

  const Scene scene = makeCrossroadsScene();
  const std::vector<Frame> frames = simulateDrone(scene, opt.sensor);

  MultiTargetTracker tracker(opt.tracker);
  TrackHistory history;
  history.reserve(frames.size());
  for (const Frame& f : frames) history.push_back(tracker.step(f.detections, f.fov));

  size_t nReal = 0, nClutter = 0;
  for (const Frame& f : frames)
    for (bool c : f.isClutter) (c ? nClutter : nReal)++;

  std::cout << "Drone surveillance: " << frames.size() << " frames, " << nReal << " real detections, " << nClutter
            << " false alarms\n\n";

  if (opt.timeline) {
    std::cout << "ID timeline  ('.' out of view, '-' untracked, '*' under trees & untracked, 1-9/A-Z = track ID)\n";
    printIdTimeline(std::cout, scene, frames, history);
    std::cout << "\n";
  }

  printScorecard(std::cout, evaluate(scene, frames, history));

  if (opt.writeCsv) {
    writeCsv(opt.outDir, scene, frames, history);
    std::cout << "\nCSV files written to " << opt.outDir << "/  (plot with: python scripts/plot_tracks.py "
              << opt.outDir << ")\n";
  }
  return 0;
}
