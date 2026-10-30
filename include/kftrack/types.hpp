#pragma once

#include "kftrack/matrix.hpp"

namespace kf {

using Vec2 = Vector<2>;

// Axis-aligned rectangle on the ground (metres).
struct Box {
  double xmin = 0, xmax = 0, ymin = 0, ymax = 0;

  bool contains(double x, double y, double margin = 0.0) const {
    return x >= xmin - margin && x <= xmax + margin && y >= ymin - margin && y <= ymax + margin;
  }
  bool contains(const Vec2& p, double margin = 0.0) const { return contains(p[0], p[1], margin); }
};

}  // namespace kf
