#pragma once
// Optimal assignment (Hungarian / Kuhn-Munkres algorithm), O(n^3).

#include <vector>

namespace kf {

// Given a rows x cols cost matrix, returns for each row the column it is
// assigned to, minimising the total cost. Works for rectangular matrices:
// when rows > cols, some rows get -1 (unassigned).
std::vector<int> solveAssignment(const std::vector<std::vector<double>>& cost);

}  // namespace kf
