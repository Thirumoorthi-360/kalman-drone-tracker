#include "kftrack/hungarian.hpp"

#include <limits>

namespace kf {

std::vector<int> solveAssignment(const std::vector<std::vector<double>>& cost) {
  const int rows = static_cast<int>(cost.size());
  if (rows == 0) return {};
  const int cols = static_cast<int>(cost[0].size());
  if (cols == 0) return std::vector<int>(rows, -1);

  // The classic potentials-based algorithm needs n <= m, so transpose if necessary.
  const bool transposed = rows > cols;
  const int n = transposed ? cols : rows;
  const int m = transposed ? rows : cols;
  auto a = [&](int i, int j) { return transposed ? cost[j][i] : cost[i][j]; };

  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, INF);
    std::vector<char> used(m + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = INF;
      int j1 = 0;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) continue;
        const double cur = a(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> rowToCol(n, -1);
  for (int j = 1; j <= m; ++j)
    if (p[j] != 0) rowToCol[p[j] - 1] = j - 1;
  if (!transposed) return rowToCol;

  std::vector<int> result(rows, -1);
  for (int i = 0; i < n; ++i)
    if (rowToCol[i] >= 0) result[rowToCol[i]] = i;
  return result;
}

}  // namespace kf
