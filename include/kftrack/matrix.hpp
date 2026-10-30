#pragma once
// Small fixed-size matrix type with no external dependencies.
//
// Sizes are template parameters, so dimension mistakes (for example multiplying
// a 4x2 matrix by a 4x4 matrix) are caught at compile time instead of at runtime.

#include <array>
#include <cmath>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace kf {

template <int R, int C>
class Matrix {
  static_assert(R > 0 && C > 0, "Matrix dimensions must be positive");

 public:
  static constexpr int kRows = R;
  static constexpr int kCols = C;

  Matrix() { data_.fill(0.0); }

  // Row-major values, e.g.  Matrix<2, 2>{1, 2,
  //                                      3, 4};
  Matrix(std::initializer_list<double> values) {
    if (static_cast<int>(values.size()) != R * C) {
      throw std::invalid_argument("Matrix initializer has the wrong number of elements");
    }
    int i = 0;
    for (double v : values) data_[i++] = v;
  }

  static Matrix Zero() { return Matrix(); }

  static Matrix Identity() {
    static_assert(R == C, "Identity() needs a square matrix");
    Matrix m;
    for (int i = 0; i < R; ++i) m(i, i) = 1.0;
    return m;
  }

  static Matrix Diagonal(std::initializer_list<double> diag) {
    static_assert(R == C, "Diagonal() needs a square matrix");
    if (static_cast<int>(diag.size()) != R) {
      throw std::invalid_argument("Diagonal() needs exactly R values");
    }
    Matrix m;
    int i = 0;
    for (double v : diag) {
      m(i, i) = v;
      ++i;
    }
    return m;
  }

  double& operator()(int r, int c) { return data_[r * C + c]; }
  double operator()(int r, int c) const { return data_[r * C + c]; }

  // Flat access, handy for vectors: v[0], v[1], ...
  double& operator[](int i) { return data_[i]; }
  double operator[](int i) const { return data_[i]; }

  Matrix<C, R> transpose() const {
    Matrix<C, R> t;
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) t(c, r) = (*this)(r, c);
    return t;
  }

  double trace() const {
    static_assert(R == C, "trace() needs a square matrix");
    double s = 0.0;
    for (int i = 0; i < R; ++i) s += (*this)(i, i);
    return s;
  }

  Matrix& operator+=(const Matrix& o) {
    for (int i = 0; i < R * C; ++i) data_[i] += o.data_[i];
    return *this;
  }
  Matrix& operator-=(const Matrix& o) {
    for (int i = 0; i < R * C; ++i) data_[i] -= o.data_[i];
    return *this;
  }
  Matrix& operator*=(double s) {
    for (double& v : data_) v *= s;
    return *this;
  }

  friend Matrix operator+(Matrix a, const Matrix& b) { return a += b; }
  friend Matrix operator-(Matrix a, const Matrix& b) { return a -= b; }
  friend Matrix operator*(Matrix a, double s) { return a *= s; }
  friend Matrix operator*(double s, Matrix a) { return a *= s; }

  template <int K>
  Matrix<R, K> operator*(const Matrix<C, K>& o) const {
    Matrix<R, K> out;
    for (int r = 0; r < R; ++r)
      for (int k = 0; k < K; ++k) {
        double s = 0.0;
        for (int c = 0; c < C; ++c) s += (*this)(r, c) * o(c, k);
        out(r, k) = s;
      }
    return out;
  }

  friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (int r = 0; r < R; ++r) {
      os << (r == 0 ? "[" : " ");
      for (int c = 0; c < C; ++c) os << (c ? ", " : "") << m(r, c);
      os << (r == R - 1 ? "]" : "\n");
    }
    return os;
  }

 private:
  std::array<double, R * C> data_;
};

template <int N>
using Vector = Matrix<N, 1>;

// Inverse by Gauss-Jordan elimination with partial pivoting.
// Fine for the small (2x2, 4x4) matrices a Kalman filter needs.
template <int N>
Matrix<N, N> inverse(Matrix<N, N> a) {
  Matrix<N, N> inv = Matrix<N, N>::Identity();
  for (int col = 0; col < N; ++col) {
    int pivot = col;
    for (int r = col + 1; r < N; ++r)
      if (std::abs(a(r, col)) > std::abs(a(pivot, col))) pivot = r;
    if (std::abs(a(pivot, col)) < 1e-12) throw std::runtime_error("inverse(): matrix is singular");
    if (pivot != col) {
      for (int c = 0; c < N; ++c) {
        std::swap(a(col, c), a(pivot, c));
        std::swap(inv(col, c), inv(pivot, c));
      }
    }
    const double d = a(col, col);
    for (int c = 0; c < N; ++c) {
      a(col, c) /= d;
      inv(col, c) /= d;
    }
    for (int r = 0; r < N; ++r) {
      if (r == col) continue;
      const double f = a(r, col);
      if (f == 0.0) continue;
      for (int c = 0; c < N; ++c) {
        a(r, c) -= f * a(col, c);
        inv(r, c) -= f * inv(col, c);
      }
    }
  }
  return inv;
}

}  // namespace kf
