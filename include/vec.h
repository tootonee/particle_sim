#pragma once

#ifndef VEC_MATH_H
#define VEC_MATH_H

#include <cmath>

static constexpr double ERR = 0.0001F;

__host__ __device__ constexpr inline double distance(double3 const &lhs,
                                                     double3 const &rhs) {
  return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) +
              (lhs.y - rhs.y) * (lhs.y - rhs.y) +
              (lhs.z - rhs.z) * (lhs.z - rhs.z));
}

__host__ __device__ constexpr inline double distance(double4 lhs, double4 rhs) {
  return sqrt(
      (lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y) +
      (lhs.z - rhs.z) * (lhs.z - rhs.z) + (lhs.w - rhs.w) * (lhs.w - rhs.w));
}

__host__ __device__ constexpr inline bool operator==(double3 const &lhs,
                                                     double3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

__host__ __device__ constexpr inline double dot(double3 const &lhs,
                                                double3 const &rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ constexpr inline double3 normalize(double3 const &val) {
  double len = distance(val, {0, 0, 0});
  return {
      .x = val.x / len,
      .y = val.y / len,
      .z = val.z / len,
  };
}

__host__ __device__ constexpr inline double4 normalize(double4 const &val) {
  double len = distance(val, {0, 0, 0, 0});
  return {
      .x = val.x / len,
      .y = val.y / len,
      .z = val.z / len,
      .w = val.w / len,
  };
}

////////////////////////////////////////////////////////////////////////////////
/// Quaternion operations
////////////////////////////////////////////////////////////////////////////////
__host__ __device__ constexpr inline double4 operator+(double4 const &lhs,
                                                       double4 const &rhs) {
  // Sorry, gotta do it like this
  const double &a1 = lhs.x;
  const double &b1 = lhs.y;
  const double &c1 = lhs.z;
  const double &d1 = lhs.w;

  const double &a2 = rhs.x;
  const double &b2 = rhs.y;
  const double &c2 = rhs.z;
  const double &d2 = rhs.w;

  return {
      .x = a1 + a2,
      .y = b1 + b2,
      .z = c1 + c2,
      .w = d1 + d2,
  };
}

__host__ __device__ constexpr inline double4 operator*(double4 const &lhs,
                                                       double4 const &rhs) {
  // Sorry, gotta do it like this
  const double &a1 = lhs.x;
  const double &b1 = lhs.y;
  const double &c1 = lhs.z;
  const double &d1 = lhs.w;

  const double &a2 = rhs.x;
  const double &b2 = rhs.y;
  const double &c2 = rhs.z;
  const double &d2 = rhs.w;

  return {
      .x = (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2),
      .y = (a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2),
      .z = (a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2),
      .w = (a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2),
  };
}

#endif
