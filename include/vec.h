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

#endif
