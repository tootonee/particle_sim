#pragma once

#ifndef VEC_MATRH_HPP
#define VEC_MATRH_HPP

static constexpr double ERR = 0.0001F;

__host__ __device__ inline double distance(double3 lhs, double3 rhs) {
    return sqrt((lhs.x - rhs.x) * (lhs.x - rhs.x) +
                (lhs.y - rhs.y) * (lhs.y - rhs.y) +
                (lhs.z - rhs.z) * (lhs.z - rhs.z));
}

__host__ __device__ inline double distance(double4 lhs, double4 rhs) {
    return sqrt(
        (lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y) +
        (lhs.z - rhs.z) * (lhs.z - rhs.z) + (lhs.w - rhs.w) * (lhs.w - rhs.w));
}

#endif
