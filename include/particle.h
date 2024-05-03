#pragma once

#ifndef PARTICLE_H
#define PARTICLE_H

#include <cstddef>
#include <random>

#include "patch.h"
#include "vec.h"

static const size_t DEFAULT_CAPACITY = 16L;
using rng_gen = std::uniform_real_distribution<double>;

struct __align__(32) particle_t {
  double radius{};
  double3 pos{0, 0, 0};
  double4 orient{1, 1, 0, 0};
  patch_t patches[DEFAULT_CAPACITY];
  size_t patch_count{0};
  size_t idx{};

  particle_t() = default;
  particle_t(particle_t const &part) {
    radius = part.radius;
    pos = part.pos;
    orient = part.orient;
    patch_count = part.patch_count;
    for (size_t i = 0; i < patch_count; i++) {
      patches[i] = part.patches[i];
    }
    idx = part.idx;
  }

  particle_t &operator=(particle_t const &part) {
    radius = part.radius;
    pos = part.pos;
    orient = part.orient;
    patch_count = part.patch_count;
    for (size_t i = 0; i < patch_count; i++) {
      patches[i] = part.patches[i];
    }
    idx = part.idx;

    return *this;
  }

  void random_particle_pos(double3 dimensions);
  void random_particle_pos(rng_gen & rng_x, rng_gen & rng_y, rng_gen & rng_z,
                           std::mt19937 & re);
  static double4 random_particle_orient(double const angle, int axis = 0);
  static __host__ __device__ double4 random_particle_orient_device(double angle, int axis = 0);
  void rotate(double4 rot);
  __host__ __device__ double interact(particle_t const &rhs,
                                      double const cosmax = 0.89,
                                      double const epsilon = -2) const;

  __host__ __device__ inline constexpr bool intersects(
      double3 const other_pos, double const other_radius) {
    double const diameter = radius + other_radius;
    return distance(pos, other_pos) <= diameter;
  }

  __host__ __device__ inline constexpr bool intersects(particle_t const &rhs)
      const {
    if (pos == rhs.pos) {
      return false;
    }
    double const diameter = radius + rhs.radius;
    return distance(pos, rhs.pos) < diameter;
  }

  void add_patch(patch_t const &p);
};

__host__ __device__ constexpr inline bool operator==(particle_t const &lhs,
                                                     particle_t const &rhs) {
  return lhs.pos.x == rhs.pos.x && lhs.pos.y == rhs.pos.y &&
         lhs.pos.z == rhs.pos.z && lhs.radius == rhs.radius;
}

#endif
