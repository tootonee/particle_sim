#pragma once

#ifndef PARTICLE_H
#define PARTICLE_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <random>

#include "patch.h"

static const size_t DEFAULT_CAPACITY = 16L;

struct __align__(32) particle_t {
  double radius{};
  double3 pos{0, 0, 0};
  double4 orient{0, 0, 0, 0};
  patch_t *patches{};
  size_t patch_count{DEFAULT_CAPACITY};
  size_t idx{};
};

__host__ __device__ bool particle_intersects(const particle_t &p1,
                                             const particle_t &p2);

__host__ __device__ bool particle_intersects(const particle_t &p1,
                                             double3 const pos, double const radius);

__host__ __device__ bool particle_intersects(double3 const pos1, double const radius1,
                                             double3 const pos2, double const radius2);

void particle_init_host(particle_t &p, size_t capacity = DEFAULT_CAPACITY);

__device__ void particle_init_device(particle_t &p,
                                     size_t capacity = DEFAULT_CAPACITY);

__host__ void random_particle_pos(particle_t &p, double3 dimensions);
using rng_gen = std::uniform_real_distribution<double>;
__host__ void random_particle_pos(particle_t &p, rng_gen &rng_x, rng_gen &rng_y,
                                  rng_gen &rng_z, std::mt19937 &re);

#endif
