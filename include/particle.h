#pragma once

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "patch.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

static const size_t DEFAULT_CAPACITY = 16L;

struct __align__(128) particle_t {
    double radius{};
    double3 pos{0, 0, 0};
    double4 orient{0, 0, 0, 0};
    patch_t *patches{};
    unsigned int patch_count{DEFAULT_CAPACITY};
    bool valid{true};
};

__host__ __device__ bool particle_intersects(const particle_t &p1,
                                             const particle_t &p2);

void particle_init_host(particle_t &p, size_t capacity = DEFAULT_CAPACITY);

__device__ void particle_init_device(particle_t &p,
                                     size_t capacity = DEFAULT_CAPACITY);

__host__ void random_particle_pos(particle_t &p, double3 dimensions);

#endif
