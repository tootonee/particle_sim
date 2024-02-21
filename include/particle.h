#pragma once

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "patch.h"
#include <cstddef>

static const size_t DEFAULT_CAPACITY = 16L;

struct Particle {
    double radius{};
    double3 pos{0, 0, 0};
    double4 orient{0, 0, 0, 0};
    size_t patch_count{};
    size_t capacity{};
    Patch *patches{};

    __host__ __device__ bool intersects(Particle const &rhs);
};

#endif
