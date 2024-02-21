#pragma once

#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include "patch.h"
#include "vector.h"

#include <cstddef>

static const size_t DEFAULT_CAPACITY = 16L;

struct Particle {
    double radius;
    vec_3d pos;
    vec_4d orient;
    size_t patch_count;
    size_t capacity;
    Patch *patches;

    __host__ __device__ bool intersects(Particle const &rhs);
};

#endif
