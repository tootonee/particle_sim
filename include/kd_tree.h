#pragma once

#ifndef KD_TREE_HPP
#define KD_TREE_HPP

#include "particle.h"
#include "patch.h"

#include <cstddef>

struct particle_search_tree {
    Particle *particles;
    size_t particle_count;

    // __host__ __device__
};

#endif
