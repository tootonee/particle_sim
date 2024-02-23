#pragma once


#ifndef PARTICLE_BOX_HPP
#define PARTICLE_BOX_HPP

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "particle.h"

struct __align__(64) particle_box_t {
    particle_t *particles{};
    double3 dimensions{};
    size_t particle_count{};
    size_t capacity{};
};

void particle_box_init_host(particle_box_t &p,
    size_t capacity = DEFAULT_CAPACITY);

__device__ void particle_box_init_device(particle_box_t &p,
    size_t capacity = DEFAULT_CAPACITY);

void particle_box_realloc_host(particle_box_t &p,
    size_t capacity = DEFAULT_CAPACITY);

__device__ void particle_box_realloc_device(particle_box_t &p,
    size_t capacity = DEFAULT_CAPACITY);

void particle_box_add_particle_host(particle_box_t &box, double radius);
__device__ void particle_box_add_particle_device(particle_box_t &p, double radius);

inline void particle_box_free_particles_host(particle_box_t &p) {
    for (size_t idx = 0; idx < p.particle_count; idx++) {
        delete[] p.particles[idx].patches;
    }
    delete[] p.particles;
    p.particle_count = 0;
    p.capacity = 0;
}

__device__ inline void particle_box_free_particles_device(particle_box_t &p) {
    for (size_t idx = 0; idx < p.particle_count; idx++) {
        cudaFree(p.particles[idx].patches);
    }
    cudaFree(p.particles);
    p.particle_count = 0;
    p.capacity = 0;
}

#endif