#pragma once

#ifndef PARTICLE_BOX_H
#define PARTICLE_BOX_H

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>

#include "particle.h"

struct __align__(32) particle_box_t
{
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

using rng_gen = std::uniform_real_distribution<double>;
void particle_box_add_particle_host(particle_box_t &box, particle_t const &p);
void particle_box_add_particle_host(particle_box_t &box, double radius);
void particle_box_add_particle_host(particle_box_t &box, double radius,
                               rng_gen &rng_x, rng_gen &rng_y,
                               rng_gen &rng_z, std::mt19937 &re);
__device__ void
particle_box_add_particle_device(particle_box_t &box,
                                 particle_t const &p);

__host__ __device__ void
particle_box_remove_particle(particle_box_t &p,
                             size_t idx);

__host__ __device__ void
particle_box_swap_particles(particle_box_t &p,
                            size_t const fst,
                            size_t const snd);

inline void particle_box_free_particles_host(particle_box_t &p)
{
    for (size_t idx = 0; idx < p.particle_count; idx++)
    {
        delete[] p.particles[idx].patches;
    }
    delete[] p.particles;
    p.particle_count = 0;
    p.capacity = 0;
}

__host__ __device__ void particle_box_free_particles_device(particle_box_t &p);

particle_box_t make_box(particle_box_t const &box);

#endif
