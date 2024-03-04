#pragma once

#ifndef PARTICLE_BOX_H
#define PARTICLE_BOX_H

#include <random>

#include "particle.h"

using rng_gen = std::uniform_real_distribution<double>;

struct __align__(32) particle_box_t {
  particle_t *particles{};
  double3 dimensions{};
  size_t particle_count{};
  size_t capacity{};

  void init(size_t cap = DEFAULT_CAPACITY);
  void realloc(size_t cap = DEFAULT_CAPACITY);

  void add_particle(particle_t const &p);
  void add_particle(double radius);
  void add_particle(double radius, rng_gen &rng_x, rng_gen &rng_y,
                    rng_gen &rng_z, std::mt19937 &re);

  __host__ __device__ void remove_particle(size_t idx);

  __host__ __device__ void swap_particles(size_t const fst, size_t const snd);

  inline void free_particles() const {
    for (size_t idx = 0; idx < particle_count; idx++) {
      cudaFree(particles[idx].patches);
      particles[idx].patch_cap = 0;
      particles[idx].patch_count = 0;
    }
    cudaFree(particles);
  }

  void make_box_uniform_particles_host(double3 const dims, double const radius,
                                       size_t const count_per_axis);
};
#endif
