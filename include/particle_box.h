#pragma once

#ifndef PARTICLE_BOX_H
#define PARTICLE_BOX_H

#include <random>
#include <cstring>

#include "particle.h"

using rng_gen = std::uniform_real_distribution<double>;

struct __align__(32) particle_box_t {
  particle_t *particles{};
  particle_t *particles_device{};
  double3 dimensions{};
  size_t particle_count{};
  size_t capacity{};

  void init(size_t cap = DEFAULT_CAPACITY);
  void realloc(size_t cap = DEFAULT_CAPACITY);

  void update_particle(size_t const p_idx);
  void add_particle(particle_t const &p);
  void add_particle(double radius, rng_gen &rng_x, rng_gen &rng_y,
                    rng_gen &rng_z, std::mt19937 &re);

  void remove_particle(size_t idx);
  void swap_particles(size_t const fst, size_t const snd);

  inline void free_particles() const {
    cudaFree(particles_device);
    delete[] particles;
  }

  void make_box_uniform_particles_host(double3 const dims, double const radius,
                                       size_t const count_per_axis);
};
#endif
