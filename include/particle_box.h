#pragma once

#ifndef PARTICLE_BOX_H
#define PARTICLE_BOX_H

#include <cstring>
#include <random>

#include "particle.h"

using rng_gen = std::uniform_real_distribution<double>;

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

struct particle_box_t {
  particle_t *particles{};
  double3 dimensions{};

  particle_t *particles_device{};
  size_t particle_count{};
  size_t capacity{};

  void init(size_t cap = DEFAULT_CAPACITY);
  void realloc(size_t cap = DEFAULT_CAPACITY);

  void add_particle(particle_t const &p);
  void add_particle(double radius, rng_gen &rng_x, rng_gen &rng_y,
                    rng_gen &rng_z, std::mt19937 &re);

  void remove_particle(size_t idx);
  void swap_particles(size_t const fst, size_t const snd);

  inline void free_particles() const {
    if (particles_device != nullptr) {
      cudaFree(particles_device);
    }
    delete[] particles;
  }

  void make_box_uniform_particles_host(double3 const dims, double const radius,
                                       size_t const count_per_axis);

  inline void update_particles() {
    cudaMemcpy(particles_device, particles, sizeof(particle_t) * capacity,
               cudaMemcpyHostToDevice);
  }

  inline void update_particle_async(size_t p_idx) {
    if (p_idx >= capacity) {
      return;
    }

    particle_t *dev = particles_device + p_idx;
    const particle_t *host = particles + p_idx;
    // gpuErrchk(
    //     cudaMemcpy(dev, host, sizeof(particle_t), cudaMemcpyHostToDevice));
    cudaMemcpyAsync(dev, host, sizeof(particle_t), cudaMemcpyHostToDevice);
  }

  inline void update_particle(size_t p_idx) {
    if (p_idx >= capacity) {
      return;
    }

    particle_t *dev = particles_device + p_idx;
    const particle_t *host = particles + p_idx;
    // gpuErrchk(
    //     cudaMemcpy(dev, host, sizeof(particle_t), cudaMemcpyHostToDevice));
    cudaMemcpy(dev, host, sizeof(particle_t), cudaMemcpyHostToDevice);
  }
};
#endif
