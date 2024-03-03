#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>

#include "particle.h"
#include "particle_box.h"
#include "patch.h"

static constexpr size_t MAX_PARTICLES_PER_CELL = 7;
using rng_gen = std::uniform_real_distribution<double>;

struct __align__(32) cell_t {
  size_t num_particles{0};
  size_t particle_indices[MAX_PARTICLES_PER_CELL]{};
};

struct __align__(32) cell_view_t {
  particle_box_t box{};
  cell_t *cells{};
  size_t cells_per_axis{};
  double3 cell_size{};

  cell_view_t(particle_box_t const &b, size_t const cs_per_axis) {
    size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
    box.dimensions = b.dimensions;
    box.particle_count = b.particle_count;
    box.capacity = b.capacity;
    box.init(box.capacity);
    cudaMemcpy(box.particles, b.particles, sizeof(particle_t) * cell_count,
               cudaMemcpyDefault);

    cells_per_axis = cs_per_axis;
    cell_size = {
        .x = box.dimensions.x / cells_per_axis,
        .y = box.dimensions.y / cells_per_axis,
        .z = box.dimensions.z / cells_per_axis,
    };
    alloc_cells();

    for (size_t idx = 0; idx < box.particle_count; idx++) {
      box.particles[idx].init(b.particles[idx].patch_cap);
      cudaMemcpy(box.particles[idx].patches, b.particles[idx].patches,
                 sizeof(patch_t) * b.particles[idx].patch_count,
                 cudaMemcpyDefault);
      box.particles[idx].patch_cap = b.particles[idx].patch_cap;
      box.particles[idx].patch_count = b.particles[idx].patch_count;
      box.particles[idx].idx = idx;
      add_particle(box.particles[idx]);
    }
  }

  void alloc_cells();
  void free();
  void free_cells();
  void realloc(size_t cap);
  void add_particle_to_box(particle_t const &p);
  double3 try_random_particle_disp(size_t const particle_idx, rng_gen &rng_x,
                                   std::mt19937 &re);
  bool add_particle(double radius, rng_gen &rng_x, rng_gen &rng_y,
                    rng_gen &rng_z, std::mt19937 &re);

  __host__ __device__ bool add_particle(particle_t const &p);
  __host__ __device__ void remove_particle(particle_t const &p);
  __host__ __device__ bool particle_intersects(particle_t const &p);
  __host__ __device__ bool particle_intersects(double3 const pos,
                                               double const radius);

  inline __host__ __device__ size_t get_cell_idx(particle_t const &p) {
    uint3 const particle_idx = {
        .x = (uint32_t)(p.pos.x / cell_size.x),
        .y = (uint32_t)(p.pos.y / cell_size.y),
        .z = (uint32_t)(p.pos.z / cell_size.z),
    };
    return particle_idx.x * cells_per_axis * cells_per_axis +
           particle_idx.y * cells_per_axis + particle_idx.z;
  }

  inline __host__ __device__ size_t get_cell_idx(double3 const &p) {
    uint3 const particle_idx = {
        .x = (uint32_t)(p.x / cell_size.x),
        .y = (uint32_t)(p.y / cell_size.y),
        .z = (uint32_t)(p.z / cell_size.z),
    };
    return particle_idx.x * cells_per_axis * cells_per_axis +
           particle_idx.y * cells_per_axis + particle_idx.z;
  }
};

#endif
