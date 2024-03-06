#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

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
    cells_per_axis = cs_per_axis;
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
      cudaMemcpy(box.particles[idx].patches, b.particles[idx].patches,
                 sizeof(patch_t) * b.particles[idx].patch_count,
                 cudaMemcpyDefault);
      box.particles[idx].patch_count = b.particles[idx].patch_count;
      box.particles[idx].idx = idx;
      add_particle(box.particles[idx]);
    }
  }

  cell_view_t(double3 const dims, size_t const cs_per_axis) {
    cells_per_axis = cs_per_axis;
    size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
    box.dimensions = dims;
    box.particle_count = 0;
    box.capacity = cell_count;
    box.init(box.capacity);

    cells_per_axis = cs_per_axis;
    cell_size = {
        .x = box.dimensions.x / cells_per_axis,
        .y = box.dimensions.y / cells_per_axis,
        .z = box.dimensions.z / cells_per_axis,
    };
    alloc_cells();
  }

  void alloc_cells();
  void free();
  void free_cells();
  void realloc(size_t cap);
  void add_particle_to_box(particle_t const &p);
  bool add_particle_to_box(double radius, rng_gen &rng_x, rng_gen &rng_y,
                           rng_gen &rng_z, std::mt19937 &re);
  double3 try_random_particle_disp(size_t const particle_idx, rng_gen &rng_x,
                                   std::mt19937 &re, double const scale = 2.0F);
  void add_particle_random_pos(double radius, rng_gen &rng_x, rng_gen &rng_y,
                               rng_gen &rng_z, std::mt19937 &re);

  __host__ __device__ bool add_particle(particle_t const &p);
  __host__ __device__ void remove_particle(particle_t const &p);
  __host__ __device__ void remove_particle_from_box(particle_t const &p);
  __host__ __device__ bool particle_intersects(particle_t const &p);
  __host__ __device__ bool particle_intersects(double3 const pos,
                                               double const radius);
  __host__ __device__ bool particle_energy_square_well(
      particle_t const &p, double const sigma = 2.0F, double const val = 1.0F);
  __host__ __device__ bool particle_energy_square_well(
      double3 const pos, double const radius, double const sigma = 2.0F,
      double const val = 1.0F);

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
