#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

#include <random>

#include "particle.h"
#include "particle_box.h"
#include "patch.h"

static constexpr size_t MAX_PARTICLES_PER_CELL = 6;
using rng_gen = std::uniform_real_distribution<double>;

struct __align__(32) cell_t {
  size_t num_particles{0};
  size_t particle_indices[MAX_PARTICLES_PER_CELL]{};
  double energy{};
};

struct __align__(32) cell_view_t {
  particle_box_t box{};
  cell_t *cells{};
  cell_t *cells_device{};
  size_t cells_per_axis{};
  double3 cell_size{};

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

  void update_cell(size_t const cell_idx);
  void alloc_cells();
  void free();
  void free_cells();
  void realloc(size_t cap);
  void add_particle_to_box(particle_t const &p);
  bool add_particle_to_box(double radius, rng_gen &rng_x, rng_gen &rng_y,
                           rng_gen &rng_z, std::mt19937 &re);
  void add_particle_random_pos(double radius, rng_gen &rng_x, rng_gen &rng_y,
                               rng_gen &rng_z, std::mt19937 &re);
  double3 try_random_particle_disp(size_t const particle_idx, rng_gen &rng_x,
                                   std::mt19937 &re, double const scale = 2.0F);
  __host__ __device__ double3 try_random_particle_disp(
      size_t const particle_idx, double const offset,
      double const scale = 2.0F);

  bool add_particle(particle_t const &p);
  void remove_particle(particle_t const &p);
  void remove_particle_from_box(particle_t const &p);
  double particle_energy_square_well(
      particle_t const &p, double const sigma = 0.2f, double const val = 1.0f);
  double particle_energy_patch(particle_t const &p, double const cosmax = 0.96,
                               double const sigma = 0.2,
                               double const epsilon = 0.2);
  double particle_energy_square_well_device(
      particle_t const &p, double const sigma = 0.2f, double const val = 1.0f);
  double particles_in_range(const size_t idx, const double r1, const double r2)
      const;
  double total_energy(double const sigma = 0.2F, double const val = 1.0F);
  double try_move_particle(size_t const p_idx, double3 const new_pos,
                           double4 const rotation, double prob_r, double temp);

  inline constexpr __host__ __device__ size_t get_cell_idx(double3 const &p) {
    uint3 const particle_idx = {
        .x = (uint32_t)(p.x / cell_size.x),
        .y = (uint32_t)(p.y / cell_size.y),
        .z = (uint32_t)(p.z / cell_size.z),
    };
    return particle_idx.x * cells_per_axis * cells_per_axis +
           particle_idx.y * cells_per_axis + particle_idx.z;
  }

  __host__ __device__ bool particle_intersects(particle_t const &p);
};
__global__ void energy_square_well(cell_view_t const view, particle_t const &p,
                                   double *output, int3 strides,
                                   double3 coeff_vals,
                                   double const sigma = 2.0F,
                                   double const val = 1.0F);

#endif
