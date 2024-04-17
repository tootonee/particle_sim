#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

#include <random>

#include "particle.h"
#include "particle_box.h"
#include "patch.h"

static constexpr size_t MAX_PARTICLES_PER_CELL = 255;
static constexpr size_t MAX_PARTICLES_PER_CELL_ALIGNED = 256;
using rng_gen = std::uniform_real_distribution<double>;

struct cell_t {
  size_t num_particles{0};
  size_t particle_indices[MAX_PARTICLES_PER_CELL];
};

struct cell_view_t {
  particle_box_t box{};
  cell_t *cells{};
  size_t cells_per_axis{};
  size_t *cell_indices{};
  double3 cell_size{};
  double *energies_device{};
  double *energies{};

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
  void add_particle_random_pos(double radius, rng_gen &rng_x, rng_gen &rng_y,
                               rng_gen &rng_z, std::mt19937 &re);
  double3 try_random_particle_disp(size_t const particle_idx,
                                   double3 const offset,
                                   double const scale = 2.0F);

  bool add_particle(particle_t const &p);
  void remove_particle(particle_t const &p);
  void remove_particle_from_box(particle_t const &p);
  double particle_energy_square_well(particle_t const &p,
                                     double const sigma = 0.2f,
                                     double const val = 1.0f);
  double particle_energy_yukawa(particle_t const &p);
  double particle_energy_patch(particle_t const &p, double const cosmax = 0.92,
                               double const epsilon = 0.2);
  double particle_energy_yukawa_device(particle_t const p);
  double particles_in_range(const size_t idx, const double r1,
                            const double r2) const;
  double total_energy(double const sigma = 0.2F, double const val = 1.0F);
  double try_move_particle(size_t const p_idx, double3 const new_pos,
                           double4 const rotation, double prob_r, double temp);
  double try_move_particle_device(size_t const p_idx, double3 const new_pos,
                                  double4 const rotation, double prob_r,
                                  double temp);

  inline constexpr __host__ __device__ size_t get_cell_idx(double3 const &p) {
    uint3 const particle_idx = {
        .x = (uint32_t)(p.x / cell_size.x),
        .y = (uint32_t)(p.y / cell_size.y),
        .z = (uint32_t)(p.z / cell_size.z),
    };
    return particle_idx.x * cells_per_axis * cells_per_axis +
           particle_idx.y * cells_per_axis + particle_idx.z;
  }

  bool particle_intersects(particle_t const &p);
};
#endif
