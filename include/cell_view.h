#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

#include "particle.h"
#include "particle_box.h"

static constexpr size_t MAX_PARTICLES_PER_CELL = 7;

struct __align__(32) cell_t {
  size_t num_particles{0};
  size_t particle_indices[7]{};
};

struct __align__(32) cell_view_t {
  particle_box_t box{};
  cell_t *cells{};
  size_t cells_per_axis{};
  double3 cell_size{};
};

void cell_view_alloc_host(cell_view_t &view, particle_box_t box,
                          size_t const cells_per_axis);
void cell_view_init_host(cell_view_t &view, particle_box_t box,
                         size_t const cells_per_axis);
void cell_view_free_host(cell_view_t &view);
void cell_view_add_particle_to_box_host(cell_view_t &view, particle_t const &p);
void cell_view_add_particle_host(cell_view_t &view, double radius,
                                 rng_gen &rng_x, rng_gen &rng_y, rng_gen &rng_z,
                                 std::mt19937 &re);
double3 cell_view_try_random_particle_pos(cell_view_t &view, size_t const particle_idx,
                                 rng_gen &rng_x, std::mt19937 &re);

cell_view_t cell_view_device_from_host_obj(cell_view_t const &view);

void cell_view_alloc_device(cell_view_t &view,
                                       __device__ particle_box_t box,
                                       size_t const cells_per_axis);
void cell_view_add_particle_to_box_device(cell_view_t &view,
                                                     particle_t const &p);

__host__ __device__ void cell_view_remove_particle(cell_view_t &view, particle_t const &p);
__host__ __device__ void cell_view_free_device(cell_view_t &view);
__host__ __device__ bool cell_view_add_particle(cell_view_t &view,
                                                particle_t const &p);
__host__ __device__ void cell_view_remove_particle(cell_view_t &view,
                                                   particle_t const &p);
__host__ __device__ bool cell_view_particle_intersects(cell_view_t const &view,
                                                       particle_t const &p);

__host__ __device__ bool cell_view_particle_intersects(cell_view_t const &view,
    double3 const pos, double const radius);

inline __host__ __device__ size_t
cell_view_get_cell_idx(cell_view_t const &view, particle_t const &p) {
  uint3 const particle_idx = {
      .x = (uint32_t)(p.pos.x / view.cell_size.x),
      .y = (uint32_t)(p.pos.y / view.cell_size.y),
      .z = (uint32_t)(p.pos.z / view.cell_size.z),
  };
  return particle_idx.x * view.cells_per_axis * view.cells_per_axis +
         particle_idx.y * view.cells_per_axis + particle_idx.z;
}

#endif
