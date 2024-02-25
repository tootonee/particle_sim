#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

#include "particle.h"
#include "particle_box.h"

struct __align__(32) cell_t {
    size_t num_particles;
    size_t particle_indices[7];
};

struct __align__(32) cell_view_t {
    particle_box_t box;
    cell_t *cells;
    size_t cells_per_axis;
};

struct __align__(32) cell_block_t {
    cell_t cells[8];
    size_t num_cells
};

void cell_view_init_host(cell_view_t &view, particle_box_t box,
                         size_t const cells_per_axis);
void cell_view_add_particle_host(cell_view_t &view, particle_t const p);
void cell_view_upscape_host(cell_view_t &view, size_t scale);

__device__ cell_view_t cell_view_device_from_host_obj(cell_view_t const &view);
__device__ void cell_view_add_particle_device(cell_view_t &view,
                                              particle_t const p);
__device__ void cell_view_upscape_device(cell_view_t &view, size_t const scale);

__host__ __device__ size_t cell_view_get_particle_idx(cell_t const &cell,
                                               particle_t const &p);
__host__ __device__ size_t cell_view_get_cell_idx(cell_view_t const &view,
                                               particle_t const &p);
__host__ __device__ cell_block_t cell_view_get_particle_neighbourhood(cell_view_t *view,
                                                    particle_t const &p);
__host__ __device__ bool particle_intersects(cell_view_t const &view,
                                                particle_t const &p);

#endif
