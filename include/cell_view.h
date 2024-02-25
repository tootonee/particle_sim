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

void cell_view_init_host(cell_view_t &view, particle_box_t box,
                         size_t const cells_per_axis);
void cell_view_add_particle_host(cell_view_t &view, size_t num_cells);
void cell_view_upscape_host(cell_view_t &view, size_t scale);

__host__ __device__ cell_t *cell_view_get_cell(cell_view_t *view,
                                               particle_t const &particle);
__host__ __device__ void cell_view_get_particle_idx(cell_view_t *view,
                                                    particle_t const &particle);

__device__ cell_view_t cell_view_init_device(cell_view_t const &view);
__device__ void cell_view_add_particle_device(cell_view_t &view,
                                              particle_box_t *box,
                                              size_t num_cells);
__device__ void cell_view_upscape_device(cell_view_t &view, size_t scale);

#endif
