#pragma once

#ifndef CELL_VIEW_H
#define CELL_VIEW_H

#include "particle.h"
#include "particle_box.h"

struct __align__(32) cell_t
{
    size_t num_particles;
    size_t particle_indices[7];
};

struct __align__(32) cell_view_t
{
    particle_box_t *box;
    cell_t *cells;
    size_t num_cells;
};

void cell_view_init(cell_view_t *view, particle_box_t *box, size_t num_cells);

#endif