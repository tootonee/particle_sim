#include "cell_view.h"
#include "particle_box.h"
#include "particle.h"
#include <cmath>

void cell_view_free_host(cell_view_t &view) {
    delete[] view.cells;
}

void cell_view_alloc_host(
    cell_view_t &view, particle_box_t box,
    size_t const cells_per_axis
) {
    double3 cell_size = {
        .x = box.dimensions.x / cells_per_axis,
        .y = box.dimensions.y / cells_per_axis,
        .z = box.dimensions.z / cells_per_axis
    };
    size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
    view.cells_per_axis = cells_per_axis;
    view.cells = new cell_t[cell_count];
    view.cell_size = cell_size;
}

void cell_view_init_host(cell_view_t &view, particle_box_t box, 
    size_t const cells_per_axis) {
    cell_view_alloc_host(view, box, cells_per_axis);
    size_t p_idx = 0;

    while (p_idx != box.particle_count) {
        if (!cell_view_add_particle(view, box.particles[p_idx])) {
            view.cells_per_axis *= 2;
            cell_view_free_host(view);
            cell_view_alloc_host(view, box, view.cells_per_axis);
            p_idx = 0;
        }
        p_idx++;
    }
}

__host__ __device__ bool cell_view_add_particle(
    cell_view_t &view, particle_t const p
) {
    size_t cell_idx = cell_view_get_cell_idx(view, p);
    cell_t &cell = view.cells[cell_idx];
    if (cell.num_particles == MAX_PARTICLES_PER_CELL) {
        return false;
    }
    cell.particle_indices[cell.num_particles++] = p.idx;
    return true;
}

__host__ __device__ size_t cell_view_get_cell_idx(
    cell_view_t const &view, particle_t const &p
) {
    uint3 particle_idx = {
        .x = static_cast<uint32_t>(p.pos.x / view.cell_size.x),
        .y = static_cast<uint32_t>(p.pos.y / view.cell_size.y),
        .z = static_cast<uint32_t>(p.pos.z / view.cell_size.z),
    };
    return particle_idx.x * view.cells_per_axis * view.cells_per_axis +
        particle_idx.y * view.cells_per_axis + particle_idx.z;
}