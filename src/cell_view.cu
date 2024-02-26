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
    view.box = box;
    for (size_t i = 0; i < cell_count; i++) {
        view.cells[i].num_particles = 0;
    }
}

void cell_view_init_host(cell_view_t &view, particle_box_t box, 
    size_t const cells_per_axis) {
    cell_view_alloc_host(view, box, cells_per_axis);
    size_t p_idx = 0;

    while (p_idx < box.particle_count) {
        if (!cell_view_add_particle(view, box.particles[p_idx])) {
            view.cells_per_axis *= 2;
            cell_view_free_host(view);
            cell_view_alloc_host(view, box, view.cells_per_axis);
            p_idx = 0;
        }
        p_idx++;
    }
}

void cell_view_add_particle_host(cell_view_t &view,
                                    double radius,
                                    rng_gen &rng_x, rng_gen &rng_y,
                                    rng_gen &rng_z, std::mt19937 &re) {
  particle_t p{};
  p.radius = radius;
  p.idx = view.box.particle_count;
  particle_init_host(p);
  do {
    random_particle_pos(p, rng_x, rng_y, rng_z, re);
  } while (cell_view_particle_intersects(view, p));
  cell_view_add_particle_to_box_host(view, p);
}

void cell_view_add_particle_to_box_host(cell_view_t &view,
    particle_t const &p) {
    size_t p_idx = view.box.particle_count;
    particle_box_add_particle_host(view.box, p);

    while (p_idx < view.box.particle_count) {
        if (!cell_view_add_particle(view, view.box.particles[p_idx])) {
            view.cells_per_axis *= 2;
            cell_view_free_host(view);
            cell_view_alloc_host(view, view.box, view.cells_per_axis);
            p_idx = 0;
        }
        p_idx++;
    }
}

__host__ __device__ bool cell_view_add_particle(
    cell_view_t &view, particle_t const & p
) {
    size_t cell_idx = cell_view_get_cell_idx(view, p);
    cell_t &cell = view.cells[cell_idx];
    if (cell.num_particles >= MAX_PARTICLES_PER_CELL) {
        return false;
    }
    cell.particle_indices[cell.num_particles++] = p.idx;
    return true;
}

__host__ __device__ void cell_view_remove_particle(cell_view_t &view, particle_t const &p) {
    size_t cell_idx = cell_view_get_cell_idx(view, p);
    cell_t &cell = view.cells[cell_idx];
    if (cell.num_particles == 0) {
        return;
    }
    for (size_t i = 0; i < cell.num_particles; i++) {
        if (cell.particle_indices[i] == p.idx) {
            cell.particle_indices[i] = cell.particle_indices[cell.num_particles - 1];
            cell.num_particles--;
            return;
        }
    }

}

__host__ __device__ bool cell_view_particle_intersects(cell_view_t const &view,
    particle_t const &p) {
    // check cell with particle, also neighboring cells by combining different
    // combinations of -1, 0, 1 for each axis
    for (double coeff_x = -1; coeff_x <= 1; coeff_x += 1) {
        for (double coeff_y = -1; coeff_y <= 1; coeff_y += 1) {
            for (double coeff_z = -1; coeff_z <= 1; coeff_z += 1) {
                particle_t p1 {};
                p1.pos = {
                    .x = p.pos.x + coeff_x * view.cell_size.x,
                    .y = p.pos.y + coeff_y * view.cell_size.y,
                    .z = p.pos.z + coeff_z * view.cell_size.z,
                };
                size_t cell_idx = cell_view_get_cell_idx(view, p1);
                if (cell_idx >= view.cells_per_axis * view.cells_per_axis * view.cells_per_axis) {
                    continue;
                }
                cell_t const &cell = view.cells[cell_idx];
                for (size_t i = 0; i < cell.num_particles; i++) {
                    if (particle_intersects(p, view.box.particles[cell.particle_indices[i]])) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}