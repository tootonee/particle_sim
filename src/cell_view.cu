#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "vec.h"
#include <cmath>

void cell_view_t::alloc_cells() {
    size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;

    cells = new cell_t[cell_count];

    energies = new double[MAX_PARTICLES_PER_CELL + 1];
    cudaMalloc(&energies_device, sizeof(double) * MAX_PARTICLES_PER_CELL);
    cudaMalloc(&cell_indices,
               sizeof(size_t) * cell_count * MAX_PARTICLES_PER_CELL);
}

void cell_view_t::free_cells() { delete[] cells; }

void cell_view_t::free() {
    box.free_particles();
    free_cells();
    delete[] energies;
    cudaFree(energies_device);
    cudaFree(cell_indices);
}

bool cell_view_t::add_particle_async(particle_t const &p) {
    size_t cell_idx = get_cell_idx(p.pos);
    if (cell_idx >= cells_per_axis * cells_per_axis * cells_per_axis) {
        return false;
    }
    cell_t &cell = cells[cell_idx];
    if (cell.num_particles >= MAX_PARTICLES_PER_CELL) {
        return false;
    }
    cell.particle_indices[cell.num_particles] = p.idx;
    cell.num_particles += 1;
    cudaMemcpyAsync(cell_indices + cell_idx * MAX_PARTICLES_PER_CELL,
                    cell.particle_indices, sizeof(size_t) * cell.num_particles,
                    cudaMemcpyHostToDevice);
    return true;
}

bool cell_view_t::add_particle(particle_t const &p) {
    size_t cell_idx = get_cell_idx(p.pos);
    if (cell_idx >= cells_per_axis * cells_per_axis * cells_per_axis) {
        return false;
    }
    cell_t &cell = cells[cell_idx];
    if (cell.num_particles >= MAX_PARTICLES_PER_CELL) {
        return false;
    }
    cell.particle_indices[cell.num_particles] = p.idx;
    cell.num_particles += 1;
    cudaMemcpy(cell_indices + cell_idx * MAX_PARTICLES_PER_CELL,
               cell.particle_indices, sizeof(size_t) * cell.num_particles,
               cudaMemcpyHostToDevice);
    return true;
}

void cell_view_t::add_particle_to_box(particle_t const &p) {
    size_t p_idx = box.particle_count;
    box.add_particle(p);

    while (p_idx < box.particle_count) {
        if (!add_particle(box.particles[p_idx])) {
            free_cells();
            alloc_cells();
            p_idx = 0;
        }
        p_idx++;
    }
}

double3 cell_view_t::try_random_particle_disp(size_t const particle_idx,
                                              double3 const offset,
                                              double const scale) {
    if (particle_idx >= box.particle_count) {
        return {-1, -1, -1};
    }
    particle_t const &p_orig = box.particles[particle_idx];
    const double3 old_pos = p_orig.pos;

    double3 disp = {
            2 * scale * offset.x,
            2 * scale * offset.y,
            2 * scale * offset.z,
    };

    disp = {
            p_orig.pos.x + disp.x,
            p_orig.pos.y + disp.y,
            p_orig.pos.z + disp.z,
    };

    if (disp.x < 0) {
        disp.x += box.dimensions.x;
    }

    if (disp.y < 0) {
        disp.y += box.dimensions.y;
    }

    if (disp.z < 0) {
        disp.z += box.dimensions.z;
    }

    if (disp.x >= box.dimensions.x) {
        disp.x -= box.dimensions.x;
    }

    if (disp.y >= box.dimensions.y) {
        disp.y -= box.dimensions.y;
    }

    if (disp.z >= box.dimensions.z) {
        disp.z -= box.dimensions.z;
    }

//    box.particles[particle_idx].pos = disp;

    if (particle_intersects(box.particles[particle_idx])) {
//        box.particles[particle_idx].pos = old_pos;
        return {-1, -1, -1};
    }

//    box.particles[particle_idx].pos = old_pos;
    return disp;
}

__host__ __device__ double3 try_random_particle_disp_device(
        cell_t* cells,
        double3 cell_size,
        size_t* cell_indices,
        size_t cell_count,
        particle_t* particles,
        size_t particle_count,
        double3 box_dimensions,
        size_t particle_idx,
        double3 offset
) {
    double MAX_STEP = 0.5;
    if (particle_idx >= particle_count) {
        return make_double3(-1, -1, -1);
    }
    particle_t const &p_orig = particles[particle_idx];
    const double3 old_pos = p_orig.pos;
    double3 disp = make_double3(
            2 * MAX_STEP * offset.x,
            2 * MAX_STEP * offset.y,
            2 * MAX_STEP * offset.z
    );

    disp = make_double3(
            p_orig.pos.x + disp.x,
            p_orig.pos.y + disp.y,
            p_orig.pos.z + disp.z
    );

    if (disp.x < 0) {
        disp.x += box_dimensions.x;
    }
    if (disp.y < 0) {
        disp.y += box_dimensions.y;
    }
    if (disp.z < 0) {
        disp.z += box_dimensions.z;
    }
    if (disp.x >= box_dimensions.x) {
        disp.x -= box_dimensions.x;
    }
    if (disp.y >= box_dimensions.y) {
        disp.y -= box_dimensions.y;
    }
    if (disp.z >= box_dimensions.z) {
        disp.z -= box_dimensions.z;
    }
        particles[particle_idx].pos = disp;
    if (particle_intersects_device(particles, particle_idx, cell_size,box_dimensions,
                                   cell_indices,
                                   cell_count,
                                   cells)) {
        particles[particle_idx].pos = old_pos;
        return make_double3(-1, -1, -1);
    }

    particles[particle_idx].pos = old_pos;  //to make outside of the function
    return disp;
}
// double3 cell_view_t::try_random_particle_disp(size_t const particle_idx,
//                                               rng_gen &rng, std::mt19937 &re,
//                                               double const scale) {
//   if (particle_idx >= box.particle_count) {
//     return {-1, -1, -1};
//   }
//
//   particle_t const &p_orig = box.particles[particle_idx];
//   const double3 old_pos = p_orig.pos;
//
//   double3 disp = {
//       2 * scale * (rng(re) - 0.5),
//       2 * scale * (rng(re) - 0.5),
//       2 * scale * (rng(re) - 0.5),
//   };
//
//   disp = {
//       p_orig.pos.x + disp.x,
//       p_orig.pos.y + disp.y,
//       p_orig.pos.z + disp.z,
//   };
//
//   if (disp.x < 0) {
//     disp.x += box.dimensions.x;
//   }
//
//   if (disp.y < 0) {
//     disp.y += box.dimensions.y;
//   }
//
//   if (disp.z < 0) {
//     disp.z += box.dimensions.z;
//   }
//
//   if (disp.x >= box.dimensions.x) {
//     disp.x -= box.dimensions.x;
//   }
//
//   if (disp.y >= box.dimensions.y) {
//     disp.y -= box.dimensions.y;
//   }
//
//   if (disp.z >= box.dimensions.z) {
//     disp.z -= box.dimensions.z;
//   }
//
//   box.particles[particle_idx].pos = disp;
//
//   if (particle_intersects(box.particles[particle_idx])) {
//     box.particles[particle_idx].pos = old_pos;
//     return {-1, -1, -1};
//   }
//
//   box.particles[particle_idx].pos = old_pos;
//   return disp;
// }

void cell_view_t::remove_particle(particle_t const &p) {
    size_t cell_idx = get_cell_idx(p.pos);
    cell_t &cell = cells[cell_idx];
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

void cell_view_t::remove_particle_from_box(particle_t const &p) {
    size_t cell_idx = get_cell_idx(p.pos);
    cell_t &cell = cells[cell_idx];
    if (cell.num_particles == 0) {
        return;
    }

    for (size_t i = 0; i < cell.num_particles; i++) {
        if (cell.particle_indices[i] == p.idx) {
            cell.particle_indices[i] = cell.particle_indices[cell.num_particles - 1];
            cell.num_particles--;
            break;
        }
    }
    if (p.idx <= box.particle_count && box.particles[p.idx] == p) {
        box.remove_particle(p.idx);
    }
}

bool cell_view_t::particle_intersects(particle_t const &p) {
    const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
    // check cell with particle, also neighboring cells by combining different
    // combinations of -1, 0, 1 for each axis
    double3 coeff_val = {
            .x = 2 * ceil(p.radius / cell_size.x),
            .y = 2 * ceil(p.radius / cell_size.y),
            .z = 2 * ceil(p.radius / cell_size.z),
    };
    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        if (x < 0) {
            x += box.dimensions.x;
        }
        if (x > box.dimensions.x) {
            x -= box.dimensions.x;
        }
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) {
                y += box.dimensions.y;
            }
            if (y > box.dimensions.y) {
                y -= box.dimensions.y;
            }
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
                 coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) {
                    z += box.dimensions.z;
                }
                if (z > box.dimensions.z) {
                    z -= box.dimensions.z;
                }
                size_t cell_idx = get_cell_idx((double3){x, y, z});
                if (cell_idx >= cell_cnt) {
                    continue;
                }
                cell_t const &cell = cells[cell_idx];
                for (size_t i = 0; i < cell.num_particles; i++) {
                    if (box.particles[cell.particle_indices[i]].intersects(p)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}
__host__ __device__ bool particle_intersects_device(
        particle_t* particles,
        size_t particle_idx,
        double3 cell_size,
        double3 box_dimensions,
        size_t* cell_indices,
        size_t cell_count,
        cell_t* cells
) {
    const size_t cell_cnt = cell_count * cell_count * cell_count;
    particle_t const &p = particles[particle_idx];
    double3 coeff_val = {
            .x = 2 * ceil(p.radius / cell_size.x),
            .y = 2 * ceil(p.radius / cell_size.y),
            .z = 2 * ceil(p.radius / cell_size.z),
    };

    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        if (x < 0) x += box_dimensions.x;
        if (x > box_dimensions.x) x -= box_dimensions.x;
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) y += box_dimensions.y;
            if (y > box_dimensions.y) y -= box_dimensions.y;
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z; coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) z += box_dimensions.z;
                if (z > box_dimensions.z) z -= box_dimensions.z;
                double3 position = make_double3(x, y, z);
//                printf("X= %f\n", position.x);
//                printf("Y= %f\n", position.y);
//                printf("Z= %f\n", position.z);
                size_t cell_idx = get_cell_idx_device(position, cell_size, box_dimensions);
//                printf("cell_idx %d\n", cell_idx);
                if (cell_idx >= cell_cnt) continue;
                cell_t const &cell = cells[cell_idx];
//                printf("NUM OF PARTICLES %d\n", cell.num_particles);
                for (size_t i = 0; i < cell.num_particles; i++) {
                    if (particles[cell.particle_indices[i]].intersects(p)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

__host__ __device__ size_t get_cell_idx_device(double3 position, double3 cell_size, double3 box_dimensions) {
    uint3 idx = {
            .x = static_cast<uint>(position.x / cell_size.x) % static_cast<uint>(box_dimensions.x / cell_size.x),
            .y = static_cast<uint>(position.y / cell_size.y) % static_cast<uint>(box_dimensions.y / cell_size.y),
            .z = static_cast<uint>(position.z / cell_size.z) % static_cast<uint>(box_dimensions.z / cell_size.z)
    };
    return idx.x + idx.y * (uint)(box_dimensions.x / cell_size.x) + idx.z * (uint)(box_dimensions.x / cell_size.x) * (uint)(box_dimensions.y / cell_size.y);
}
void cell_view_t::add_particle_random_pos(double radius, rng_gen &rng_x,
                                          rng_gen &rng_y, rng_gen &rng_z,
                                          std::mt19937 &re) {
    if (box.capacity <= box.particle_count) {
        box.realloc(box.capacity * 2);
    }
    particle_t p{};
    p.radius = radius;
    p.idx = box.particle_count;
    do {
        p.random_particle_pos(rng_x, rng_y, rng_z, re);
    } while (particle_intersects(p) || !add_particle(p));
    box.particles[box.particle_count] = p;
    box.particle_count++;
}

double cell_view_t::particle_energy_yukawa(particle_t const &p) {
    const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
    double result = 0.0F;
    double const dist = 6;
    // check cell with particle, also neighboring cells by combining different
    // combinations of -1, 0, 1 for each axis
    double3 coeff_val = {
            .x = ceil(dist / cell_size.x),
            .y = ceil(dist / cell_size.y),
            .z = ceil(dist / cell_size.z),
    };
    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        double3 offset{0, 0, 0};
        if (x < 0) {
            x += box.dimensions.x;
            offset.x = box.dimensions.x;
        }
        if (x >= box.dimensions.x) {
            x -= box.dimensions.x;
            offset.x = -box.dimensions.x;
        }
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) {
                y += box.dimensions.y;
                offset.y = box.dimensions.y;
            }
            if (y >= box.dimensions.y) {
                y -= box.dimensions.y;
                offset.y = -box.dimensions.y;
            }
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
                 coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) {
                    z += box.dimensions.z;
                    offset.z = box.dimensions.z;
                }
                if (z >= box.dimensions.z) {
                    z -= box.dimensions.z;
                    offset.z = -box.dimensions.z;
                }

                const size_t cell_idx = get_cell_idx((double3){x, y, z});
                if (cell_idx >= cell_cnt) {
                    continue;
                }

                cell_t const &cell = cells[cell_idx];
                for (size_t i = 0; i < cell.num_particles; i++) {
                    particle_t const &part = box.particles[cell.particle_indices[i]];
                    if (part.idx == p.idx) {
                        continue;
                    }

                    double3 part_pos = {
                            .x = part.pos.x - offset.x,
                            .y = part.pos.y - offset.y,
                            .z = part.pos.z - offset.z,
                    };
                    double d_p_part = distance(p.pos, part_pos);
                    result += 0.5 * exp(-1.5 * (d_p_part - 1)) / d_p_part;
                }
            }
        }
    }
    return result;
}

double cell_view_t::particle_energy_square_well(particle_t const &p,
                                                double const sigma,
                                                double const val) {
    const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
    double result = 0.0F;
    double const dist = 2 * p.radius + sigma;
    // check cell with particle, also neighboring cells by combining different
    // combinations of -1, 0, 1 for each axis
    double3 coeff_val = {
            .x = ceil(dist / cell_size.x),
            .y = ceil(dist / cell_size.y),
            .z = ceil(dist / cell_size.z),
    };
    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        if (x < 0) {
            x += box.dimensions.x;
        }
        if (x >= box.dimensions.x) {
            x -= box.dimensions.x;
        }
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) {
                y += box.dimensions.y;
            }
            if (y >= box.dimensions.y) {
                y -= box.dimensions.y;
            }
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
                 coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) {
                    z += box.dimensions.z;
                }
                if (z >= box.dimensions.z) {
                    z -= box.dimensions.z;
                }

                const size_t cell_idx = get_cell_idx((double3){x, y, z});
                if (cell_idx >= cell_cnt) {
                    continue;
                }

                cell_t const &cell = cells[cell_idx];
                for (size_t i = 0; i < cell.num_particles; i++) {
                    particle_t const &part = box.particles[cell.particle_indices[i]];
                    if (part.idx == p.idx) {
                        continue;
                    }

                    double d_p_part = distance(p.pos, part.pos);
                    if (d_p_part <= dist) {
                        result += val;
                    }
                }
            }
        }
    }
    return result;
}

double cell_view_t::total_energy(double const sigma, double const val) {
    box.update_particles();
    double total = 0.0F;
    for (size_t p_idx = 0; p_idx <= box.particle_count; p_idx++) {
        total += particle_energy_yukawa_device(box.particles[p_idx]);
        // total += particle_energy_patch(box.particles[p_idx], 0.89, -2);
    }
    return total * 0.5L;
}

inline constexpr void check_scale(double &min_scale, int &cur_dir,
                                  int const &dir, double const &val) {
    if (min_scale < val) {
        min_scale = val;
        cur_dir = dir;
    }
}

__global__ void in_range_helper(size_t p_idx, double r1, double r2,
                                double3 box_size, size_t p_count,
                                particle_t *particles, double *vals) {
    size_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_idx >= p_count) {
        return;
    }
    if (thread_idx == p_idx) {
        vals[thread_idx] = 0;
        return;
    }
    double result = 0;
    particle_t const &p = particles[p_idx];
    particle_t const &part = particles[thread_idx];
    for (double coeff_x = -1; coeff_x < 2; coeff_x += 1) {
        for (double coeff_y = -1; coeff_y < 2; coeff_y += 1) {
            for (double coeff_z = -1; coeff_z < 2; coeff_z += 1) {
                const double3 p_pos = {
                        .x = part.pos.x + box_size.x * coeff_x,
                        .y = part.pos.y + box_size.y * coeff_y,
                        .z = part.pos.z + box_size.z * coeff_z,
                };

                const double dist = distance(p_pos, p.pos);
                if (dist <= r2 && dist >= r1) {
                    result += 1;
                }
            }
        }
    }
    vals[thread_idx] = result;
}

double cell_view_t::particles_in_range_device(const size_t idx, const double r1,
                                              const double r2) const {
    double *n_part{};
    cudaMalloc(&n_part, sizeof(double) * box.particle_count);
    double *n_part_host = new double[box.particle_count];
    double result = 0.0L;
    size_t blocks = box.particle_count / 256 + 1;
    size_t threads = 256;

    in_range_helper<<<blocks, threads>>>(idx, r1, r2, box.dimensions,
                                         box.particle_count, box.particles_device,
                                         n_part);
    cudaMemcpy(n_part_host, n_part, sizeof(double) * box.particle_count,
               cudaMemcpyDeviceToHost);
    cudaFree(n_part);
#pragma omp parallel for
    for (size_t i = 0; i < box.particle_count; i++) {
        result += n_part_host[i];
    }
    delete[] n_part_host;

    return result;
}

double cell_view_t::particles_in_range(const size_t idx, const double r1,
                                       const double r2) const {
    double result = 0.0L;
    const double3 &pos = box.particles[idx].pos;
    for (size_t p_idx = 0; p_idx < box.particle_count; p_idx++) {
        if (p_idx == idx) {
            continue;
        }

        const particle_t &part = box.particles[p_idx];
        for (double coeff_x = -1; coeff_x < 2; coeff_x += 1) {
            for (double coeff_y = -1; coeff_y < 2; coeff_y += 1) {
                for (double coeff_z = -1; coeff_z < 2; coeff_z += 1) {
                    const double3 p_pos = {
                            .x = part.pos.x + box.dimensions.x * coeff_x,
                            .y = part.pos.y + box.dimensions.y * coeff_y,
                            .z = part.pos.z + box.dimensions.z * coeff_z,
                    };

                    const double dist = distance(p_pos, pos);
                    if (dist <= r2 && dist >= r1) {
                        result += 1;
                    }
                }
            }
        }
    }
    return result;
}

__device__ inline void do_energy_calc(size_t p_idx, particle_t const *particles,
                                      size_t p_cnt, double cosmax, double sigma,
                                      double val, double &result, size_t offset,
                                      size_t idx, particle_t const &p) {

    size_t thread_idx = idx + offset;
    if (thread_idx >= p_cnt || thread_idx == p_idx) {
        return;
    }

    particle_t const &part = particles[thread_idx];
    double const dist = distance(p.pos, part.pos);

    if (dist > (p.radius + part.radius + sigma)) {
        return;
    }

    result += val;

    if (dist > p.radius * 2.238) {
        return;
    }

    result += p.interact(part, cosmax, 2 * val);
}

__global__ void energy_yukawa_helper(particle_t const p, double3 p_pos,
                                     size_t cell_idx, size_t num_p,
                                     size_t const *p_indices,
                                     particle_t const *particles,
                                     double *p_energies) {
    const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t thr_idx = cell_idx * MAX_PARTICLES_PER_CELL + t_idx;
    const size_t p_idx = p_indices[thr_idx];
    if (t_idx >= num_p) {
        return;
    }

    if (p_idx == p.idx) {
        p_energies[t_idx] = 0;
        return;
    }

    const particle_t &part = particles[p_idx];
    const double dist = distance(part.pos, p_pos);
    const double res = 0.5 * exp(-1.5 * (dist - 1)) / dist;
    p_energies[t_idx] = res;
}

double cell_view_t::particle_energy_yukawa_device(particle_t const p) {
    double const dist = 6;
    double3 coeff_val = {
            .x = ceil(dist / cell_size.x),
            .y = ceil(dist / cell_size.y),
            .z = ceil(dist / cell_size.z),
    };

    double result = 0.0F;
#pragma omp parallel for
    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        double3 offset{0, 0, 0};
        if (x < 0) {
            x += box.dimensions.x;
            offset.x = box.dimensions.x;
        }
        if (x >= box.dimensions.x) {
            x -= box.dimensions.x;
            offset.x = -box.dimensions.x;
        }
#pragma omp parallel for
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) {
                y += box.dimensions.y;
                offset.y = box.dimensions.y;
            }
            if (y >= box.dimensions.y) {
                y -= box.dimensions.y;
                offset.y = -box.dimensions.y;
            }
#pragma omp parallel for
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
                 coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) {
                    z += box.dimensions.z;
                    offset.z = box.dimensions.z;
                }
                if (z >= box.dimensions.z) {
                    z -= box.dimensions.z;
                    offset.z = -box.dimensions.z;
                }

                const size_t cell_idx = get_cell_idx((double3){x, y, z});
                cell_t const &cell = cells[cell_idx];

                if (cell.num_particles < 65) {
#pragma omp parallel for
                    for (size_t i = 0; i < cell.num_particles; i++) {
                        particle_t const &part = box.particles[cell.particle_indices[i]];
                        if (part.idx == p.idx) {
                            continue;
                        }

                        double3 part_pos = {
                                .x = part.pos.x - offset.x,
                                .y = part.pos.y - offset.y,
                                .z = part.pos.z - offset.z,
                        };

                        double d_p_part = distance(p.pos, part_pos);
                        result += 0.5 * exp(-1.5 * (d_p_part - 1)) / d_p_part;
                    }
                } else {
                    double3 p_pos = {
                            .x = p.pos.x + offset.x,
                            .y = p.pos.y + offset.y,
                            .z = p.pos.z + offset.z,
                    };

                    size_t blocks = cell.num_particles / 256 + 1;
                    size_t threads = cell.num_particles > 256 ? 256 : cell.num_particles;
                    energy_yukawa_helper<<<blocks, threads>>>(
                            p, p_pos, cell_idx, cell.num_particles, cell_indices,
                            box.particles_device, energies_device);
                    cudaMemcpy(energies, energies_device,
                               sizeof(double) * cell.num_particles,
                               cudaMemcpyDeviceToHost);
#pragma omp parallel for
                    for (size_t i = 0; i < cell.num_particles; i++) {
                        result += energies[i];
                    }
                }
            }
        }
    }
    return result;
}

__global__ void interact_helper(particle_t const p, size_t cell_idx,
                                size_t const *p_indices,
                                particle_t const *particles,
                                double *p_energies) {
    const int t_idx = threadIdx.x;
    size_t thr_idx = cell_idx * MAX_PARTICLES_PER_CELL + t_idx;
    const size_t p_idx = p_indices[thr_idx];
    if (p.idx == p_idx) {
        p_energies[t_idx] = 0;
        return;
    }
    const particle_t &part = particles[p_idx];
    const double dist = distance(part.pos, p.pos);
    if (dist >= 1.119) {
        p_energies[t_idx] = 0;
        return;
    }
    const double res = p.interact(part, 0.89, -250);
    p_energies[t_idx] = res;
}

double cell_view_t::particle_energy_patch_device(particle_t const p) {
    double const dist = 1.5;
    double3 coeff_val = {
            .x = ceil(dist / cell_size.x),
            .y = ceil(dist / cell_size.y),
            .z = ceil(dist / cell_size.z),
    };

    double result = 0.0F;
#pragma omp unroll partial
    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        if (x < 0) {
            x += box.dimensions.x;
        }
        if (x >= box.dimensions.x) {
            x -= box.dimensions.x;
        }
#pragma omp unroll partial
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) {
                y += box.dimensions.y;
            }
            if (y >= box.dimensions.y) {
                y -= box.dimensions.y;
            }
#pragma omp unroll partial
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
                 coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) {
                    z += box.dimensions.z;
                }
                if (z >= box.dimensions.z) {
                    z -= box.dimensions.z;
                }

                const size_t cell_idx = get_cell_idx((double3){x, y, z});
                cell_t const &cell = cells[cell_idx];

                if (cell.num_particles < 65) {
#pragma omp unroll partial
                    for (size_t i = 0; i < cell.num_particles; i++) {
                        particle_t const &part = box.particles[cell.particle_indices[i]];
                        if (part.idx == p.idx || distance(p.pos, part.pos) >= 1.119) {
                            continue;
                        }

                        result += p.interact(part);
                    }
                } else {
                    interact_helper<<<1, cell.num_particles>>>(
                            p, cell_idx, cell_indices, box.particles_device, energies_device);
                    cudaMemcpy(energies, energies_device,
                               sizeof(double) * MAX_PARTICLES_PER_CELL,
                               cudaMemcpyDeviceToHost);
#pragma omp unroll partial
                    for (size_t i = 0; i < cell.num_particles; i++) {
                        result += energies[i];
                    }
                }
            }
        }
    }
    return result;
}

double cell_view_t::try_move_particle(size_t const p_idx, double3 const new_pos,
                                      double4 const rotation, double prob_r,
                                      double temp) {
    if (new_pos.x == -1) {
        return 0;
    }

    particle_t const &part = box.particles[p_idx];
    particle_t p{part};

    // double old_energy = particle_energy_yukawa(p);
    double old_energy = particle_energy_yukawa(p) + particle_energy_patch(p);

    p.pos = new_pos;
    p.rotate(rotation);
    // double new_energy = particle_energy_yukawa(p);
    double new_energy = particle_energy_yukawa(p) + particle_energy_patch(p);
    double const d_energy = (new_energy - old_energy);

    double prob = exp(-(d_energy / temp));
    if (prob_r >= prob) {
        return 0;
    }

    remove_particle(part);
    box.particles[p_idx] = p;
    box.update_particle_async(p_idx);
    add_particle_async(p);
    cudaDeviceSynchronize();
    return d_energy;
}

double cell_view_t::try_move_particle_device(size_t const p_idx,
                                             double3 const new_pos,
                                             double4 const rotation,
                                             double prob_r, double temp) {
    if (new_pos.x == -1) {
        return 0;
    }

    particle_t const &part = box.particles[p_idx];
    particle_t p{part};

    // double old_energy = particle_energy_yukawa_device(p);
    double old_energy =
            particle_energy_yukawa_device(p) + particle_energy_patch_device(p);

    p.pos = new_pos;
    p.rotate(rotation);
    // double new_energy = particle_energy_yukawa_device(p);
    double new_energy =
            particle_energy_yukawa_device(p) + particle_energy_patch_device(p);
    double const d_energy = (new_energy - old_energy);

    double prob = exp(-(d_energy / temp));
    if (prob_r >= prob) {
        return 0;
    }

    remove_particle(part);
    box.particles[p_idx] = p;
    box.update_particle_async(p_idx);
    add_particle_async(p);
    cudaDeviceSynchronize();
    return d_energy;
}

double cell_view_t::particle_energy_patch(particle_t const &p,
                                          double const cosmax,
                                          double const epsilon) {
    const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
    double result = 0.0F;
    double const dist = 2.238F * p.radius;
    // check cell with particle, also
    // neighboring cells by combining
    // different combinations of -1, 0, 1
    // for each axis
    double3 coeff_val = {
            .x = ceil(dist / cell_size.x),
            .y = ceil(dist / cell_size.y),
            .z = ceil(dist / cell_size.z),
    };
    for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
        double x = p.pos.x + coeff_x * cell_size.x;
        if (x < 0) {
            x += box.dimensions.x;
        }
        if (x >= box.dimensions.x) {
            x -= box.dimensions.x;
        }
        for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
            double y = p.pos.y + coeff_y * cell_size.y;
            if (y < 0) {
                y += box.dimensions.y;
            }
            if (y >= box.dimensions.y) {
                y -= box.dimensions.y;
            }
            for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
                 coeff_z += 1) {
                double z = p.pos.z + coeff_z * cell_size.z;
                if (z < 0) {
                    z += box.dimensions.z;
                }
                if (z >= box.dimensions.z) {
                    z -= box.dimensions.z;
                }

                const size_t cell_idx = get_cell_idx((double3){x, y, z});
                if (cell_idx >= cell_cnt) {
                    continue;
                }

                cell_t const &cell = cells[cell_idx];
                for (size_t i = 0; i < cell.num_particles; i++) {
                    particle_t const &part = box.particles[cell.particle_indices[i]];
                    if (part.idx == p.idx) {
                        continue;
                    }
                    if (distance(part.pos, p.pos) >= dist) {
                        continue;
                    }

                    const double res = part.interact(p, cosmax, epsilon);
                    result += res;
                }
            }
        }
    }
    return result;
}
