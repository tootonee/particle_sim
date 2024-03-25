#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "vec.h"
#include <cmath>

void cell_view_t::alloc_cells() {
  size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
  cells = new cell_t[cell_count];
  cudaMalloc(&cells_device, sizeof(cell_t) * cell_count);
  // cudaMallocManaged(&cells, sizeof(cell_t) * cell_count, cudaMemAttachGlobal);
}

void cell_view_t::free_cells() { 
    delete[] cells;
    cudaFree(cells_device);
}

void cell_view_t::free() {
  free_cells();
  box.free_particles();
}

void cell_view_t::update_cell(size_t const cell_idx) {
  const size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
  if (cell_idx >= cell_count) {
      return;
  }

  cudaMemcpy(cells_device + cell_idx, cells + cell_idx, sizeof(cell_t), cudaMemcpyHostToDevice);
}

bool cell_view_t::add_particle(particle_t const &p) {
  size_t cell_idx = get_cell_idx(p);
  if (cell_idx >= cells_per_axis * cells_per_axis * cells_per_axis) {
    return false;
  }
  cell_t &cell = cells[cell_idx];
  if (cell.num_particles >= MAX_PARTICLES_PER_CELL) {
    return false;
  }
  cell.particle_indices[cell.num_particles] = p.idx;
  cell.num_particles += 1;
  update_cell(cell_idx);
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
                                              rng_gen &rng, std::mt19937 &re,
                                              double const scale) {
  if (particle_idx >= box.particle_count) {
    return {-1, -1, -1};
  }

  particle_t const &p_orig = box.particles[particle_idx];
  const double3 old_pos = p_orig.pos;

  double3 disp = {
      scale * (rng(re) - 0.5),
      scale * (rng(re) - 0.5),
      scale * (rng(re) - 0.5),
  };

  disp = {
      std::abs(p_orig.pos.x + disp.x),
      std::abs(p_orig.pos.y + disp.y),
      std::abs(p_orig.pos.z + disp.z),
  };

  if (disp.x >= box.dimensions.x) {
    disp.x -= p_orig.pos.x;
  }

  if (disp.y >= box.dimensions.y) {
    disp.y -= p_orig.pos.y;
  }

  if (disp.z >= box.dimensions.z) {
    disp.z -= p_orig.pos.z;
  }

  box.particles[particle_idx].pos = disp;

  if (particle_intersects(box.particles[particle_idx])) {
    box.particles[particle_idx].pos = old_pos;
    return {-1, -1, -1};
  }

  box.particles[particle_idx].pos = old_pos;
  return disp;
}

void cell_view_t::remove_particle(particle_t const &p) {
  size_t cell_idx = get_cell_idx(p);
  cell_t &cell = cells[cell_idx];
  if (cell.num_particles == 0) {
    return;
  }

  for (size_t i = 0; i < cell.num_particles; i++) {
    if (cell.particle_indices[i] == p.idx) {
      cell.particle_indices[i] = cell.particle_indices[cell.num_particles - 1];
      cell.num_particles--;
      update_cell(cell_idx);
      return;
    }
  }
}

void
cell_view_t::remove_particle_from_box(particle_t const &p) {
  size_t cell_idx = get_cell_idx(p);
  cell_t &cell = cells[cell_idx];
  if (cell.num_particles == 0) {
    return;
  }

  for (size_t i = 0; i < cell.num_particles; i++) {
    if (cell.particle_indices[i] == p.idx) {
      cell.particle_indices[i] = cell.particle_indices[cell.num_particles - 1];
      cell.num_particles--;
      update_cell(cell_idx);
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

double
cell_view_t::particle_energy_square_well(particle_t const &p,
                                         double const sigma, double const val) {
  const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
  double result = 0.0F;
  // check cell with particle, also neighboring cells by combining different
  // combinations of -1, 0, 1 for each axis
  double3 coeff_val = {
      .x = 2 * ceil(sigma / cell_size.x) + 1,
      .y = 2 * ceil(sigma / cell_size.y) + 1,
      .z = 2 * ceil(sigma / cell_size.z) + 1,
  };
  for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
    double x = p.pos.x + coeff_x * cell_size.x;
    if (x < 0) {
      x = box.dimensions.x - cell_size.x;
    }
    if (x >= box.dimensions.x) {
      x = 0;
    }
    for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
      double y = p.pos.y + coeff_y * cell_size.y;
      if (y < 0) {
        y = box.dimensions.y - cell_size.y;
      }
      if (y >= box.dimensions.y) {
        y = 0;
      }
      for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
           coeff_z += 1) {
        double z = p.pos.z + coeff_z * cell_size.z;
        if (z < 0) {
          z = box.dimensions.z - cell_size.z;
        }
        if (z >= box.dimensions.z) {
          z = 0;
        }

        const size_t cell_idx = get_cell_idx((double3){x, y, z});
        if (cell_idx >= cell_cnt) {
          continue;
        }

        cell_t const &cell = cells[cell_idx];
        for (size_t i = 0; i < cell.num_particles; i++) {
          if (distance(p.pos, box.particles[cell.particle_indices[i]].pos) <=
              sigma) {
            result += val;
          }
        }
      }
    }
  }
  return result;

//   double result = 0.0L;
//
// #pragma omp parallel for
//   for (size_t idx = 0; idx < box.particle_count; idx++) {
//       if (idx = p.idx) {continue;}
//       const particle_t &part = box.particles[idx];
//
//       if (distance(p.pos, part.pos) <= sigma) {
//          result -= val;
//       }
//   }
//
//   return result;
}


double cell_view_t::total_energy() {
  double total = 0.0F;
  for (size_t p_idx = 0; p_idx <= box.particle_count; p_idx++) {
    total += particle_energy_square_well(box.particles[p_idx]);
  }
  return total / 2.0F;
}

inline constexpr void check_scale(double &min_scale,
                                                      int &cur_dir,
                                                      int const &dir,
                                                      double const &val) {
  if (min_scale < val) {
    min_scale = val;
    cur_dir = dir;
  }
}

double
cell_view_t::particles_in_range(const size_t idx, const double r1,
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

__global__ void energy_square_well(cell_view_t const view, particle_t const &p,
                                   int3 strides,
                                   double3 coeff_vals, double const sigma,
                                   double const val) {
  // const size_t cell_count = view.cells_per_axis * view.cells_per_axis * view.cells_per_axis;
  int thread_idx = threadIdx.x;
  double dx = (thread_idx % strides.x) - coeff_vals.x;
  double dy = (thread_idx % strides.y) - coeff_vals.y;
  double dz = (thread_idx % strides.z) - coeff_vals.z;

  int idx_x = (int)((p.pos.x + dx * view.cell_size.x) / view.cell_size.x);
  idx_x = idx_x % view.cells_per_axis;
  int idx_y = (int)((p.pos.y + dy * view.cell_size.y) / view.cell_size.y);
  idx_y = idx_y % view.cells_per_axis;
  int idx_z = (int)((p.pos.z + dz * view.cell_size.z) / view.cell_size.z);
  idx_z = idx_z % view.cells_per_axis;

  size_t cell_idx = idx_x * view.cells_per_axis * view.cells_per_axis +
                    idx_y * view.cells_per_axis + idx_z;
  cell_t &cell = view.cells[cell_idx];

  double result = 0;
  for (size_t i = 0; i < cell.num_particles; i++) {
    if (distance(p.pos, view.box.particles[cell.particle_indices[i]].pos) <=
        sigma) {
      result += val;
    }
  }

  /* printf("Idx = %i, result = %d\n", thread_idx, result); */
  cell.energy = result;
}

double cell_view_t::particle_energy_square_well_device(particle_t const &p,
                                                       double const sigma,
                                                       double const val) {
  const double dist = 2 * p.radius + sigma;
  double3 coeff_val = {
      .x = ceil(dist / cell_size.x),
      .y = ceil(dist / cell_size.y),
      .z = ceil(dist / cell_size.z),
  };

  int stride_x = 2 * coeff_val.x;
  int stride_y = 4 * coeff_val.x * coeff_val.y;
  int stride_z = 8 * coeff_val.x * coeff_val.y * coeff_val.z;

  int3 strides = {stride_x, stride_y, stride_z};

  const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
  const size_t thread_count = coeff_val.x * coeff_val.y * coeff_val.z * 8;
  energy_square_well<<<1, thread_count>>>(*this, p, strides,
                                          coeff_val, sigma, val);
  cudaMemPrefetchAsync (box.particles, sizeof(particle_t) * box.particle_count, 0);
  cudaMemPrefetchAsync (cells, sizeof(cell_t) * cell_cnt, 0);
  cudaDeviceSynchronize();

  double result = 0;
  for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
    double x = p.pos.x + coeff_x * cell_size.x;
    if (x < 0) {
      x = box.dimensions.x - cell_size.x;
    }
    if (x >= box.dimensions.x) {
      x = 0;
    }
    for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
      double y = p.pos.y + coeff_y * cell_size.y;
      if (y < 0) {
        y = box.dimensions.y - cell_size.y;
      }
      if (y >= box.dimensions.y) {
        y = 0;
      }
      for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
           coeff_z += 1) {
        double z = p.pos.z + coeff_z * cell_size.z;
        if (z < 0) {
          z = box.dimensions.z - cell_size.z;
        }
        if (z >= box.dimensions.z) {
          z = 0;
        }
        const size_t cell_idx = get_cell_idx((double3){x, y, z});
        if (cell_idx >= cell_cnt) {
          continue;
        }
        cell_t const &cell = cells[cell_idx];
        result += cell.energy;
      }
    }
  }
  return result;
}
