#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "vec.h"
#include <cmath>

void cell_view_t::alloc_cells() {
  size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;

  cells = new cell_t[cell_count];

  energies = new double[box.capacity / 2];
  cudaMalloc(&energies_device, sizeof(double) * box.capacity / 2);
  // cudaMemset(energies_device, 0.0F, sizeof(double) * MAX_PARTICLES_PER_CELL);
}

void cell_view_t::free_cells() { delete[] cells; }

void cell_view_t::free() {
  box.free_particles();
  free_cells();
  delete[] energies;
  cudaFree(energies_device);
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

  box.particles[particle_idx].pos = disp;

  if (particle_intersects(box.particles[particle_idx])) {
    box.particles[particle_idx].pos = old_pos;
    return {-1, -1, -1};
  }

  box.particles[particle_idx].pos = old_pos;
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

void cell_view_t::add_particle_random_pos(double radius, rng_gen &rng_x,
                                          rng_gen &rng_y, rng_gen &rng_z,
                                          std::mt19937 &re) {
  if (box.capacity <= box.particle_count) {
    box.realloc(box.capacity * 2);
    delete[] energies;
    cudaFree(energies_device);

    energies = new double[box.capacity / 2];
    cudaMalloc(&energies_device, sizeof(double) * box.capacity / 2);
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
    total +=
        particle_energy_square_well_device(box.particles[p_idx], sigma, val);
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

__global__ void energy_square_well(particle_t const p,
                                   particle_t const *particles, size_t p_cnt,
                                   double *energies_dev, double cosmax,
                                   double sigma, double val) {
  const unsigned thread_idx = (threadIdx.x + blockDim.x * blockIdx.x) << 1;

  if (thread_idx >= p_cnt) {
    return;
  }

  double result = 0.0F;
  do_energy_calc(p.idx, particles, p_cnt, cosmax, sigma, val, result, 0,
                 thread_idx, p);
  do_energy_calc(p.idx, particles, p_cnt, cosmax, sigma, val, result, 1,
                 thread_idx, p);

  energies_dev[thread_idx] = result;
}

__global__ void parallel_sum(double *energies_dev, size_t iter) {
  size_t offset = 1 << (iter);
  const unsigned thread_idx = (threadIdx.x + blockDim.x * blockIdx.x)
                              << (iter + 1);
  double d1 = energies_dev[thread_idx];
  double d2 = energies_dev[thread_idx + offset];
  energies_dev[thread_idx] = d1 + d2;
}

double cell_view_t::particle_energy_square_well_device(particle_t const p,
                                                       double const sigma,
                                                       double const val) {
  double result = 0.0F;
  unsigned blocks = 1;
  unsigned threads = box.particle_count >> 2;
  if (box.particle_count > 1024) {
    threads = 512;
    blocks = box.particle_count >> 10;
  }
  energy_square_well<<<blocks, threads>>>(p, box.particles_device,
                                          box.particle_count, energies_device,
                                          0.89, sigma, val);
  size_t energy_cnt = box.particle_count >> 1;
  parallel_sum<<<1, (energy_cnt >> 1)>>>(energies_device, 0);
  parallel_sum<<<1, (energy_cnt >> 1)>>>(energies_device, 1);
  cudaMemcpy(energies, energies_device, sizeof(double) * energy_cnt,
             cudaMemcpyDeviceToHost);

#pragma omp parallel for
  for (size_t i = 0; i < energy_cnt; i += 4) {
    result += energies[i];
  }
  return result;
}

double cell_view_t::try_move_particle(size_t const p_idx, double3 const new_pos,
                                      double4 const rotation, double prob_r,
                                      double temp) {
  if (new_pos.x == -1) {
    return 0;
  }

  double3 const old_pos = box.particles[p_idx].pos;
  particle_t &part = box.particles[p_idx];

  double old_energy = particle_energy_square_well(part, 0.2, -1) +
                      particle_energy_patch(part, 0.89, -2);

  part.pos = new_pos;
  part.rotate(rotation);
  double new_energy = particle_energy_square_well(part, 0.2, -1) +
                      particle_energy_patch(part, 0.89, -2);
  part.pos = old_pos;
  double const d_energy = (new_energy - old_energy);
  double prob = exp(-(d_energy / temp));
  if (prob_r >= prob) {
    part.rotate({
        .x = rotation.x,
        .y = -rotation.y,
        .z = -rotation.z,
        .w = -rotation.w,
    });
    return 0;
  }
  remove_particle(box.particles[p_idx]);
  part.pos = new_pos;
  add_particle(box.particles[p_idx]);
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

  double old_energy = particle_energy_square_well_device(p, 0.2, -1);

  p.pos = new_pos;
  p.rotate(rotation);
  double new_energy = particle_energy_square_well_device(p, 0.2, -1);
  double const d_energy = (new_energy - old_energy);

  double prob = exp(-(d_energy / temp));
  if (prob_r >= prob) {
    return 0;
  }

  remove_particle(part);
  box.particles[p_idx] = p;
  box.update_particle(p_idx);
  add_particle(p);
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
