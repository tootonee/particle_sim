#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "vec.h"
#include <cmath>

void cell_view_t::alloc_cells() {
  size_t cell_count = cells_per_axis * cells_per_axis * cells_per_axis;
  cudaMallocManaged(&cells, sizeof(cell_t) * cell_count);
}

void cell_view_t::free_cells() { cudaFree(cells); }

void cell_view_t::free() {
  free_cells();
  box.free_particles();
}

__host__ __device__ bool cell_view_t::add_particle(particle_t const &p) {
  size_t cell_idx = get_cell_idx(p);
  if (cell_idx >= cells_per_axis * cells_per_axis * cells_per_axis) {
    return false;
  }
  cell_t &cell = cells[cell_idx];
  if (cell.num_particles >= MAX_PARTICLES_PER_CELL) {
    return false;
  }
  cell.particle_indices[cell.num_particles++] = p.idx;
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
  double radius = p_orig.radius;
  double3 disp = {
      scale * cell_size.x * (rng(re) - 0.5),
      scale * cell_size.y * (rng(re) - 0.5),
      scale * cell_size.z * (rng(re) - 0.5),
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

  if (particle_intersects(disp, radius)) {
    return {-1, -1, -1};
  }
  return disp;
}

__host__ __device__ void cell_view_t::remove_particle(particle_t const &p) {
  size_t cell_idx = get_cell_idx(p);
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

__host__ __device__ void
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
      return;
    }
  }
  if (p.idx <= box.particle_count && box.particles[p.idx] == p) {
    box.remove_particle(p.idx);
  }
}

__host__ __device__ bool cell_view_t::particle_intersects(particle_t const &p) {
  const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
  // check cell with particle, also neighboring cells by combining different
  // combinations of -1, 0, 1 for each axis
  double3 coeff_val = {
      .x = floor(p.radius / cell_size.x) + 1.0F,
      .y = floor(p.radius / cell_size.y) + 1.0F,
      .z = floor(p.radius / cell_size.z) + 1.0F,
  };
  for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
    double x = p.pos.x + coeff_x * cell_size.x;
    if (x < 0) {
      x = box.dimensions.x - cell_size.x;
    }
    if (x > box.dimensions.x) {
      x = 0;
    }
    size_t idx_x = (size_t)(x / cell_size.x);
    for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
      double y = p.pos.y + coeff_y * cell_size.y;
      if (y < 0) {
        y = box.dimensions.y - cell_size.y;
      }
      if (y > box.dimensions.y) {
        y = 0;
      }
      size_t idx_y = (size_t)(y / cell_size.y);
      for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
           coeff_z += 1) {
        double z = p.pos.z + coeff_z * cell_size.z;
        if (z < 0) {
          z = box.dimensions.z - cell_size.z;
        }
        if (z > box.dimensions.z) {
          z = 0;
        }
        size_t idx_z = (size_t)(z / cell_size.z);
        size_t cell_idx = idx_x * cells_per_axis * cells_per_axis +
                          idx_y * cells_per_axis + idx_z;
        if (cell_idx >= cell_cnt) {
          continue;
        }
        cell_t const &cell = cells[cell_idx];
        for (size_t i = 0; i < cell.num_particles; i++) {
          if (p.intersects(box.particles[cell.particle_indices[i]])) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

__host__ __device__ bool cell_view_t::particle_intersects(double3 const pos,
                                                          double const radius) {

  const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
  // check cell with particle, also neighboring cells by combining different
  // combinations of -1, 0, 1 for each axis
  double3 coeff_val = {
      .x = floor(radius / cell_size.x) + 1.0F,
      .y = floor(radius / cell_size.y) + 1.0F,
      .z = floor(radius / cell_size.z) + 1.0F,
  };
  for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
    double x = pos.x + coeff_x * cell_size.x;
    if (x < 0) {
      x = box.dimensions.x - cell_size.x;
    }
    if (x > box.dimensions.x) {
      x = 0;
    }
    size_t idx_x = (size_t)(x / cell_size.x);
    for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
      double y = pos.y + coeff_y * cell_size.y;
      if (y < 0) {
        y = box.dimensions.y - cell_size.y;
      }
      if (y > box.dimensions.y) {
        y = 0;
      }
      size_t idx_y = (size_t)(y / cell_size.y);
      for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
           coeff_z += 1) {
        double z = pos.z + coeff_z * cell_size.z;
        if (z < 0) {
          z = box.dimensions.z - cell_size.z;
        }
        if (z > box.dimensions.z) {
          z = 0;
        }
        size_t idx_z = (size_t)(z / cell_size.z);
        size_t cell_idx = idx_x * cells_per_axis * cells_per_axis +
                          idx_y * cells_per_axis + idx_z;
        if (cell_idx >= cell_cnt) {
          continue;
        }
        cell_t const &cell = cells[cell_idx];
        for (size_t i = 0; i < cell.num_particles; i++) {
          if (box.particles[cell.particle_indices[i]].intersects(pos, radius)) {
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
  bool intersects = true;
  particle_t *p = box.particles + box.particle_count;
  p->radius = radius;
  p->idx = box.particle_count;
  p->init();
  do {
    p->random_particle_pos(rng_x, rng_y, rng_z, re);
    intersects = particle_intersects(*p);
  } while (intersects);
  box.particle_count++;
}

__host__ __device__ bool
cell_view_t::particle_energy_square_well(particle_t const &p,
                                         double const sigma, double const val) {
  const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
  double result = 0.0F;
  // check cell with particle, also neighboring cells by combining different
  // combinations of -1, 0, 1 for each axis
  double3 coeff_val = {
      .x = floor(p.radius / cell_size.x) + 1.0F,
      .y = floor(p.radius / cell_size.y) + 1.0F,
      .z = floor(p.radius / cell_size.z) + 1.0F,
  };
  for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
    double x = p.pos.x + coeff_x * cell_size.x;
    if (x < 0) {
      x = box.dimensions.x - cell_size.x;
    }
    if (x > box.dimensions.x) {
      x = 0;
    }
    size_t idx_x = (size_t)(x / cell_size.x);
    for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
      double y = p.pos.y + coeff_y * cell_size.y;
      if (y < 0) {
        y = box.dimensions.y - cell_size.y;
      }
      if (y > box.dimensions.y) {
        y = 0;
      }
      size_t idx_y = (size_t)(y / cell_size.y);
      for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
           coeff_z += 1) {
        double z = p.pos.z + coeff_z * cell_size.z;
        if (z < 0) {
          z = box.dimensions.z - cell_size.z;
        }
        if (z > box.dimensions.z) {
          z = 0;
        }
        size_t idx_z = (size_t)(z / cell_size.z);
        size_t cell_idx = idx_x * cells_per_axis * cells_per_axis +
                          idx_y * cells_per_axis + idx_z;
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
}

__host__ __device__ bool
cell_view_t::particle_energy_square_well(double3 const pos, double const radius,
                                         double const sigma, double const val) {
  const size_t cell_cnt = cells_per_axis * cells_per_axis * cells_per_axis;
  double result = 0.0F;
  // check cell with particle, also neighboring cells by combining different
  // combinations of -1, 0, 1 for each axis
  double3 coeff_val = {
      .x = floor(radius / cell_size.x) + 1.0F,
      .y = floor(radius / cell_size.y) + 1.0F,
      .z = floor(radius / cell_size.z) + 1.0F,
  };
  for (double coeff_x = -coeff_val.x; coeff_x <= coeff_val.x; coeff_x += 1) {
    double x = pos.x + coeff_x * cell_size.x;
    if (x < 0) {
      x = box.dimensions.x - cell_size.x;
    }
    if (x > box.dimensions.x) {
      x = 0;
    }
    size_t idx_x = (size_t)(x / cell_size.x);
    for (double coeff_y = -coeff_val.y; coeff_y <= coeff_val.y; coeff_y += 1) {
      double y = pos.y + coeff_y * cell_size.y;
      if (y < 0) {
        y = box.dimensions.y - cell_size.y;
      }
      if (y > box.dimensions.y) {
        y = 0;
      }
      size_t idx_y = (size_t)(y / cell_size.y);
      for (double coeff_z = -coeff_val.z; coeff_z <= coeff_val.z;
           coeff_z += 1) {
        double z = pos.z + coeff_z * cell_size.z;
        if (z < 0) {
          z = box.dimensions.z - cell_size.z;
        }
        if (z > box.dimensions.z) {
          z = 0;
        }
        size_t idx_z = (size_t)(z / cell_size.z);
        size_t cell_idx = idx_x * cells_per_axis * cells_per_axis +
                          idx_y * cells_per_axis + idx_z;
        if (cell_idx >= cell_cnt) {
          continue;
        }
        cell_t const &cell = cells[cell_idx];
        for (size_t i = 0; i < cell.num_particles; i++) {
          if (distance(pos, box.particles[cell.particle_indices[i]].pos) <=
              sigma) {
            result += val;
          }
        }
      }
    }
  }
  return result;
}
