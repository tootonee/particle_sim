#include "particle_box.h"

void particle_box_init_host(particle_box_t &p, size_t capacity) {
  p.particles = new particle_t[capacity];
  p.capacity = capacity;
}

__device__ void particle_box_init_device(particle_box_t p, size_t capacity) {
  p.capacity = capacity;
  cudaMalloc(&p.particles, sizeof(particle_t) * capacity);
}

void particle_box_realloc_host(particle_box_t &p, size_t capacity) {
  if (capacity <= p.capacity) {
    return;
  }
  particle_t *new_particles = new particle_t[capacity];
  p.capacity = capacity;
  for (size_t idx = 0; idx < p.particle_count; idx++) {
    new_particles[idx] = p.particles[idx];
  }
  delete[] p.particles;
  p.particles = new_particles;
}

__device__ void particle_box_realloc_device(particle_box_t p, size_t capacity) {
  if (capacity <= p.capacity) {
    return;
  }
  particle_t *new_particles;
  cudaMalloc(&new_particles, sizeof(particle_t) * capacity);
  for (size_t idx = 0; idx < p.particle_count; idx++) {
    new_particles[idx] = p.particles[idx];
  }
  cudaFree(p.particles);
  p.particles = new_particles;
}

void particle_box_add_particle_host(particle_box_t &box, double radius,
                                    rng_gen &rng_x, rng_gen &rng_y,
                                    rng_gen &rng_z, std::mt19937 &re) {
  if (box.capacity >= box.particle_count) {
    particle_box_realloc_host(box, box.particle_count * 2);
  }

  bool intersects = true;
  particle_t *p = box.particles + box.particle_count;
  p->radius = radius;
  p->valid = true;
  p->idx = box.particle_count;
  particle_init_host(box.particles[box.particle_count]);
  do {
    intersects = false;
    random_particle_pos(*p, rng_x, rng_y, rng_z, re);

    for (size_t i = 0; i < box.particle_count; i++) {
      if (particle_intersects(box.particles[i], *p)) {
        intersects = true;
        break;
      }
    }
  } while (intersects);
  box.particle_count++;
}

void particle_box_add_particle_host(particle_box_t &box, double radius) {
  if (box.capacity >= box.particle_count) {
    particle_box_realloc_host(box, box.particle_count * 2);
  }

  bool intersects = true;
  particle_t *p = box.particles + box.particle_count;
  p->radius = radius;
  p->valid = true;
  p->idx = box.particle_count;
  particle_init_host(box.particles[box.particle_count]);
  do {
    intersects = false;
    random_particle_pos(*p, box.dimensions);

    for (size_t i = 0; i < box.particle_count; i++) {
      if (particle_intersects(box.particles[i], *p)) {
        intersects = true;
        break;
      }
    }
  } while (intersects);
  box.particle_count++;
}

void particle_box_add_particle_host(particle_box_t &box, particle_t const &p) {
  if (box.capacity >= box.particle_count) {
    particle_box_realloc_host(box, box.particle_count * 2);
  }
  box.particles[box.particle_count] = p;
  box.particles[box.particle_count].idx = box.particle_count;
  box.particle_count++;
}

__device__ void particle_box_add_particle_device(particle_box_t box,
                                                 particle_t const &p) {
  if (box.capacity >= box.particle_count) {
    particle_box_realloc_device(box, box.particle_count * 2);
  }
  box.particles[box.particle_count] = p;
  box.particles[box.particle_count].idx = box.particle_count;
  box.particle_count++;
}

__global__ void assign_patch(particle_t *p, patch_t *patches) {
  p->patches = patches;
}

particle_box_t make_box(particle_box_t const &box) {
  particle_box_t res{};

  res.dimensions = box.dimensions;
  res.particle_count = box.particle_count;
  res.capacity = box.capacity;

  cudaMalloc(&res.particles, sizeof(particle_t) * box.capacity);
  cudaMemcpy(res.particles, box.particles,
             sizeof(particle_t) * box.particle_count, cudaMemcpyHostToDevice);
  for (size_t i = 0; i < box.particle_count; i++) {
    patch_t *tmp;
    cudaMalloc(&tmp, sizeof(patch_t) * box.particles[i].patch_count);
    cudaMemcpy(tmp, box.particles[i].patches,
               sizeof(patch_t) * box.particles[i].patch_count,
               cudaMemcpyHostToDevice);
    assign_patch<<<1, 1>>>(res.particles + i, tmp);
  }

  return res;
}

__host__ __device__ void particle_box_remove_particle(particle_box_t &p,
                                                      size_t idx) {
  if (idx >= p.particle_count) {
    return;
  }
  p.particles[idx] = p.particles[p.particle_count - 1];
  p.particle_count -= 1;
}

__host__ __device__ void particle_box_swap_particles(particle_box_t &p,
                                                     size_t const fst,
                                                     size_t const snd) {
  if (fst >= p.particle_count || snd >= p.particle_count) {
    return;
  }
  particle_t tmp = p.particles[fst];
  p.particles[fst] = p.particles[snd];
  p.particles[fst].idx = fst;
  p.particles[snd] = tmp;
  p.particles[snd].idx = snd;
}

__global__ void freePatches(particle_t *p) { cudaFree(p->patches); }

__host__ __device__ void particle_box_free_particles_device(particle_box_t &p) {
  for (size_t idx = 0; idx < p.particle_count; idx++) {
    freePatches<<<1, 1>>>(p.particles + idx);
  }
  cudaFree(p.particles);
  p.particle_count = 0;
  p.capacity = 0;
}

particle_box_t make_box_uniform_particles_host(double3 const dimensions,
                                               double const radius,
                                               size_t const count_per_axis) {
  particle_box_t res{};
  res.dimensions = dimensions;
  res.capacity = count_per_axis * count_per_axis * count_per_axis;
  particle_box_init_host(res, res.capacity);

  double3 axis_steps = {
      .x = dimensions.x / count_per_axis,
      .y = dimensions.y / count_per_axis,
      .z = dimensions.z / count_per_axis,
  };

  for (size_t x = 0; x < count_per_axis; x++) {
    for (size_t y = 0; y < count_per_axis; y++) {
      for (size_t z = 0; z < count_per_axis; z++) {
        particle_t p{};
        p.radius = radius;
        p.valid = true;
        p.pos.x = x * axis_steps.x + radius;
        p.pos.y = y * axis_steps.y + radius;
        p.pos.z = z * axis_steps.z + radius;
        p.idx = res.particle_count;
        particle_init_host(p);

        res.particles[x * count_per_axis * count_per_axis + y * count_per_axis +
                      z] = p;
        res.particle_count += 1;
      }
    }
  }
  return res;
}
