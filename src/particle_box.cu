#include "particle_box.h"

void particle_box_t::init(size_t cap) {
  free_particles();
  capacity = cap;
  particles = new particle_t[capacity];
  cudaMalloc(&particles_device, sizeof(particle_t) * capacity);
  // cudaMallocManaged(&particles, sizeof(particle_t) * capacity,
  // cudaMemAttachGlobal);
}

void particle_box_t::realloc(size_t cap) {
  if (cap <= capacity) {
    return;
  }
  particle_t *new_particles = new particle_t[cap];
  particle_t *new_particles_device = new particle_t[cap];
  std::memcpy(new_particles, particles, sizeof(particle_t) * capacity);
  cudaMalloc(&particles_device, sizeof(particle_t) * capacity);
  cudaMemcpy(new_particles_device, particles_device,
             sizeof(particle_t) * capacity, cudaMemcpyHostToHost);
  // cudaMallocManaged(&new_particles, sizeof(particle_t) * cap,
  // cudaMemAttachGlobal); cudaMemcpy(new_particles, particles,
  // sizeof(particle_t) * particle_count,
  //            cudaMemcpyDefault);
  capacity = cap;
  cudaFree(particles_device);
  delete[] particles;
  particles = new_particles;
  particles_device = new_particles_device;
}

void particle_box_t::add_particle(double radius, rng_gen &rng_x, rng_gen &rng_y,
                                  rng_gen &rng_z, std::mt19937 &re) {
  if (capacity <= particle_count) {
    realloc(capacity * 2);
  }
  bool intersects = true;
  particle_t *p = particles + particle_count;
  p->radius = radius;
  p->idx = particle_count;
  do {
    intersects = false;
    p->random_particle_pos(rng_x, rng_y, rng_z, re);

    for (size_t i = 0; i < particle_count; i++) {
      if (particles[i].intersects(*p)) {
        intersects = true;
        break;
      }
    }
  } while (intersects);
  cudaMemcpy(particles_device + particle_count, p, sizeof(particle_t),
             cudaMemcpyHostToDevice);
  particle_count++;
}

void particle_box_t::update_particle(size_t const p_idx) {
  if (particle_count <= particle_count) {
    return;
  }
  cudaMemcpy(particles_device + p_idx, particles + p_idx, sizeof(particle_t),
             cudaMemcpyHostToDevice);
}

void particle_box_t::add_particle(particle_t const &p) {
  if (capacity <= particle_count) {
    realloc(capacity * 2);
  }
  particles[particle_count] = p;
  particles[particle_count].idx = particle_count;
  cudaMemcpy(particles_device + particle_count, particles + particle_count,
             sizeof(particle_t), cudaMemcpyHostToDevice);
  particle_count++;
}

void particle_box_t::remove_particle(size_t idx) {
  if (idx >= particle_count) {
    return;
  }
  particles[idx] = particles[particle_count - 1];
  particle_count -= 1;
  update_particle(idx);
}

void particle_box_t::swap_particles(size_t const fst, size_t const snd) {
  if (fst >= particle_count || snd >= particle_count) {
    return;
  }
  particle_t tmp = particles[fst];
  particles[fst] = particles[snd];
  particles[fst].idx = fst;
  particles[snd] = tmp;
  particles[snd].idx = snd;
  update_particle(fst);
  update_particle(snd);
}

void particle_box_t::make_box_uniform_particles_host(
    double3 const dims, double const radius, size_t const count_per_axis) {
  free_particles();
  dimensions = dims;
  capacity = count_per_axis * count_per_axis * count_per_axis;
  init(capacity);

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
        p.pos.x = x * axis_steps.x + radius;
        p.pos.y = y * axis_steps.y + radius;
        p.pos.z = z * axis_steps.z + radius;
        p.idx = particle_count;

        particles[x * count_per_axis * count_per_axis + y * count_per_axis +
                  z] = p;
        particle_count += 1;
      }
    }
  }
}
