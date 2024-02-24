#include "particle.h"
#include "vec.h"

#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <random>

__host__ __device__ bool particle_intersects(particle_t const &lhs,
                                             particle_t const &rhs) {
  double const diameter = lhs.radius + rhs.radius;
  return (rhs.valid) && (lhs.valid) && (distance(lhs.pos, rhs.pos) < diameter);
}

void particle_init_host(particle_t &p, size_t capacity) {
  delete[] p.patches;
  p.patch_count = capacity;
  p.patches = new patch_t[capacity];
}

__device__ void particle_init_device(particle_t &p, size_t capacity) {
  cudaFree(p.patches);
  p.patch_count = capacity;
  cudaMalloc(&p.patches, capacity * sizeof(patch_t));
}

__host__ void random_particle_pos(particle_t &p, double3 dimensions) {
  std::random_device r;
  std::mt19937 re(r());

  std::uniform_real_distribution<double> unif_x(0, dimensions.x);
  std::uniform_real_distribution<double> unif_y(0, dimensions.y);
  std::uniform_real_distribution<double> unif_z(0, dimensions.z);

  p.pos.x = unif_x(re);
  p.pos.y = unif_y(re);
  p.pos.z = unif_z(re);
};

__host__ void random_particle_pos(particle_t &p, rng_gen &rng_x, rng_gen &rng_y,
                                  rng_gen &rng_z, std::mt19937 &re) {
  p.pos.x = rng_x(re);
  p.pos.y = rng_y(re);
  p.pos.z = rng_z(re);
}
