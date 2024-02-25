#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"

#include <iostream>
#include <random>
#include <stdio.h>

// __global__ void print_particle(particle_box_t box) {
//   int id = threadIdx.x;
//   double3 const v = box.particles[id].pos;
//   printf("(%lf, %lf, %lf)\n", v.x, v.y, v.z);
//   // cudaFree(box.particles[id].patches);
// }

int main() {
  particle_box_t box{};
  box.dimensions = {10, 10, 10};
  particle_box_init_host(box, 512);

  // particle_t p1{.radius = 1.0F};
  // particle_t p2{.radius = 1.0F};

  std::random_device r;
  std::mt19937 re(r());

  std::uniform_real_distribution<double> unif_x(0, box.dimensions.x);
  std::uniform_real_distribution<double> unif_y(0, box.dimensions.y);
  std::uniform_real_distribution<double> unif_z(0, box.dimensions.z);

  for (size_t i = 0; i < 512; i++) {
    particle_box_add_particle_host(box, 0.5, unif_x, unif_y, unif_z, re);
  }

  export_particles_to_pdb(box, "stochastic.pdb");
  particle_box_free_particles_host(box);

  particle_box_t b = make_box_uniform_particles_host({10, 10, 10}, 0.5, 8);
  export_particles_to_pdb(b, "uniform.pdb");
  particle_box_free_particles_host(b);

  // for (size_t i = 0; i < box.particle_count; i++) {
  //   double3 const v = box.particles[i].pos;
  //   std::cout << '(' << v.x << ", " << v.y << ", " << v.z << ")," <<
  //   std::endl;
  // }

  // particle_box_t b = make_box(box);
  // cudaDeviceSynchronize();

  // print_particle<<<1, 512>>>(b);

  // cudaDeviceSynchronize();

  // particle_box_free_particles_host(box);
  // particle_box_free_particles_device(b);

  // cudaDeviceSynchronize();

  return 0;
}
