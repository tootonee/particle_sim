#include "particle.h"
#include "particle_box.h"
#include <stdio.h>

__global__ void print_particle(particle_box_t box) {
  int id = threadIdx.x;
  double3 const v = box.particles[id].pos;
  printf("(%lf, %lf, %lf)\n", v.x, v.y, v.z);
  // cudaFree(box.particles[id].patches);
}

int main() {
  particle_box_t box{};
  box.dimensions = {10, 50, 10};
  particle_box_init_host(box, 512);

  // particle_t p1{.radius = 1.0F};
  // particle_t p2{.radius = 1.0F};

  for (size_t i = 0; i < 512; i++) {
    // std::cout << "I = " << i << std::endl;
    particle_box_add_particle_host(box, 1);
  }

  // for (size_t i = 0; i < 100; i++) {
  //   particle_box_add_particle_host(box, 1);
  // }

  // for (size_t i = 0; i < box.particle_count; i++) {
  //   double3 const v = b.particles[i].pos;
  //   std::cout << '(' << v.x << ", " << v.y << ", " << v.z << ")," <<
  //   std::endl;
  // }

  particle_box_t b = make_box(box);
  cudaDeviceSynchronize();

  print_particle<<<1, 512>>>(b);

  cudaDeviceSynchronize();

  particle_box_free_particles_host(box);
  particle_box_free_particles_device(b);

  cudaDeviceSynchronize();

  return 0;
}
