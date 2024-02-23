#include <cstdio>
#include <iostream>

#include "particle.h"
#include "particle_box.h"

int main() {
  particle_box_t box{};
  box.dimensions = {10, 50, 10};
  particle_box_init_host(box, 512);

  // particle_t p1{.radius = 1.0F};
  // particle_t p2{.radius = 1.0F};

  for (size_t i = 0; i < 512; i++) {
    std::cout << "I = " << i << std::endl;
    particle_box_add_particle_host(box, 1);
  }

  // for (size_t i = 0; i < 100; i++) {
  //   particle_box_add_particle_host(box, 1);
  // }

  for (size_t i = 0; i < box.particle_count; i++) {
    double3 const &v = box.particles[i].pos;
    std::cout << '(' << v.x << ", " << v.y << ", " << v.z << ")," << std::endl;
  }

  particle_box_free_particles_host(box);

  return 0;
}
