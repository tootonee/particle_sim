#include "cell_view.h"
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
  std::random_device r;
  std::mt19937 re(r());

  std::uniform_real_distribution<double> unif_x(0, 10);
  std::uniform_real_distribution<double> unif_y(0, 10);
  std::uniform_real_distribution<double> unif_z(0, 10);
  cell_view_t view({10, 10, 10}, 8);
  std::cout << "Box particles = " << view.box.particles << std::endl;

  for (size_t i = 0; i < 512; i++) {
    std::cout << "I = " << i << std::endl;
    view.add_particle_random_pos(0.5, unif_x, unif_y, unif_z, re);
  }

  export_particles_to_pdb(view.box, "stochastic.pdb");
  view.free();

  return 0;
}
