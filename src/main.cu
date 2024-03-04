#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

// __global__ void print_particle(particle_box_t box) {
//   int id = threadIdx.x;
//   double3 const v = box.particles[id].pos;
//   printf("(%lf, %lf, %lf)\n", v.x, v.y, v.z);
//   // cudaFree(box.particles[id].patches);
// }

constexpr size_t ITERATIONS = 1'000'000;
constexpr size_t ITERATIONS_PER_EXPORT = 1'000;
constexpr double TEMPERATURE = 2.0F;
constexpr double BOLTZMANN_C = 1.380649e-23;

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

  std::uniform_real_distribution<double> unif_r(0, 0.999F);
  for (size_t iters = 0; iters <= ITERATIONS; iters++) {
    if (iters % ITERATIONS_PER_EXPORT == 0) {
      const size_t idx = iters / ITERATIONS_PER_EXPORT;
      char buf[16];
      std::sprintf(buf, "data/%06li.pdb", idx);
      export_particles_to_pdb(view.box, buf);
    }
    size_t const p_idx = unif_r(re) * view.box.particle_count;
    double const radius = view.box.particles[p_idx].radius;
    double3 const old_pos = view.box.particles[p_idx].pos;

    double old_energy = view.particle_energy_square_well(old_pos, radius);

    double3 const new_pos = view.try_random_particle_disp(p_idx, unif_r, re);
    if (new_pos.x == -1) {
      continue;
    }
    double new_energy = view.particle_energy_square_well(new_pos, radius);
    double prob = (old_energy - new_energy) / (BOLTZMANN_C * TEMPERATURE);
    if (unif_r(re) >= prob) {
      continue;
    }
  }

  view.free();

  return 0;
}
