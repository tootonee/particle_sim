#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <vector>

// __global__ void print_particle(particle_box_t box) {
//   int id = threadIdx.x;
//   double3 const v = box.particles[id].pos;
//   printf("(%lf, %lf, %lf)\n", v.x, v.y, v.z);
//   // cudaFree(box.particles[id].patches);
// }

/* constexpr size_t ITERATIONS = 100'000; */
constexpr size_t ITERATIONS = 100'000;
constexpr size_t ITERATIONS_PER_EXPORT = 10;
/* constexpr double TEMPERATURE = 275.15L; */
constexpr double TEMPERATURE = 25.15L;
/* constexpr double BOLTZMANN_C = 1.380649e-23L; */
constexpr double BOLTZMANN_C = 1.380649e-1L;

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

  std::vector<double> energies{};

  std::uniform_real_distribution<double> unif_r(0, 0.999L);
  for (size_t iters = 0; iters <= ITERATIONS; iters++) {
    if (iters % ITERATIONS_PER_EXPORT == 0) {
      const size_t idx = iters / ITERATIONS_PER_EXPORT;
      char buf[16];
      std::sprintf(buf, "data/%06li.pdb", idx);
      export_particles_to_pdb(view.box, buf);
      energies.push_back(view.total_energy());
      std::cout << "I = " << idx << std::endl;
    }
#pragma omp parallel for
    for (size_t i = 0; i < view.box.particle_count; i++) {
      size_t const p_idx = unif_r(re) * view.box.particle_count;
      double const radius = view.box.particles[p_idx].radius;
      double3 const old_pos = view.box.particles[p_idx].pos;
      // std::cout << "Particle pos = <" << old_pos.x << ", " << old_pos.y << ",
      // "
      //           << old_pos.z << ">" << std::endl;

      double old_energy =
          view.particle_energy_square_well(old_pos, radius, 2.5);

      double3 const new_pos = view.try_random_particle_disp(p_idx, unif_r, re);
      if (new_pos.x == -1) {
        continue;
      }
      double new_energy =
          view.particle_energy_square_well(new_pos, radius, 2.5);

      double prob = (new_energy - old_energy) / (BOLTZMANN_C * TEMPERATURE);
      // std::cout << "Old energy = " << old_energy
      //           << ", New energy = " << new_energy << ", Prob = " << prob
      //           << std::endl;
      if (unif_r(re) >= prob) {
        continue;
      }
      view.remove_particle(view.box.particles[p_idx]);
      view.box.particles[p_idx].pos = new_pos;
      view.add_particle(view.box.particles[p_idx]);
    }
  }

  std::vector<double> distribution{};
  double3 pos = {
      view.box.dimensions.x / 2,
      view.box.dimensions.y / 2,
      view.box.dimensions.z / 2,
  };
  double radius = 0.1L;

  while (radius <= 10) {
    const size_t p_idx = unif_r(re) * view.box.particle_count;
    const double num =
        view.particles_in_range(view.box.particles[p_idx].pos, radius);
    const double res = num / (M_PI * 1.072330292 * radius * radius * 0.1L);
    distribution.push_back(res);
    radius += 0.1;
  }

  view.free();

  std::ofstream file("output.txt");
  file << std::fixed << std::setprecision(3);
  for (const double energy : energies) {
    file << energy << std::endl;
  }

  std::ofstream file_distr("distr_output.txt");
  file_distr << std::fixed << std::setprecision(3);
  for (const double num_p : distribution) {
    file_distr << num_p << std::endl;
  }

  return 0;
}
