#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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

constexpr size_t PARTICLE_COUNT = 500;
/* constexpr size_t ITERATIONS = 10'000; */
constexpr size_t ITERATIONS = 10'000;
constexpr size_t ITERATIONS_PER_EXPORT = 100;
constexpr size_t ITERATIONS_PER_GRF_EXPORT = 100;
constexpr double TEMPERATURE = 1.5;
/* constexpr double TEMPERATURE = 2.15L; */
/* constexpr double BOLTZMANN_C = 1.380649e-23L; */
/* constexpr double BOLTZMANN_C = 1.380649e-1L; */
/* constexpr double BOLTZMANN_C = 1.0L; */

std::map<double, double> do_distr(cell_view_t const &view,
                                  double const rho = 0.5L,
                                  double const start = 1L,
                                  double const dr = 0.01L,
                                  double const max_r = 5L) {
  std::map<double, double> distr{};
  double radius = start - dr;
  double v_old = 0;
  double v_new = radius * radius * radius;

  while (radius < max_r) {
    double num = 0.0F;
    for (size_t p_idx = 0; p_idx < view.box.particle_count; p_idx++) {
      num += view.particles_in_range(p_idx, radius, radius + dr);
    }
    v_old = v_new;
    radius += dr;
    v_new = radius * radius * radius;
    double const val = 3 * num / (4 * M_PI * rho * (v_new - v_old));
    distr[radius] = val / view.box.particle_count;
  }
  return distr;
}

int main() {
  std::random_device r;
  std::mt19937 re(r());

  std::uniform_real_distribution<double> unif_x(0, 10);
  std::uniform_real_distribution<double> unif_y(0, 10);
  std::uniform_real_distribution<double> unif_z(0, 10);
  cell_view_t view({10, 10, 10}, 16);
  std::cout << "Box particles = " << view.box.particles << std::endl;

  std::vector<std::pair<size_t, size_t>> intersects{};

  for (size_t i = 0; i < PARTICLE_COUNT; i++) {
    std::cout << "I = " << i << std::endl;
    view.add_particle_random_pos(0.5, unif_x, unif_y, unif_z, re);
  }

  std::uniform_real_distribution<double> unif_r(0, 1);

  double const rho =
      view.box.particle_count /
      (view.box.dimensions.x * view.box.dimensions.y * view.box.dimensions.z);
  std::map<double, double> distr{};

  for (size_t iters = 0; iters <= ITERATIONS; iters++) {
    if (iters % ITERATIONS_PER_GRF_EXPORT == 0) {
      std::map<double, double> tmp_distr = do_distr(view, rho, 1, 0.05L);
      for (const auto &[radius, value] : tmp_distr) {
          distr[radius] += value;
      }
    }

    if (iters % ITERATIONS_PER_EXPORT == 0) {
      const size_t idx = iters / ITERATIONS_PER_EXPORT;
      char buf[16];
      std::sprintf(buf, "data/%06li.pdb", idx);
      export_particles_to_pdb(view.box, buf);

      std::cout << "I = " << idx << std::endl;
    }

    for (size_t i = 0; i < view.box.particle_count; i++) {
      size_t const p_idx = static_cast<size_t>(unif_r(re) * view.box.particle_count) % view.box.particle_count;
      double3 const old_pos = view.box.particles[p_idx].pos;
      particle_t &part = view.box.particles[p_idx];

      // double const old_energy = view.particle_energy_square_well(part, 0.2, 1);

      // double const old_energy =
      // view.particle_energy_square_well_device(part, 1.5);

      double3 const new_pos =
          view.try_random_particle_disp(p_idx, unif_r, re, 0.5);
      if (new_pos.x == -1) {
        continue;
      }
      // part.pos = new_pos;
      // double new_energy = view.particle_energy_square_well(part, 0.2, 1);
      // // double const new_energy =
      // //     view.particle_energy_square_well_device(part, 1.5);
      // part.pos = old_pos;

      // double prob = exp((old_energy - new_energy) / TEMPERATURE);
      // if (unif_r(re) >= prob) {
      //   continue;
      // }
      view.remove_particle(view.box.particles[p_idx]);
      part.pos = new_pos;
      view.box.update_particle(p_idx);
      view.add_particle(view.box.particles[p_idx]);
    }
  }

  std::ofstream other_file("output.dat");
  other_file << std::fixed << std::setprecision(6);
  double const coeff = (ITERATIONS / ITERATIONS_PER_GRF_EXPORT) + 1; 
  for (const auto &[r, val] : distr) {
    double const real_val = val / coeff;
    if (real_val <= 0.1) {
      continue;
    }
    other_file << r << "    " << real_val << std::endl;
  }

  // std::map<double, double> tmp_distr = do_distr(view, rho, 0.95, 0.01L);
  // for (const auto &[radius, value] : tmp_distr) {
  //     distr[radius] += value;
  // }
  // std::ofstream other_file("output.dat");
  // other_file << std::fixed << std::setprecision(6);
  // double const coeff = (ITERATIONS / ITERATIONS_PER_GRF_EXPORT) + 1; 
  // for (const auto &[r, val] : distr) {
  //   if (val <= 0.1) {
  //     continue;
  //   }
  //   other_file << r << "    " << val << std::endl;
  // }

  view.free();

  return 0;
}
