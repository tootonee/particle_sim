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

constexpr size_t PARTICLE_COUNT = 600;
constexpr size_t MOVES_PER_ITER = 600;
constexpr size_t ITERATIONS = 10'000;
constexpr size_t ITERATIONS_PER_EXPORT = 10;
constexpr size_t ITERATIONS_PER_GRF_EXPORT = 2500;
// constexpr double TEMPERATURE = 0.88;
constexpr double TEMPERATURE = 3;
// constexpr double MAX_STEP = 0.15;
// constexpr double MAX_STEP = 0.2886751346L;
constexpr double MAX_STEP = 0.5;

std::map<double, double> do_distr(cell_view_t const &view,
                                  double const rho = 0.5L,
                                  double const start = 1L,
                                  double const dr = 0.01L,
                                  double const max_r = 5L) {
  std::map<double, double> distr{};
  double radius = start;
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
  cell_view_t view({10, 10, 10}, 10);

  // view.box.make_box_uniform_particles_host({10, 10, 10}, 0.5, 8);
  for (size_t i = 0; i < PARTICLE_COUNT; i++) {
    view.add_particle_random_pos(0.5, unif_x, unif_y, unif_z, re);
    // view.add_particle(view.box.particles[i]);
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, 1, 0, 0},
    });
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, -1, 0, 0},
    });
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, 0, 0, 1},
    });
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, 0, 0, -1},
    });
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, 0, 1, 0},
    });
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, 0, -1, 0},
    });
  }

  std::uniform_real_distribution<double> unif_r(0, 1);

  double const rho =
      view.box.particle_count /
      (view.box.dimensions.x * view.box.dimensions.y * view.box.dimensions.z);
  std::map<double, double> distr{};
  double init_energy = view.total_energy(0.5, 0.25);

  std::vector<double> energies;
  for (size_t iters = 1; iters <= ITERATIONS; iters++) {
    if (iters % ITERATIONS_PER_GRF_EXPORT == 0) {
      std::map<double, double> tmp_distr = do_distr(view, rho, 1, 0.01L, 5);
      for (const auto &[radius, value] : tmp_distr) {
        distr[radius] += value;
      }
    }

    if (iters % ITERATIONS_PER_EXPORT == 0) {
      const size_t idx = iters / ITERATIONS_PER_EXPORT;
      char buf[16];
      std::sprintf(buf, "data/%06li.pdb", idx);
      export_particles_to_pdb(view.box, buf);
      std::cout << "I = " << idx << ", energy = " << init_energy << std::endl;
    }

    for (size_t i = 0; i < MOVES_PER_ITER; i++) {
      size_t const p_idx =
          static_cast<size_t>(unif_r(re) * view.box.particle_count) %
          view.box.particle_count;
      double const offset = unif_r(re) - 0.5;
      double3 const new_pos =
          view.try_random_particle_disp(p_idx, offset, MAX_STEP);
      double const prob_rand = unif_r(re);
      double angle = unif_r(re) * M_PI;
      double4 rotation =
          particle_t::random_particle_orient(angle, (i + iters) % 3);
      init_energy += view.try_move_particle(p_idx, new_pos, rotation, prob_rand,
                                            TEMPERATURE);
    }
    energies.push_back(init_energy);
  }

  std::ofstream other_file("output.dat");
  other_file << std::fixed << std::setprecision(6);
  double const coeff = ITERATIONS / ITERATIONS_PER_GRF_EXPORT;
  for (const auto &[r, val] : distr) {
    double const real_val = val / coeff;
    if (real_val <= 0.5) {
      continue;
    }
    other_file << r << "    " << real_val << std::endl;
  }
  std::ofstream file("energies.dat");
  file << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < energies.size(); i++) {
    file << i / ITERATIONS << "    "
         << 0.5 * energies[i] / view.box.particle_count << std::endl;
  }
  view.free();

  return 0;
}
