#include "cell_view.h"
#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"

#include "curand_gen.h"
#include "exceptions.h"
#include <algorithm>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <random>
#include <sstream>
#include <vector>

// constexpr size_t PARTICLE_COUNT = 200;
// constexpr size_t MOVES_PER_ITER = 200;
constexpr size_t ITERATIONS = 10'000;
constexpr size_t ITERATIONS_PER_EXPORT = 10;
constexpr size_t ITERATIONS_PER_GRF_EXPORT = 10;
constexpr double TEMPERATURE = 0.85;
// constexpr double TEMPERATURE = 3;
// constexpr double MAX_STEP = 0.2886751346L;
constexpr double MAX_STEP = 0.5;
//
// int main() {
//   float *devFloats;
//   float *hostFloats = new float[200];
//   curand_gen_t gen(10, 10);
//   cudaMalloc(&devFloats, sizeof(float) * 200);
//
//   gen.generate_random_numbers(devFloats);
//   cudaMemcpy(hostFloats, devFloats, sizeof(float) * 200,
//              cudaMemcpyDeviceToHost);
//   for (int i = 0; i < 200; i++) {
//     std::cout << hostFloats[i] << std::endl;
//   }
//   cudaFree(devFloats);
//   delete[] hostFloats;
//   return 0;
// }

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

int main(int argc, char *argv[]) {
  size_t PARTICLE_COUNT = 200;
  size_t MOVES_PER_ITER = 200;

  switch (argc) {
  case 3:
    try {
      PARTICLE_COUNT = std::stoul(argv[2]);
      MOVES_PER_ITER = std::stoul(argv[1]);
    } catch (const std::exception &e) {
      throw InvalidArgumentType();
    }
    break;
  case 2:
    try {
      MOVES_PER_ITER = std::stoul(argv[1]);
    } catch (const std::exception &e) {
      throw InvalidArgumentType();
    }
    break;
  default:
    throw InvalidNumberOfArguments();
  }

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

  double const rho =
      view.box.particle_count /
      (view.box.dimensions.x * view.box.dimensions.y * view.box.dimensions.z);
  std::map<double, double> distr{};
  double init_energy = view.total_energy(0.2, -1);
  std::vector<double> energies;

  double *hostFloats = new double[4 * MOVES_PER_ITER];
  curand_gen_t gen(20, MOVES_PER_ITER / 10);
  // std::uniform_real_distribution<double> unif_r(0, 1);

  for (size_t iters = 1; iters <= 2 * ITERATIONS; iters++) {
    if (iters >= ITERATIONS) {
      if (iters % ITERATIONS_PER_GRF_EXPORT == 0) {
        std::map<double, double> tmp_distr = do_distr(view, rho, 1, 0.02L, 5);
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
    }

    // for (size_t i = 0; i < MOVES_PER_ITER; i++) {
    //   size_t const p_idx =
    //       static_cast<size_t>(unif_r(re) * view.box.particle_count) %
    //       view.box.particle_count;
    //   double const offset = unif_r(re) - 0.5;
    //   double3 const new_pos =
    //       view.try_random_particle_disp(p_idx, offset, MAX_STEP);
    //   double const prob_rand = unif_r(re);
    //   double angle = unif_r(re) * M_PI;
    //   double4 rotation =
    //       particle_t::random_particle_orient(angle, (i + iters) % 3);
    //   init_energy += view.try_move_particle(p_idx, new_pos, rotation,
    //   prob_rand,
    //                                         TEMPERATURE);
    // }
    //
    gen.generate_random_numbers();
    gen.copyToHost(hostFloats);
    for (size_t i = 0; i < MOVES_PER_ITER; i++) {
      size_t const r_idx = i * 4;
      size_t const p_idx =
          static_cast<size_t>(hostFloats[r_idx] * view.box.particle_count) %
          view.box.particle_count;
      double const offset = hostFloats[r_idx + 1] - 0.5;
      double3 const new_pos =
          view.try_random_particle_disp(p_idx, offset, MAX_STEP);
      double const prob_rand = hostFloats[r_idx + 2];
      double angle = hostFloats[r_idx + 3] * M_PI;
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
  delete[] hostFloats;

  return 0;
}
