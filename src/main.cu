#include "cell_view.h"
#include "conf.h"
#include "curand_gen.h"
#include "exceptions.h"
#include "export_to_lammps.h"
#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"
#include "time_calculation.h"
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

std::map<double, double> do_distr(cell_view_t const &view,
                                  double const rho = 0.5L,
                                  double const start = 1L,
                                  double const dr = 0.01L,
                                  double const max_r = 5L) {
  std::map<double, double> distr{};
  double radius = start;

  for (radius = start; radius < max_r; radius += dr) {
    std::cout << "Prog = " << (radius - start) / (max_r - start) * 100 << "%\n";
    double num = 0.0F;
    for (size_t p_idx = 0; p_idx < view.box.particle_count; p_idx++) {
      num += view.particles_in_range_device(p_idx, radius, radius + dr);
    }
    double v_old = (radius - dr) * (radius - dr) * (radius - dr);
    double v_new = radius * radius * radius;
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

  // std::uniform_real_distribution<double> unif_x(0, 30);
  // std::uniform_real_distribution<double> unif_y(0, 30);
  // std::uniform_real_distribution<double> unif_z(0, 30);
  // cell_view_t view({30, 30, 30}, 6);

  std::uniform_real_distribution<double> unif_x(0, 15);
  std::uniform_real_distribution<double> unif_y(0, 15);
  std::uniform_real_distribution<double> unif_z(0, 15);
  cell_view_t view({15, 15, 15}, 2);

  // view.box.make_box_uniform_particles_host({10, 10, 10}, 0.5, 8);
  for (size_t i = 0; i < PARTICLE_COUNT; i++) {
    view.add_particle_random_pos(0.5, unif_x, unif_y, unif_z, re);
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, 1, 0, 0},
    });
    view.box.particles[i].add_patch({
        .radius = 0.05,
        .pos = {1, -1, 0, 0},
    });
    // view.box.particles[i].add_patch({
    //     .radius = 0.05,
    //     .pos = {1, 0, 0, 1},
    // });
    // view.box.particles[i].add_patch({
    //     .radius = 0.05,
    //     .pos = {1, 0, 0, -1},
    // });
    // view.box.particles[i].add_patch({
    //     .radius = 0.05,
    //     .pos = {1, 0, 1, 0},
    // });
    // view.box.particles[i].add_patch({
    //     .radius = 0.05,
    //     .pos = {1, 0, -1, 0},
    // });
  }
  std::cout << "Particle gen done!\n";

  double const rho =
      view.box.particle_count /
      (view.box.dimensions.x * view.box.dimensions.y * view.box.dimensions.z);
  std::map<double, double> distr{};
  // double init_energy = view.total_energy(0.2, -1);
  double init_energy = 0;
  std::vector<double> energies;

  size_t N{};
  if (MOVES_PER_ITER > THREADS_PER_BLOCK) {
    N = 6 * (MOVES_PER_ITER / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK;
  } else {
    N = 6 * MOVES_PER_ITER;
  }
  double *hostFloats = new double[N];
  size_t blocks{};
  size_t threads{};
  if (MOVES_PER_ITER > THREADS_PER_BLOCK) {
    blocks = 3 * (MOVES_PER_ITER / THREADS_PER_BLOCK) + 3;
    threads = THREADS_PER_BLOCK;
  } else {
    blocks = 3;
    threads = MOVES_PER_ITER;
  }
  curand_gen_t gen(blocks, threads);
  auto start = getCurrentTimeFenced();
  std::uniform_real_distribution<double> unif_r(0, 1);

  start = getCurrentTimeFenced();
  for (size_t iters = 1; iters <= ITERATIONS; iters++) {
    if (iters % ITERATIONS_PER_EXPORT == 0) {
      const size_t idx = iters / ITERATIONS_PER_EXPORT;
      char buf[25];
      std::sprintf(buf, "data_cpu/%06li.pdb", idx);
      export_particles_to_pdb(view.box, buf);
      std::cout << "I = " << iters << ", energy = " << init_energy << std::endl;
    }

    if (iters % ITERATIONS_PER_GRF_EXPORT == 0) {
      //pho move
      double const rho = view.box.particle_count /
            (view.box.dimensions.x * view.box.dimensions.y * view.box.dimensions.z);
      std::map<double, double> tmp_distr = do_distr(view, rho, 1, 0.01L, 8);
      for (const auto &[radius, value] : tmp_distr) {
        distr[radius] += value;
      }
    }
    //
    // for (size_t i = 0; i < MOVES_PER_ITER; i++) {
    //   size_t const p_idx =
    //       static_cast<size_t>(unif_r(re) * view.box.particle_count) %
    //       view.box.particle_count;
    //   double3 const offset = {
    //       .x = unif_r(re) - 0.5,
    //       .y = unif_r(re) - 0.5,
    //       .z = unif_r(re) - 0.5,
    //   };
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

    gen.generate_random_numbers();
    gen.copyToHost(hostFloats);
    std::uniform_real_distribution<double> choice_muvt(0.0, 1.0);
    double choice_probability = choice_muvt(re); //moving/rotation or adding/removing
    if (choice_probability < 0.5) {
      for (size_t i = 0; i < MOVES_PER_ITER; i++) { //think if we need it outside
        size_t const r_idx = i * 6;
        size_t const p_idx =
            static_cast<size_t>(hostFloats[r_idx] * view.box.particle_count) %
            view.box.particle_count;
        double3 const offset = {
            .x = hostFloats[r_idx + 1] - 0.5,
            .y = hostFloats[r_idx + 2] - 0.5,
            .z = hostFloats[r_idx + 3] - 0.5,
        };
        double3 const new_pos =
            view.try_random_particle_disp(p_idx, offset, MAX_STEP);
        double const prob_rand = hostFloats[r_idx + 4];
        double angle = hostFloats[r_idx + 5] * M_PI;
        double4 rotation =
            particle_t::random_particle_orient(angle, (i + iters) % 3);
        init_energy += view.try_move_particle(p_idx, new_pos, rotation, prob_rand,
                                              TEMPERATURE);
      }
    } else {
      init_energy += view.add_particle_muvt(unif_x, unif_y, unif_z, re);
    }

    energies.push_back(init_energy);
  }

  auto finish = getCurrentTimeFenced();
  auto total_time = finish - start;
  std::cout << "TIME " << to_us(total_time) << std::endl;

  std::ofstream other_file("output_cpu.dat");
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
