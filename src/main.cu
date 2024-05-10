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
#include <unordered_map>
#include "domain_decomposition.h"
#include "thread_pool.h"
#include <future>
#include <memory>

std::map<double, double> do_distr(cell_view_t const &view,
                                  double const rho = 0.5L,
                                  double const start = 1L,
                                  double const dr = 0.01L,
                                  double const max_r = 5L) {
  std::map<double, double> distr{};
  double radius = start;

  for (radius = start; radius < max_r; radius += dr) {
    // std::cout << "Prog = " << (radius - start) / (max_r - start) * 100 << "%\n";
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
  cell_view_t view({10, 10, 10}, 4);

  // thread_pool pool;
  // std::vector<std::future<double>> futures;
 

  

  // view.box.make_box_uniform_particles_host({10, 10, 10}, 0.5, 8);
  for (size_t i = 0; i < PARTICLE_COUNT; i++) {
    view.add_particle_random_pos(0.5, unif_x, unif_y, unif_z, re);
    view.box.particles[i].add_patch({
        .radius = 0.119,
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
  // std::cout << "Particle gen done!\n";

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
  std::uniform_int_distribution<int> unif_domain(0, 7);

  start = getCurrentTimeFenced();
  

  auto domain_decomposition = domain_division(view);
  int num_particles = 0;
  int particle_idx = 0;
  int domain = 0;
  // for (auto value : domain_decomposition){
  //   std::cout << value.first << "Type of domain" << std::endl;
  //   for (auto idx : value.second){
  //     std::cout << idx << std::endl;
  //   }
  // }
  // 
  // int num_particles = 0;
  // for (int i = 0; i < 100; i++){
  //   domain = unif_domain(re);
  //   std::cout << "Try " << i << "Domain" << domain << std::endl;
  //   for (auto cell : domain_decomposition[domain]){
  //     num_particles = view.cells[cell].num_particles;
  //     std::cout << view.cells[cell].particle_indices[(int)(num_particles * unif_r(re))] << std::endl;
  //   }
  // }




  // std::cout << view.get_cell_idx({0, 0, 0}).particle_indices
  // size_t cell_idx = view.get_cell_idx({8, 8, 0});
  // cell_t& cell = view.cells[cell_idx];
  // size_t num_particles = cell.num_particles;
  // for (size_t i = 0; i < num_particles; i++){
  //   std::cout << cell.particle_indices[i] << std::endl;
  // }


  // ThreadSafeQueue<double> results;
  // ThreadSafeQueue<Task> tasks;
  // std::vector<std::thread> workers;
    // for (int i = 0; i < NUMTHREADS; ++i) {
    //     workers.emplace_back(worker, std::ref(tasks), std::ref(results));
    // }
  int sequential_index = 0;
  for (size_t iters = 1; iters <= ITERATIONS; iters++) {
    if (iters % ITERATIONS_PER_EXPORT == 0) {
      const size_t idx = iters / ITERATIONS_PER_EXPORT;
      char buf[25];
      std::sprintf(buf, "data_cpu/%06li.pdb", idx);
      export_particles_to_pdb(view.box, buf);
      // std::cout << "I = " << iters << ", energy = " << init_energy << std::endl;
    }

    if (iters % ITERATIONS_PER_GRF_EXPORT == 0) {
      std::map<double, double> tmp_distr = do_distr(view, rho, 1, 0.01L, 8);
      for (const auto &[radius, value] : tmp_distr) {
        distr[radius] += value;
      }
    }

    // gen.generate_random_numbers();
    // gen.copyToHost(hostFloats);
    // sequential_index = 0;

/// <---------------------------------------------------------------
///CODE FOR THREAD POOL
    // while(sequential_index < MOVES_PER_ITER){
    //   domain = unif_domain(re);
    //   for (auto cell : domain_decomposition[domain]){
    //     num_particles = view.cells[cell].num_particles;
    //     particle_idx = view.cells[cell].particle_indices[(int)(num_particles * unif_r(re))];
    //     Task task = {
    //       view,
    //       particle_idx,
    //       iters,
    //       sequential_index,
    //       hostFloats,
    //       TEMPERATURE
    //     };
    //     sequential_index++;
    //     std::future<double> future = pool.submit([task]() {
    //       return worker(task);
    //     });
    //     futures.push_back(std::move(future));
    //   }
    // }

    // for (auto& future : futures){
    //     init_energy += future.get();
    // }

    // futures.clear();

/// <---------------------------------------------------------------

/// <---------------------------------------------------------------
/// CODE FOR SAFE QUEUE;

    // gen.generate_random_numbers();
    // gen.copyToHost(hostFloats);
    // sequential_index = 0;

    // while(sequential_index < MOVES_PER_ITER){
    //   domain = unif_domain(re);
    //   for (auto cell : domain_decomposition[domain]){
    //     num_particles = view.cells[cell].num_particles;
    //     particle_idx = view.cells[cell].particle_indices[(int)(num_particles * unif_r(re))];
    //     Task task = {
    //       view,
    //       particle_idx,
    //       iters,
    //       sequential_index,
    //       hostFloats,
    //       TEMPERATURE,
    //       false,
    //     };
    //     sequential_index++;
    //     tasks.push(task);
    //   }
    // }

    // for( int i = 0; i < sequential_index; i++){
    //   std::shared_ptr<double> result = results.wait_and_pop();
    //   init_energy += *result;
    //  }
        

/// <---------------------------------------------------------------
    for (size_t i = 0; i < MOVES_PER_ITER; i++) {
      size_t const p_idx =
          static_cast<size_t>(unif_r(re) * view.box.particle_count) %
          view.box.particle_count;
    double const x = unif_r(re) - 0.5;
    double const y = unif_r(re) - 0.5;
    double const z = sqrt(1 - x * x - y * y);
    double3 const offset = {
        .x = x,
        .y = y,
        .z = z,
    };
    double3 const new_pos =
        view.try_random_particle_disp(p_idx, offset, MAX_STEP);
    double const prob_rand = unif_r(re);
    double angle = unif_r(re) * M_PI;
    double4 rotation =
        particle_t::random_particle_orient(angle, (i + iters) % 3);
    init_energy += view.try_move_particle(p_idx, new_pos, rotation,
    prob_rand,
                                          TEMPERATURE);
    }

    // gen.generate_random_numbers();
    // gen.copyToHost(hostFloats);

    // for (size_t i = 0; i < MOVES_PER_ITER; i++) {
    //   size_t const r_idx = i * 6;
    //   size_t const p_idx =
    //       static_cast<size_t>(hostFloats[r_idx] * view.box.particle_count) %
    //       view.box.particle_count;
    //   double const x = hostFloats[r_idx + 1] - 0.5;
    //   double const y = hostFloats[r_idx + 2] - 0.5;
    //   double const z = sqrt(1 - x * x - y * y);
    //   double3 const offset = {
    //       .x = x,
    //       .y = y,
    //       .z = z,
    //   };
    //   double3 const new_pos =
    //       view.try_random_particle_disp(p_idx, offset, MAX_STEP);
    //   double const prob_rand = hostFloats[r_idx + 4];
    //   double angle = hostFloats[r_idx + 5] * M_PI;
    //   double4 rotation =
    //       particle_t::random_particle_orient(angle, (i + iters) % 3);
    //   init_energy += view.try_move_particle(p_idx, new_pos, rotation, prob_rand,
    //                                         TEMPERATURE);
    // }
    energies.push_back(init_energy);
  }

  // for (int i = 0; i < NUMTHREADS; i++){
  //   Task task = {
  //     view,
  //     particle_idx,
  //     0,
  //     sequential_index,
  //     hostFloats,
  //     TEMPERATURE,
  //     true,
  //   };
  //   tasks.push(task);
  // }
  // for(auto &t: workers){
  //               if (t.joinable()) t.join();
  //           }

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
