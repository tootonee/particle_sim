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

std::vector<size_t> getCellsInDomain(const cell_view_t& view, int domainNumber) {
    std::vector<size_t> cellsInDomain;
    size_t cells_per_axis = view.cells_per_axis;
    for (size_t x = 0; x < cells_per_axis; ++x) {
        for (size_t y = 0; y < cells_per_axis; ++y) {
            for (size_t z = 0; z < cells_per_axis; ++z) {
                size_t cell_index = x * cells_per_axis * cells_per_axis + y * cells_per_axis + z;
                int calculatedDomain = 1;
                if (x % 2 == 0) {
                    calculatedDomain += 4;
                }
                if (y % 2 == 0) {
                    calculatedDomain += 2;
                }
                if (z % 2 == 0) {
                    calculatedDomain += 1;
                }
                if (calculatedDomain == domainNumber) {
                    cellsInDomain.push_back(cell_index);
                }
            }
        }
    }
    return cellsInDomain;
}

void accessCell(cell_view_t& view, size_t cellIndex) {
    if (cellIndex >= view.cells_per_axis * view.cells_per_axis * view.cells_per_axis) {
        std::cout << "Cell index is out of bounds!" << std::endl;
        return;
    }
    cell_t& cell = view.cells[cellIndex];
    std::cout << "Number of particles in cell " << cellIndex << ": " << cell.num_particles << std::endl;
}
void simulateParticleMovements(
        cell_view_t& view,
        double* hostFloats,
        size_t movesPerIter,
        double& initEnergy,
        int iters,
        double maxStep,
        double temperature
) {
    for (size_t i = 0; i < movesPerIter; i++) {
        size_t r_idx = i * 6;
        size_t p_idx = static_cast<size_t>(hostFloats[r_idx] * view.box.particle_count) % view.box.particle_count;

        double3 offset = {
                .x = hostFloats[r_idx + 1] - 0.5,
                .y = hostFloats[r_idx + 2] - 0.5,
                .z = hostFloats[r_idx + 3] - 0.5,
        };
        double3 new_pos = view.try_random_particle_disp(p_idx, offset, maxStep);
        double prob_rand = hostFloats[r_idx + 4];
        double angle = hostFloats[r_idx + 5] * M_PI;
        double4 rotation = particle_t::random_particle_orient(angle, (i + iters) % 3);

        initEnergy += view.try_move_particle_device(p_idx, new_pos, rotation, prob_rand, temperature);
    }
}


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

struct ParticleMovement {
    size_t p_idx;
    size_t cellIndex;
    double3 newPos;
};

__global__ void simulateParticleMovementsKernel(
        cell_view_t view,
        double* hostFloats,
        size_t movesPerIter,
        double* energies,
        int iters,
        double maxStep,
        double temperature,
        size_t* cellIndices,
        size_t numCells,
        ParticleMovement* movements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCells) {
        size_t cellIndex = cellIndices[idx];
        cell_t cell = view.cells[cellIndex];
        double initEnergy = 0.0;

        for (size_t i = 0; i < movesPerIter; i++) {
            size_t r_idx = i * 6;
            if (cell.num_particles == 0) continue;
            size_t particleIndexInCell = static_cast<size_t>(hostFloats[r_idx] * cell.num_particles) % cell.num_particles;
            size_t p_idx = cell.particle_indices[particleIndexInCell];

            double3 offset = {
                    hostFloats[r_idx + 1] - 0.5,
                    hostFloats[r_idx + 2] - 0.5,
                    hostFloats[r_idx + 3] - 0.5
            };
            double3 new_pos = view.try_random_particle_disp_device(view.cells,view.cell_size, view.cell_indices, view.cells_per_axis,view.box.particles,view.box.particle_count,view.box.dimensions,p_idx, offset, maxStep);
//            double prob_rand = hostFloats[r_idx + 4];
//            double angle = hostFloats[r_idx + 5] * M_PI;
//            double4 rotation = particle_t::random_particle_orient_device(angle, (i + iters) % 3);
//
//            initEnergy += view.try_move_particle_device(p_idx, new_pos, rotation, prob_rand, temperature);
            movements[idx * movesPerIter + i].p_idx = p_idx;
            movements[idx * movesPerIter + i].cellIndex = cellIndex;
            movements[idx * movesPerIter + i].newPos = new_pos;
        }

        energies[idx] = initEnergy;
    }
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
    std::uniform_real_distribution<double> unif_x(0, 15);
    std::uniform_real_distribution<double> unif_y(0, 15);
    std::uniform_real_distribution<double> unif_z(0, 15);
    cell_view_t view({15, 15, 15}, 4);
    std::vector<int> domains = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(domains.begin(), domains.end(), std::default_random_engine(seed));

    for (int num : domains) {
        std::vector<size_t> cellsInDomain = getCellsInDomain(view, num);
        std::cout << "Cells in domain " << num << ": ";
        for (size_t index : cellsInDomain) {
            std::cout << index << " ";
        }
        std::cout << std::endl;
    }

    int moves_per_iter = MOVES_PER_ITER/(view.cells_per_axis*view.cells_per_axis*view.cells_per_axis/8);

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
        size_t domain_idx = (iters - 1) % 8;
        std::vector <size_t> cellsInDomain = getCellsInDomain(view, domain_idx);

        if (iters % ITERATIONS_PER_EXPORT == 0) {
            const size_t idx = iters / ITERATIONS_PER_EXPORT;
            char buf[25];
            std::sprintf(buf, "data/%06li.pdb", idx);
            export_particles_to_pdb(view.box, buf);
            std::cout << "I = " << iters << ", energy = " << init_energy << std::endl;
        }

        if (iters % ITERATIONS_PER_GRF_EXPORT == 0) {
            std::map<double, double> tmp_distr = do_distr(view, rho, 1, 0.1L, 8);
            for (const auto &[radius, value]: tmp_distr) {
                distr[radius] += value;
            }
        }

        gen.generate_random_numbers();
        gen.copyToHost(hostFloats);

        size_t numCells = cellsInDomain.size();
        double *hostEnergies = new double[numCells];
        ParticleMovement *hostMovements = new ParticleMovement[numCells * moves_per_iter];

        size_t *deviceCellIndices;
        cudaMalloc(&deviceCellIndices, sizeof(size_t) * cellsInDomain.size());
        cudaMemcpy(deviceCellIndices, cellsInDomain.data(), sizeof(size_t) * cellsInDomain.size(),
                   cudaMemcpyHostToDevice);


        double *deviceEnergies;
        cudaMalloc(&deviceEnergies, numCells * sizeof(double));

        ParticleMovement *deviceMovements;
        cudaMalloc(&deviceMovements, numCells * moves_per_iter * sizeof(ParticleMovement));

        dim3 blockSize(256);
        dim3 gridSize((numCells + blockSize.x - 1) / blockSize.x);

        simulateParticleMovementsKernel<<<gridSize, blockSize>>>(
                view, hostFloats, moves_per_iter, deviceEnergies, iters, MAX_STEP, TEMPERATURE, deviceCellIndices,
                numCells, deviceMovements
        );

        cudaMemcpy(hostEnergies, deviceEnergies, numCells * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostMovements, deviceMovements, numCells * moves_per_iter * sizeof(ParticleMovement),
                   cudaMemcpyDeviceToHost);

        cudaFree(deviceMovements);
        cudaFree(deviceEnergies);
        cudaFree(deviceCellIndices);

        for (size_t i = 0; i < numCells * moves_per_iter; i++) {
            ParticleMovement &movement = hostMovements[i];
            double3 new_pos = movement.newPos;
            if (!(new_pos.x == -1 && new_pos.y == -1 && new_pos.z == -1)) {
                view.box.particles[movement.p_idx].pos = new_pos;
                std::cout<<new_pos.x;
            }
        }
        for (size_t i = 0; i < numCells; i++) {
            init_energy += hostEnergies[i];
        }
    }
    auto finish = getCurrentTimeFenced();
    auto total_time = finish - start;
    std::cout << "TIME " << to_us(total_time) << std::endl;
    for (int i=0; i<65; i++){
        accessCell(view, i);
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
