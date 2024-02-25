#include "particle.h"
#include "particle_box.h"
#include "pdb_export.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>

void export_particles_to_pdb(particle_box_t const &box,
                             std::string const &filename) {
  std::ofstream file(filename);
  file << std::fixed << std::setprecision(3);

#ifndef NDEBUG
  std::cout << std::fixed << std::setprecision(3);
#endif

  for (size_t p_idx = 0; p_idx < box.particle_count; p_idx++) {
#ifndef NDEBUG
    std::cout << "ATOM" << std::right << std::setw(7) << std::fixed << (p_idx + 1)
              << "  N   NONE   1" << std::right << std::setw(12) << std::fixed
              << box.particles[p_idx].pos.x << std::right << std::setw(8)
              << box.particles[p_idx].pos.y << std::right << std::setw(8)
              << box.particles[p_idx].pos.z << std::right << std::setw(8)
              << "1.00  1.00" << std::endl;
#endif
    file << "ATOM" << std::right << std::setw(7) << std::fixed << (p_idx + 1)
              << "  N   NONE   1" << std::right << std::setw(12) << std::fixed
              << box.particles[p_idx].pos.x << std::right << std::setw(8)
              << box.particles[p_idx].pos.y << std::right << std::setw(8)
              << box.particles[p_idx].pos.z << std::right << std::setw(8)
              << "1.00  1.00" << std::endl;
  }
}
