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
    std::cout << "ATOM" << std::right << std::setw(7) << std::fixed
              << (p_idx + 1) << "  N   NONE   1" << std::right << std::setw(12)
              << std::fixed << box.particles[p_idx].pos.x << std::right
              << std::setw(8) << box.particles[p_idx].pos.y << std::right
              << std::setw(8) << box.particles[p_idx].pos.z << std::right
              << std::setw(8) << "  1.00  1.00" << std::endl;
#endif
    file << "ATOM" << std::right << std::setw(7) << std::fixed << (p_idx + 1)
         << "  N   NONE   1" << std::right << std::setw(12) << std::fixed
         << box.particles[p_idx].pos.x << std::right << std::setw(8)
         << box.particles[p_idx].pos.y << std::right << std::setw(8)
         << box.particles[p_idx].pos.z << std::right << std::setw(8)
         << "  1.00  1.00" << std::endl;

    // for (size_t i = 0; i < box.particles[p_idx].patch_count; i++) {
    //   const patch_t &patch = box.particles[p_idx].patches[i];
    //   file << "PATCH" << std::right << std::setw(6) << std::fixed << (p_idx +
    //   1)
    //       << "  P   NONE   1" << std::right << std::setw(12) << std::fixed
    //       << patch.pos.y*patch.radius + box.particles[p_idx].pos.x <<
    //       std::right << std::setw(8)
    //       << patch.pos.z*patch.radius + box.particles[p_idx].pos.y <<
    //       std::right << std::setw(8)
    //       << patch.pos.w*patch.radius + box.particles[p_idx].pos.z <<
    //       std::right << std::setw(8)
    //       << "  1.00  1.00" << std::endl;
    // }
  }
  size_t atom_index_offset = box.particle_count;
  for (size_t p_idx = 0; p_idx < box.particle_count; p_idx++) {
    for (size_t i = 0; i < box.particles[p_idx].patch_count; i++) {
      const patch_t &patch = box.particles[p_idx].patches[i];
      file << "ATOM" << std::right << std::setw(7) << std::fixed
           << (atom_index_offset + i + 1) << "  P   NONE   1" << std::right
           << std::setw(12) << std::fixed
           << patch.pos.y * box.particles[p_idx].radius +
                  box.particles[p_idx].pos.x
           << std::right << std::setw(8)
           << patch.pos.z * box.particles[p_idx].radius +
                  box.particles[p_idx].pos.y
           << std::right << std::setw(8)
           << patch.pos.w * box.particles[p_idx].radius +
                  box.particles[p_idx].pos.z
           << std::right << std::setw(8) << "  1.00  1.00" << std::endl;
    }
    atom_index_offset += box.particles[p_idx].patch_count;
  }
}
