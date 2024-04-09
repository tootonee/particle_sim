#include "particle.h"
#include "particle_box.h"
#include "export_to_lammps.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
void export_particles_to_lammps(particle_box_t const &box, size_t iteration, double radius) {
    std::ofstream lammps_file("trajectory.lammpstrj", std::ios_base::app);
    lammps_file << "ITEM: TIMESTEP\n" << iteration << "\n";
    lammps_file << "ITEM: NUMBER OF ATOMS\n" << box.particle_count<< "\n";
    lammps_file << "ITEM: BOX BOUNDS ff ff ff\n0 " << 10 << "\n0 " << 10 << "\n0 " << 10 << "\n";
    lammps_file << "ITEM: ATOMS id type x y z radius\n";

    for (size_t p_idx = 0; p_idx < box.particle_count; p_idx++) {
        lammps_file << p_idx + 1 << " 1 " << box.particles[p_idx].pos.x  << " " <<  box.particles[p_idx].pos.y  << " " <<  box.particles[p_idx].pos.z << " " << radius << "\n";
    }
    lammps_file.close();

}

