#pragma once

#ifndef EXPORT_TO_LAMMPS_H
#define EXPORT_TO_LAMMPS_H

#include <string>

#include "particle.h"
#include "particle_box.h"

void export_particles_to_lammps(particle_box_t const &box, size_t iteration, double radius);

#endif
