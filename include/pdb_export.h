#pragma once

#ifndef PDB_EXPORT_H
#define PDB_EXPORT_H

#include <string>

#include "particle.h"
#include "particle_box.h"

void export_particles_to_pdb(particle_box_t const &box,
                             std::string const &filename);

#endif
