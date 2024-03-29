#include "particle.h"
#include "vec.h"

#include <random>

void particle_t::random_particle_pos(double3 dimensions) {
  std::random_device r;
  std::mt19937 re(r());

  std::uniform_real_distribution<double> unif_x(0, dimensions.x);
  std::uniform_real_distribution<double> unif_y(0, dimensions.y);
  std::uniform_real_distribution<double> unif_z(0, dimensions.z);

  pos.x = unif_x(re);
  pos.y = unif_y(re);
  pos.z = unif_z(re);
};

void particle_t::random_particle_orient(rng_gen &rng_r, std::mt19937 &re,
                                        int axis) {
  double angle = rng_r(re);
  double4 rotation{
      .x = cos(angle / 2),
      .y = 0,
      .z = 0,
      .w = 0,
  };
  switch (axis) {
  case 2:
    rotation.w = sin(angle / 2);
    break;
  case 1:
    rotation.z = sin(angle / 2);
    break;
  default:
    rotation.x = sin(angle / 2);
    break;
  };
  double4 conj = {
      .x = rotation.x,
      .y = -rotation.y,
      .z = -rotation.z,
      .w = -rotation.w,
  };
};

void particle_t::random_particle_pos(rng_gen &rng_x, rng_gen &rng_y,
                                     rng_gen &rng_z, std::mt19937 &re) {
  pos.x = rng_x(re);
  pos.y = rng_y(re);
  pos.z = rng_z(re);
}
