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

void particle_t::random_particle_pos(rng_gen &rng_x, rng_gen &rng_y,
                                     rng_gen &rng_z, std::mt19937 &re) {
  pos.x = rng_x(re);
  pos.y = rng_y(re);
  pos.z = rng_z(re);
}
