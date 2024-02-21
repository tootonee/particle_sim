#include "particle.h"

__host__ __device__ bool Particle::intersects(Particle const &rhs) {
  double const diameter = radius + rhs.radius;
  return vec_3d_distance_squared(pos, rhs.pos) < diameter * diameter;
}
