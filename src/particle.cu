#include "particle.h"
#include "vec.h"

__host__ __device__ bool Particle::intersects(Particle const &rhs) {
  double const diameter = radius + rhs.radius;
  return distance(pos, rhs.pos) - diameter < ERR;
}
