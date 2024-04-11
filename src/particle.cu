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

double4 particle_t::random_particle_orient(double const angle, int axis) {
  const double cos_2 = cos(angle / 2);
  const double sin_2 = sin(angle / 2);
  switch (axis) {
  case 2:
    return {cos_2, 0, 0, sin_2};
    break;
  case 1:
    return {cos_2, 0, sin_2, 0};
    break;
  default:
    return {cos_2, sin_2, 0, 0};
    break;
  };
};

void particle_t::rotate(double4 const rot) {
  double4 conj = {
      .x = rot.x,
      .y = -rot.y,
      .z = -rot.z,
      .w = -rot.w,
  };

  orient = rot * (orient * conj);
  for (size_t i = 0; i < patch_count; i++) {
    patches[i].pos = rot * (patches[i].pos * conj);
  }
}

void particle_t::random_particle_pos(rng_gen &rng_x, rng_gen &rng_y,
                                     rng_gen &rng_z, std::mt19937 &re) {
  pos.x = rng_x(re);
  pos.y = rng_y(re);
  pos.z = rng_z(re);
}

__host__ __device__ double particle_t::interact(particle_t const &rhs,
                                                double const cosmax,
                                                double const epsilon) const {
  double3 dist = normalize((double3){
      .x = pos.x - rhs.pos.x,
      .y = pos.y - rhs.pos.y,
      .z = pos.z - rhs.pos.z,
  });

  for (size_t i = 0; i < patch_count; i++) {
    patch_t const &p = patches[i];
    double3 p_pos = normalize((double3){
        p.pos.y,
        p.pos.z,
        p.pos.w,
    });
    double p_cos = -dot(p_pos, dist);

    if (p_cos < cosmax) {
      continue;
    }

    for (size_t j = 0; j < rhs.patch_count; j++) {
      patch_t const &q = rhs.patches[j];
      double3 q_pos = normalize((double3){
          q.pos.y,
          q.pos.z,
          q.pos.w,
      });
      double q_cos = dot(q_pos, dist);
      if (q_cos < cosmax) {
        continue;
      }

      return epsilon;
    }
  }

  return 0;
}

void particle_t::add_patch(patch_t const &p) {
  if (patch_count >= DEFAULT_CAPACITY) {
    return;
  }
  patches[patch_count] = p;
  patch_count++;
}
