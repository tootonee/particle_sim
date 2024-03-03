#pragma once

#ifndef PATCH_H
#define PATCH_H

#include <cuda_runtime_api.h>

struct __align__(32) patch_t {
  double radius{};
  double4 pos{};
  size_t particle_idx{};
};

#endif
