#pragma once

#ifndef SPAITAL_HASH_HPP
#define SPAITAL_HASH_HPP

#include "point.h"

struct __align__(64) hash_entry {
    double3 pos{};
    point_t **points{};
    size_t point_count{};
    size_t point_capacity{};
};

struct __align__(32) spatial_hash {
    hash_entry *entries{};
    size_t entry_count{};
    size_t entry_capacity{};
    double entry_radius;
}

spatial_hash hash_new();
void add_point(spatial_hash &hash, point_t *p);

#endif