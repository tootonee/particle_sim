#pragma once

#ifndef VECTOR_HPP
#define VECTOR_HPP

struct vec_3d {
    double x{};
    double y{};
    double z{};
};

__host__ __device__ inline double vec_3d_distance_squared(vec_3d const lhs,
                                                          vec_3d const rhs) {
    return (lhs.x - rhs.x) * (lhs.x - rhs.x) +
           (lhs.y - rhs.y) * (lhs.y - rhs.y) +
           (lhs.z - rhs.z) * (lhs.z - rhs.z);
}

__host__ __device__ inline double vec_3d_dot(vec_3d const lhs,
                                             vec_3d const rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ inline vec_3d vec_3d_scale(double const scale,
                                               vec_3d const rhs) {
    return {
        rhs.x * scale,
        rhs.y * scale,
        rhs.x * scale,
    };
}

__host__ __device__ inline vec_3d vec_3d_add(vec_3d const lhs,
                                             vec_3d const rhs) {
    return {
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z,
    };
}

struct vec_4d {
    double x{};
    double y{};
    double z{};
    double w{};
};

__host__ __device__ inline double vec_4d_distance_squared(vec_4d const lhs,
                                                          vec_4d const rhs) {
    return (lhs.x - rhs.x) * (lhs.x - rhs.x) +
           (lhs.y - rhs.y) * (lhs.y - rhs.y) +
           (lhs.z - rhs.z) * (lhs.z - rhs.z) +
           (lhs.w - rhs.w) * (lhs.w - rhs.w);
}

__host__ __device__ inline double vec_4d_dot(vec_4d const lhs,
                                             vec_4d const rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ inline vec_4d scale(double const scale, vec_4d const rhs) {
    return {
        rhs.x * scale,
        rhs.y * scale,
        rhs.x * scale,
        rhs.w * scale,
    };
}

__host__ __device__ inline vec_4d vec_4d_add(vec_4d const lhs,
                                             vec_4d const rhs) {
    return {
        lhs.x + rhs.x,
        lhs.y + rhs.y,
        lhs.z + rhs.z,
        lhs.w + rhs.w,
    };
}

#endif
