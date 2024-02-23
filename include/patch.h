#pragma once

#ifndef PATCH_HPP
#define PATCH_HPP

struct __align__(32) patch_t {
    double radius{};
    double3 pos{};
};

#endif
