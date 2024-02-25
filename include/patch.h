#pragma once

#ifndef PATCH_H
#define PATCH_H

struct __align__(32) patch_t {
    double radius{};
    double3 pos{};
};

#endif
