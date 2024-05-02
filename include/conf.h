#pragma once

#ifndef CONF
#define CONF

constexpr size_t ITERATIONS = 100;
constexpr size_t ITERATIONS_PER_EXPORT = 100;
constexpr size_t ITERATIONS_PER_GRF_EXPORT = 2500;
constexpr double TEMPERATURE = 60;
constexpr double MAX_STEP = 0.5;
constexpr size_t THREADS_PER_BLOCK = 256;
constexpr double RADIUS = 0.5;

#endif
