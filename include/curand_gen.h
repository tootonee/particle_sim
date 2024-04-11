#pragma once

#ifndef CURAND_GEN
#define CURAND_GEN

#include <cstdint>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

__global__ void setup_kernel(curandState *state, uint64_t seed) {
  int thr_id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, thr_id, 0, &state[thr_id]);
}

__global__ void generate_random_numbers_kernel(curandState *gen_states,
                                               double *numbers) {
  int thr_id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState *localState = gen_states + thr_id;
  numbers[thr_id * 2 + 0] = curand_uniform(localState);
  numbers[thr_id * 2 + 1] = curand_uniform(localState);
}

class curand_gen_t {
public:
  curandState *gen_states;
  cudaStream_t genStream;
  size_t n_blocks;
  size_t n_threads;
  double *devFloats;

  curand_gen_t(size_t blocks, size_t threads)
      : n_blocks(blocks), n_threads(threads) {
    cudaMalloc(&gen_states, blocks * threads * sizeof(curandState));
    cudaMalloc(&devFloats, 2 * blocks * threads * sizeof(double));
    cudaStreamCreateWithFlags(&genStream, cudaStreamNonBlocking);
    setup_kernel<<<blocks, threads, 0, genStream>>>(gen_states, time(nullptr));
  }

  ~curand_gen_t() {
    cudaFree(gen_states);
    cudaFree(devFloats);
  }

  void generate_random_numbers() {
    generate_random_numbers_kernel<<<n_blocks, n_threads, 0, genStream>>>(
        gen_states, devFloats);
  }

  void copyToHost(double *nums) const {
    cudaMemcpy(nums, devFloats, sizeof(double) * n_blocks * n_threads,
               cudaMemcpyDeviceToHost);
  }
};

#endif
