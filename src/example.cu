#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <map>
#include <stdint.h>

#include <stdio.h>

void ShannonEntropy(int *data, int N, int &min, int &max, float &entropy);

__global__ void setup_kernel(curandState *state, uint64_t seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate_randoms(curandState *globalState, float *randoms) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = globalState[tid];
  randoms[tid * 2 + 0] = curand_uniform(&localState);
  randoms[tid * 2 + 1] = curand_uniform(&localState);
}

int main() {
  printf("\nTwoStepRandom\n");
  int threads = 256;
  int blocks = 5120;
  int threadCount = blocks * threads;
  int N = blocks * threads * 2;

  curandState *dev_curand_states;
  float *randomValues;
  float *host_randomValues;
  int *host_int;

  float time_elapsed_setup;
  float time_elapsed;
  cudaEvent_t startTime;
  cudaEvent_t stopTime;
  cudaStream_t computeStream;

  // Init host memory
  host_randomValues = (float *)malloc(N * sizeof(float));
  host_int = (int *)malloc(N * sizeof(float));

  // Init device memory
  cudaMalloc(&dev_curand_states, threadCount * sizeof(curandState));
  cudaMalloc(&randomValues, N * sizeof(float));

  cudaEventCreate(&startTime);
  cudaEventCreate(&stopTime);
  cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking);

  //  ----- Setup seeds -----
  cudaEventRecord(startTime, computeStream);

  setup_kernel<<<blocks, threads, 0, computeStream>>>(dev_curand_states,
                                                      time(NULL));

  cudaEventRecord(stopTime, computeStream);
  cudaEventSynchronize(stopTime);
  cudaEventElapsedTime(&time_elapsed_setup, startTime, stopTime);

  // ----- Generate random numbers -----
  cudaEventRecord(startTime, computeStream);

  // Needs both read and write from global memory
  generate_randoms<<<blocks, threads, 0, computeStream>>>(dev_curand_states,
                                                          randomValues);

  cudaEventRecord(stopTime, computeStream);
  cudaEventSynchronize(stopTime);

  cudaEventElapsedTime(&time_elapsed, startTime, stopTime);

  // ----- Concluding Steps -----

  cudaMemcpy(host_randomValues, randomValues, N * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Convert floats to ints for the shannnon entropy function
  for (int i = 0; i < N; ++i) {
    // Print a few values out
    if (i < 8) {
      printf("%.3f, ", host_randomValues[i]);
    }

    host_int[i] = (int)(host_randomValues[i] * 10000.0f);
  }

  printf("\n");
  printf("Elapsed time setup    %9.3f\n", time_elapsed_setup);
  printf("Elapsed time generate %9.3f\n", time_elapsed);

  int min, max;
  float entropy;
  ShannonEntropy(host_int, N, min, max, entropy);
  printf("Shannon Entropy <%6.3f>\n", entropy);

  cudaFree(dev_curand_states);
  cudaFree(randomValues);
  free(host_randomValues);
  free(host_int);

  return 0;
}

void ShannonEntropy(int *data, int N, int &min, int &max, float &entropy) {
  entropy = 0; // Init
  min = UINT_MAX;
  max = 0;

  std::map<int, long> counts;
  typename std::map<int, long>::iterator it;

  for (int dataIndex = 0; dataIndex < N; dataIndex++) {
    int dValue = data[dataIndex];
    if (dValue < min) {
      min = dValue;
    }
    if (dValue > max) {
      max = dValue;
    }
    counts[dValue]++;
  }

  it = counts.begin();
  while (it != counts.end()) {
    float p_x = (float)it->second / N;
    if (p_x > 0)
      entropy -= (float)(p_x * log(p_x) / log(2));
    it++;
  }
}
