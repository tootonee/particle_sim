#include "particle.h"
#include <cstdio>

int main() {
  printf("Hi there!\n");

  Particle p1{
      1.0F, {}, {}, 0L, 0L, nullptr,
  };
  Particle p2{
      1.0F, {}, {}, 0L, 0L, nullptr,
  };

  printf("P1 intersects P2: %i\n", p1.intersects(p2));

  return 0;
}
