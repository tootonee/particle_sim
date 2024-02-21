#include "particle.h"
#include <cstdio>

int main() {
  printf("Hi there!\n");

  Particle p1{.radius = 1.0F};
  Particle p2{.radius = 1.0F};

  printf("P1 intersects P2: %i\n", p1.intersects(p2));

  return 0;
}
