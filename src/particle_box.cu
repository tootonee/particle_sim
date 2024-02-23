#include "particle_box.h"


void particle_box_init_host(particle_box_t &p,
    size_t capacity) {
    p.particles = new particle_t[capacity];
    p.capacity = capacity;
}

__device__ void particle_box_init_device(particle_box_t &p,
    size_t capacity) {
    p.capacity = capacity;
    cudaMalloc(&p.particles, sizeof(particle_t) * capacity);
}

void particle_box_realloc_host(particle_box_t &p,
    size_t capacity) {
    if (capacity <= p.capacity) {
        return;
    }
    particle_t *new_particles = new particle_t[capacity];
    p.capacity = capacity;
    for (size_t idx = 0; idx < p.particle_count; idx++) {
        new_particles[idx] = p.particles[idx];
    }
    delete[] p.particles;
    p.particles = new_particles;
}

__device__ void particle_box_realloc_device(particle_box_t &p,
    size_t capacity) {
    if (capacity <= p.capacity) {
        return;
    }
    particle_t *new_particles;
    cudaMalloc(&new_particles, sizeof(particle_t) * capacity);
    for (size_t idx = 0; idx < p.particle_count; idx++) {
        new_particles[idx] = p.particles[idx];
    }
    cudaFree(p.particles);
    p.particles = new_particles;
}

void particle_box_add_particle_host(particle_box_t &box, double radius) {
    if (box.capacity >= box.particle_count) {
        particle_box_realloc_host(box, box.particle_count * 2);
    }

    bool intersects = true;
    particle_t *p= box.particles + box.particle_count;
    p->radius = radius;
    p->valid = true;
    particle_init_host(box.particles[box.particle_count]);
    do {
        intersects = false;
        random_particle_pos(*p, box.dimensions);

        for (size_t i = 0; i < box.particle_count; i++) {
            if(particle_intersects(box.particles[i], *p)) {
                intersects = true;
                break;
            }
        }
    } while (intersects);
    box.particle_count++;
}