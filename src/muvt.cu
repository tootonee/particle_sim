#include "cell_view.h" //for add_particle_random_pos
#include "conf.h" //RADIUS
#include <cmath> //sqrt
#include <iostream>


//const double mu = 0.0000000000000000000000000000000001; //p
const double h = 6.62607015e-34; //planck
const double mass = 9.11e-30; //kg -- p
const double boltzmann = 1.380649e-23;
constexpr double temperature_kelvin = TEMPERATURE+273.15;
const double beta = 1.0 / (boltzmann * temperature_kelvin);
// const double z = exp(beta * mu);
const double z = 0.317;
const double wavelength = h / sqrt(2 * M_PI * mass * boltzmann * temperature_kelvin);
constexpr double V = 3.0 / 4.0 * M_PI * (RADIUS * RADIUS * RADIUS);


double adding_probability(size_t N, double energy) {

    // std::cout << "V: " << V << std::endl;
    // std::cout << "boltzman: " << boltzmann << std::endl;
    // std::cout << "z: " << z << std::endl;
    // std::cout << "wavelength: " << wavelength << std::endl;
    // std::cout << "N: " << N << std::endl;
    // std::cout << "-beta: " << -beta << std::endl;
    // std::cout << "energy: " << energy << std::endl;
    // return V * z / (wavelength * wavelength * wavelength * N) * exp(-beta * energy);
    std::random_device r;
    std::mt19937 re(r());
    std::uniform_real_distribution<double> unif_r(0.0, 1.0);
    double random_num = unif_r(re);
    return random_num;
}


double cell_view_t::add_particle_muvt(rng_gen &unif_x,
                        rng_gen &unif_y, rng_gen &unif_z,
                        std::mt19937 &re) {

    std::cout << "initial number of particles: " << box.particle_count << std::endl;

    if (box.capacity <= box.particle_count) {
        box.realloc(box.capacity * 2);
    }
    particle_t p{};
    p.radius = RADIUS;
    p.idx = box.particle_count;

    do {
        p.random_particle_pos(unif_x, unif_y, unif_z, re);
    } while (particle_intersects(p) || !add_particle(p));

    box.particles[box.particle_count] = p;
    box.particle_count++;
    //add_particle_random_pos(RADIUS, unif_x, unif_y, unif_z, re); //arguments
    
    //add 6 next time
    box.particles[box.particle_count-1].add_patch({
        .radius = 0.05,
        .pos = {1, 1, 0, 0},
        });
    box.particles[box.particle_count-1].add_patch({
        .radius = 0.05,
        .pos = {1, -1, 0, 0},
        });
    
    double new_energy = particle_energy_yukawa(p) + particle_energy_patch(p);
    
    std::uniform_real_distribution<double> unif_r(0.0, 1.0);
    double random_num = unif_r(re);
    double probability = adding_probability(box.particle_count, new_energy);
    
    std::cout << "+1 number of particles: " << box.particle_count << std::endl;
    std::cout << "probability: " << probability << std::endl;
    std::cout << "random number: " << random_num << std::endl;

    if (random_num >= probability) {
        //we need remove particle
        std::cout << "-----removing-----" << std::endl;
        remove_particle_from_box(p);
        std::cout << "-1 number of particles: " << box.particle_count << std::endl;
        return 0;
    }
    std::cout << "final number of particles: " << box.particle_count << std::endl;
    return new_energy;
}


double cell_view_t::remove_particle_muvt (rng_gen &unif_x,
    rng_gen &unif_y, rng_gen &unif_z,
    std::mt19937 &re) {
        
    }