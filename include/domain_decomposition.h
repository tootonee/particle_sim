#ifndef DOMAIN_DECOMPOSITION
#define DOMAIN_DECOMPOSITION

#include <unordered_map>
#include <vector>
#include "cell_view.h"
#include <vector_types.h>
#include <safe_queue.h>
#include <conf.h>
#include <mutex>


struct Task
{
    cell_view_t &view;
    size_t particle_idx;
    size_t iters;
    size_t sequential_index;
    std::vector<double> hostFloats;
    double temp;
    bool end;
};

void worker(ThreadSafeQueue<Task>& tasksQueue, ThreadSafeQueue<double>& resultsQueue,  std::mutex& mutex){
    while(true){
        std::shared_ptr<Task> taskPtr = tasksQueue.wait_and_pop();

        const Task& task = *taskPtr;
        if (task.end) {
            break;
        }
        size_t const r_idx = task.sequential_index * 6;
        size_t const p_idx = task.particle_idx;
        double const x = task.hostFloats[r_idx + 1] - 0.5;
        double const y = task.hostFloats[r_idx + 2] - 0.5;
        double const z = sqrt(1 - x * x - y * y);
        double3 const offset = {
            .x = x,
            .y = y,
            .z = z,
        };
        double cell_length = task.view.cell_size.x;
        double max_step = cell_length - 2 * RADIUS;
        double3 new_pos;
        {
            std::lock_guard<std::mutex> lock(mutex);
            new_pos = task.view.try_random_particle_disp(p_idx, offset, max_step);

        }
        
        double const prob_rand = task.hostFloats[r_idx + 4];
        double angle = task.hostFloats[r_idx + 5] * M_PI;
        double4 rotation =
            particle_t::random_particle_orient(angle, (task.sequential_index + task.iters) % 3);
        double energy;
        {
            std::lock_guard<std::mutex> lock(mutex);  // Locks the mutex and releases it when the scope ends
            energy = task.view.try_move_particle(p_idx, new_pos, rotation, prob_rand, TEMPERATURE);
        }
        resultsQueue.push(energy);
    }
}

std::unordered_map<int, std::vector<size_t>> domain_division (cell_view_t &view){

    size_t cells_per_axis = view.cells_per_axis;
    double axis_lenght_cell = view.cell_size.x;
    double total_length = view.box.dimensions.x;
    double length_to_centre = axis_lenght_cell / 2;
    double3 start_points[8];
    size_t idx = 0;
    for (size_t x = 0; x <= 1; x++){
        for (size_t y = 0; y <= 1; y++){
        for (size_t z = 0; z <= 1; z++){
            start_points[idx] = {length_to_centre + x * axis_lenght_cell, length_to_centre + y * axis_lenght_cell, length_to_centre + z * axis_lenght_cell};
            idx++;
        }
        }
    }
    std::unordered_map<int, std::vector<size_t>> domain_decomposition;
    for (size_t type = 0; type < 8; type++){
        for (size_t x_axis = 0; x_axis < cells_per_axis; x_axis += 2){
        for (size_t y_axis = 0; y_axis < cells_per_axis; y_axis += 2){
            for (size_t z_axis = 0; z_axis < cells_per_axis; z_axis += 2){
            double3 coordinates = {start_points[type].x + x_axis*axis_lenght_cell, start_points[type].y + y_axis*axis_lenght_cell, start_points[type].z + z_axis*axis_lenght_cell};
            if (coordinates.x > total_length || coordinates.y > total_length || coordinates.z > total_length){
                continue;
            }
            domain_decomposition[type].push_back(view.get_cell_idx(coordinates));
            }
        }
        }
    }
    return domain_decomposition;
}

#endif 