#pragma once
#include "structure_on_device.h"

__global__ 
void SortNeighborsonLocalGraph(KernelPair<float, int>* old_neighbors, int total_num_of_points, float* d_data, int num_of_points_one_batch, 
                                    int num_of_initial_neighbors, int num_of_final_neighbors, KernelPair<float, int>* distance_matrix){

    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int b_dim = gridDim.x;

    extern __shared__ KernelPair<float, int> shared_memory_space[];
    KernelPair<float, int>* neighbors_array = shared_memory_space;

    int num_of_iterations = (total_num_of_points + b_dim - 1) / b_dim;
    int step_id = 1;
    int substep_id = 1;
    int size_of_warp = 32;
    KernelPair<float, int> temporary_neighbor;

    for (int i = 0; i < num_of_iterations; i++) {
 
        int crt_point_id = i * b_dim + b_id;
        
        if(crt_point_id >= total_num_of_points){
            break;
        }
        
        KernelPair<float, int>* crt_distances = distance_matrix + crt_point_id * num_of_points_one_batch;

        //compute neighbors of small world graph from distances of pairs with other nodes in the local graph
        int num_of_sequences_with_fixed_size = (crt_point_id % num_of_points_one_batch + num_of_initial_neighbors - 1) / num_of_initial_neighbors;

        //initialise current best neighbors
        for(int j = 0; j < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; j++){
            int unrollt_id = j * size_of_warp + t_id;
            if (unrollt_id < num_of_points_one_batch) {
                if (unrollt_id < crt_point_id % num_of_points_one_batch) {
                    neighbors_array[unrollt_id] = crt_distances[unrollt_id];
                } else {
                    neighbors_array[unrollt_id].first = Max;
                    neighbors_array[unrollt_id].second = total_num_of_points;
                }
            }
        }

        int num_of_candidates = num_of_initial_neighbors;

step_id = 1;
substep_id = 1;

for (; step_id <= num_of_candidates / 2; step_id *= 2) {
    substep_id = step_id;

    for (; substep_id >= 1; substep_id /= 2) {
        for (int temparory_id = 0; temparory_id < (num_of_candidates/2+size_of_warp-1) / size_of_warp; temparory_id++) {
            int unrollt_id = ((t_id + size_of_warp * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
            
            if (unrollt_id < num_of_candidates) {
                if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                    if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                } else {
                    if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                }
            }
        }
    }
}


        for (int j = 1; j < num_of_sequences_with_fixed_size; j++) {

            //load other neighbors
            for(int l = 0; l < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; l++){
                int unrollt_id = j * num_of_initial_neighbors + l * size_of_warp + t_id;
                int unroll_local_t_id = l * size_of_warp + t_id;

                if (unrollt_id < crt_point_id % num_of_points_one_batch && unroll_local_t_id < num_of_initial_neighbors) {
                    neighbors_array[unroll_local_t_id + num_of_initial_neighbors] = crt_distances[unrollt_id];
                } else {
                    if (unroll_local_t_id < num_of_initial_neighbors) {
                        neighbors_array[unroll_local_t_id + num_of_initial_neighbors].first = Max;
                        neighbors_array[unroll_local_t_id + num_of_initial_neighbors].second = total_num_of_points;
                    }
                }
            }

            int num_of_visited_points_one_batch = num_of_initial_neighbors;
            num_of_candidates = num_of_initial_neighbors;
            int length_of_compared_list = num_of_initial_neighbors;

step_id = 1;
substep_id = 1;

for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
    substep_id = step_id;

    for (; substep_id >= 1; substep_id /= 2) {
        for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch/2+size_of_warp-1) / size_of_warp; temparory_id++) {
            int unrollt_id = num_of_candidates + ((t_id + size_of_warp * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
            
            if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                    if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                } else {
                    if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                }
            }
        }
    }
}


for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_warp - 1) / size_of_warp; temparory_id++) {
    int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_warp * temparory_id;
    if (unrollt_id < num_of_candidates) {
        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
            temporary_neighbor = neighbors_array[unrollt_id];
            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
            neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
        }
    }
}

step_id = num_of_candidates / 2;
substep_id = num_of_candidates / 2;
for (; substep_id >= 1; substep_id /= 2) {
    for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_warp - 1) / size_of_warp; temparory_id++) {
        int unrollt_id = ((t_id + size_of_warp * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
        if (unrollt_id < num_of_candidates) {
            if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                }
            } else {
                if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                }
            }
        }
    }
}


        }

        KernelPair<float, int>* crt_old_neighbors = old_neighbors + (crt_point_id * num_of_final_neighbors);
        
        for (int j = 0; j < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; j++) {
            int unrollt_id = j * size_of_warp + t_id;
            if (unrollt_id < num_of_initial_neighbors) {
                crt_old_neighbors[unrollt_id] = neighbors_array[unrollt_id];
            }
        }

        for (int j = 0; j < (num_of_initial_neighbors + size_of_warp - 1) / size_of_warp; j++) {
            int unrollt_id = j * size_of_warp + t_id;
            if (unrollt_id < num_of_initial_neighbors) {
                crt_old_neighbors[num_of_initial_neighbors + unrollt_id].first = Max;
                crt_old_neighbors[num_of_initial_neighbors + unrollt_id].second = total_num_of_points;
            }
        }
    }
    
}
