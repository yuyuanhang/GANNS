#pragma once

#include "structure_on_device.h"

__global__ 
void AggragateForwardEdges (KernelPair<float, int>* neighbors, Edge* edge_list, int* flags, int total_num_of_points, int num_of_visited_points_one_batch, int offset_shift) {
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size_of_warp = 32;

    extern __shared__ KernelPair<float, int>* shared_memory_space_afe[];
    KernelPair<float, int>* neighbors_array = (KernelPair<float, int>*)shared_memory_space_afe;


    int first_position_of_edges = flags[b_id];
    int last_position_of_edges = flags[b_id + 1];
    
    int num_of_valid_edges = last_position_of_edges - first_position_of_edges;
    
    int target_point_id = edge_list[first_position_of_edges].target_point;

    int step_id;
    int substep_id;
    KernelPair<float, int> temporary_neighbor;

    if (target_point_id < total_num_of_points) {

        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;

            if (unrollt_id < num_of_visited_points_one_batch) {
                neighbors_array[unrollt_id] = (neighbors + (target_point_id << offset_shift))[unrollt_id];
            }
        }
        
        for (int i = 0; i < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;

            if (unrollt_id < num_of_valid_edges && unrollt_id < num_of_visited_points_one_batch) {
                neighbors_array[num_of_visited_points_one_batch + num_of_visited_points_one_batch - unrollt_id - 1].second = edge_list[first_position_of_edges + unrollt_id].source_point;
                neighbors_array[num_of_visited_points_one_batch + num_of_visited_points_one_batch - unrollt_id - 1].first = edge_list[first_position_of_edges + unrollt_id].distance;
            } else if (unrollt_id >= num_of_valid_edges && unrollt_id < num_of_visited_points_one_batch) {
                neighbors_array[num_of_visited_points_one_batch + num_of_visited_points_one_batch - unrollt_id - 1].second = total_num_of_points;
                neighbors_array[num_of_visited_points_one_batch + num_of_visited_points_one_batch - unrollt_id - 1].first = Max;
            }
        }

        int num_of_candidates = num_of_visited_points_one_batch;
        int length_of_compared_list = num_of_visited_points_one_batch;

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

        
        for (int i = 0; i < (num_of_candidates + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;
            
            if (unrollt_id < num_of_candidates) {
                (neighbors + (target_point_id << offset_shift))[unrollt_id] = neighbors_array[unrollt_id];
            }
        }

    }
    
}
