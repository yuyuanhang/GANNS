#ifndef DEVICE_FUNC_H
#define DEVICE_FUNC_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "type.h"

#define DIM PLACE_HOLDER_DIM
#define WS 32

__device__ void bitonic_sort(int lane_id, int k, NBR* arr, bool asc) {
    int step = 1;
    int substep = 1;
    NBR temp_nbr;

    for (; step <= k / 2; step *= 2) {

        substep = step;

        for (; substep >= 1; substep /= 2) {

            for (int i = 0; i < (k / 2 + WS - 1) / WS; i++) {

                int g_id = lane_id + WS * i;

                int l_id = (g_id / substep) * 2 * substep + g_id % substep;
            
                if (l_id + substep < k) {
                    bool up = ((g_id / step) % 2 == 0) ^ (!asc);

                    if ((up && arr[l_id].d > arr[l_id + substep].d) ||
                        (!up && arr[l_id].d < arr[l_id + substep].d)) {
                        temp_nbr = arr[l_id];
                        arr[l_id] = arr[l_id + substep];
                        arr[l_id + substep] = temp_nbr;
                    }
                }
            }
        }
    }
}



__device__ void bitonic_merge(int lane_id, int k, int M, NBR* arr) {
    NBR temp_nbr;

    for (int i = 0; i < (k + WS - 1) / WS; i++) {

        int l_id = M - k + lane_id + WS * i;

        if (l_id < M) {
            if (arr[l_id].d > arr[l_id + k].d) {
                temp_nbr = arr[l_id];
                arr[l_id] = arr[l_id + k];
                arr[l_id + k] = temp_nbr;
            }
        }
    }
    
    int step = M / 2;
    int substep = M / 2;

    for (; substep >= 1; substep /= 2) {

        for (int i = 0; i < (M / 2 + WS - 1) / WS; i++) {

            int g_id = lane_id + WS * i;

            int l_id = (g_id / substep) * 2 * substep + g_id % substep;

            if (l_id + substep < M) {
                bool up = ((g_id / step) % 2 == 0);

                if ((up && arr[l_id].d > arr[l_id + substep].d) ||
                    (!up && arr[l_id].d < arr[l_id + substep].d)) {
                    temp_nbr = arr[l_id];
                    arr[l_id] = arr[l_id + substep];
                    arr[l_id + substep] = temp_nbr;
                }
            }
        }
    }
}

__device__ void bitonic_merge_triplet(int lane_id, int k, int M, NBR* arr, int* f) {
    NBR temp_nbr;
    int temp_f;

    for (int i = 0; i < (k + WS - 1) / WS; i++) {

        int l_id = M - k + lane_id + WS * i;

        if (l_id < M) {
            if (arr[l_id].d > arr[l_id + k].d) {
                temp_nbr = arr[l_id];
                arr[l_id] = arr[l_id + k];
                arr[l_id + k] = temp_nbr;

                temp_f = f[l_id];
                f[l_id] = f[l_id + k];
                f[l_id + k] = temp_f;
            }
        }
    }
    
    int step = M / 2;
    int substep = M / 2;

    for (; substep >= 1; substep /= 2) {

        for (int i = 0; i < (M / 2 + WS - 1) / WS; i++) {

            int g_id = lane_id + WS * i;

            int l_id = (g_id / substep) * 2 * substep + g_id % substep;

            if (l_id + substep < M) {
                bool up = ((g_id / step) % 2 == 0);

                if ((up && arr[l_id].d > arr[l_id + substep].d) ||
                    (!up && arr[l_id].d < arr[l_id + substep].d)) {
                    temp_nbr = arr[l_id];
                    arr[l_id] = arr[l_id + substep];
                    arr[l_id + substep] = temp_nbr;

                    temp_f = f[l_id];
                    f[l_id] = f[l_id + substep];
                    f[l_id + substep] = temp_f;
                }
            }
        }
    }
}

__device__ float compute_dist(int lane_id, int id, int nbr_id, float* d) {
    float dist = 0.0f;
#if USE_DIST_CS_
    float p_l2 = 0.0f;
    float q_l2 = 0.0f;
#endif
    for (int k = 0; k + lane_id < DIM; k += 32) {
        int dim_idx = k + lane_id;
        float p = d[id * DIM + dim_idx];
        float q = d[nbr_id * DIM + dim_idx];
    
#if USE_DIST_L2_
        float delta = (p - q) * (p - q);
#elif USE_DIST_IP_
        float delta = p * q;
#elif USE_DIST_CS_
        float delta = p * q;
        float norm_p = p * p;
        float norm_q = q * q;
#endif

        dist += delta;
#if USE_DIST_CS_
        p_l2+= norm_p;
        q_l2+= norm_q;
#endif
    }
        
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
    
#if USE_DIST_CS_
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 16);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 8);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 4);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 2);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 1);

    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 16);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 8);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 4);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 2);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 1);
#endif

#if USE_DIST_L2_
    dist = sqrt(dist);
#elif USE_DIST_IP_
    dist = -dist;
#elif USE_DIST_CS_
    p_l2 = sqrt(p_l2);
    q_l2 = sqrt(q_l2);
    dist = dist / (p_l2 * q_l2);
    if (lane_id == 0) {
        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }
    }
#endif
    return dist;
}

__device__ void binary_search(int lane_id, int k, int M, NBR* arr) {
    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
        if (l_id < k) {
            float d = arr[M + l_id].d;

            int is_find = -1;
            int l = 0;
            int r = M - 1;
            int m;
            while (l <= r) {
                m = (r + l) / 2;
                if (d == arr[m].d) {
                    if (m > 0 && arr[m - 1].d == arr[m].d) {
                        r = m - 1;
                    } else {
                        is_find = m;
                        break;
                    }
                } else if (d < arr[m].d) {
                    r = m - 1;
                } else {
                    l = m + 1;
                }
            }

            if (is_find != -1) {
                if (arr[M + l_id].nbr == arr[is_find].nbr) {
                    arr[M + l_id].d = Max;
                } else {
                    is_find++;
    
                    while (arr[M + l_id].d == arr[is_find].d) {
                        if (arr[M + l_id].nbr == arr[is_find].nbr) {
                            arr[M + l_id].d = Max;
                            break;
                        }
                        is_find++;
                    }
                }
            }
        }
    }
}

__device__ float compute_dist_q(int lane_id, int id, int nbr_id, float* d, float *q_vec) {
    float dist = 0.0f;
#if USE_DIST_CS_
    float p_l2 = 0.0f;
    float q_l2 = 0.0f;
#endif
    for (int k = 0; k + lane_id < DIM; k += 32) {
        int dim_idx = k + lane_id;
        float p = q_vec[id * DIM + dim_idx];
        float q = d[nbr_id * DIM + dim_idx];
    
#if USE_DIST_L2_
        float delta = (p - q) * (p - q);
#elif USE_DIST_IP_
        float delta = p * q;
#elif USE_DIST_CS_
        float delta = p * q;
        float norm_p = p * p;
        float norm_q = q * q;
#endif

        dist += delta;
#if USE_DIST_CS_
        p_l2+= norm_p;
        q_l2+= norm_q;
#endif
    }
        
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
    
#if USE_DIST_CS_
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 16);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 8);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 4);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 2);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 1);

    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 16);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 8);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 4);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 2);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 1);
#endif

#if USE_DIST_L2_
    dist = sqrt(dist);
#elif USE_DIST_IP_
    dist = -dist;
#elif USE_DIST_CS_
    p_l2 = sqrt(p_l2);
    q_l2 = sqrt(q_l2);
    dist = dist / (p_l2 * q_l2);
    if (lane_id == 0) {
        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }
    }
#endif
    return dist;
}

#endif