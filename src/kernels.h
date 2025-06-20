#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <curand.h>
#include <curand_kernel.h>
#include "type.h"
#include "device_func.h"

__global__
void DistMat(float* d, 
    int n_d, 
    int offset,
    int k, 
    int local_g_sz,
    NBR* dist_mat)
{
    int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;
    
    for (int i = 1; i < local_g_sz; i += n_warp) {
        int l_id = i + warp_id; // local_id
        int g_id = b_id * local_g_sz + l_id; // global_id

        if (l_id >= local_g_sz) break;
        if (g_id >= n_d) break;
        
        NBR* dist_arr = dist_mat + g_id * k;

        for (int l_nbr_id = offset; l_nbr_id < offset + k; l_nbr_id++) {
            int g_nbr_id = b_id * local_g_sz + l_nbr_id;

            if(g_nbr_id >= n_d || l_nbr_id >= l_id) break;
            
            float dist = compute_dist(lane_id, g_id, g_nbr_id, d);

            if(lane_id == 0){
                dist_arr[l_nbr_id - offset].d = dist;
                dist_arr[l_nbr_id - offset].nbr = g_nbr_id;
            }
        }
    }
}



__global__ 
void SortNBR_LG(NBR* local_nbr, 
    int n_d, 
    float* d, 
    int local_g_sz, 
    int k, 
    int s_bits,
    int offset, 
    NBR* dist_mat)
{
    int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;
    int b_dim = gridDim.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * k * 2;
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    int n_t = (n_d + b_dim * n_warp - 1) / (b_dim * n_warp);

    for (int i = 0; i < n_t; i++) {
 
        int id = i * b_dim * n_warp + b_id * n_warp + warp_id;
        
        if(id >= n_d) break;
        
        NBR* dist_seg = dist_mat + id * k;

        NBR* local_nbr_seg = local_nbr + (id << s_bits);

        // load the first segment
        for(int j = 0; j < (k + WS - 1) / WS; j++){
            int l_id = j * WS + lane_id;
            if (l_id < k) {
                if (l_id < id % local_g_sz) {
                    s_nbr[l_id] = local_nbr_seg[l_id];
                } else {
                    s_nbr[l_id] = init_value;
                }
            }
        }

        // load other segments
        for(int j = 0; j < (k + WS - 1) / WS; j++){
            int l_id = j * WS + lane_id;

            if ((offset + l_id) < (id % local_g_sz) && l_id < k) {
                s_nbr[l_id + k] = dist_seg[l_id];
            } else if (l_id < k) {
                s_nbr[l_id + k] = init_value;
            }
        }
        
        bitonic_sort(lane_id, k, s_nbr + k, false);

        bitonic_merge(lane_id, k, k, s_nbr);
        
        // save
        for (int j = 0; j < (k + WS - 1) / WS; j++) {
            int l_id = j * WS + lane_id;
            if (l_id < k) {
                local_nbr_seg[l_id] = s_nbr[l_id];
            }
        }
    }
}




__global__ 
void LocalGraphMergeK(NBR* nbr, 
    NBR* local_nbr, 
    int n_d,
    float* d,
    Edge* e,
    int g_id,
    int local_g_sz,
    int M, 
    int k,
    int s_bits) // 2 * k = 1 << s_bits
{
	int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * (2 * k + M);
    int* s_f = (int*)(sm + n_warp * (2 * k + M)) + warp_id * (2 * k + M);

    int id = g_id * local_g_sz + b_id * n_warp + warp_id;
    
    if (id >= n_d) {
        return;
    }

    NBR* nbr_seg = nbr + (id << s_bits);
    NBR* local_nbr_seg = local_nbr + (id << s_bits);
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    int g_f = 1, f_f = 0;

    for (int i = 0; i < (M + 2 * k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if (l_id < M + 2 * k) {
            s_f[l_id] = 0;
            s_nbr[l_id] = init_value;
        }
    }

    if (lane_id == 0) {
        s_nbr[0].nbr = 0;
    }

    int nbr_id = 0;
    
    float dist = compute_dist(lane_id, id, nbr_id, d);
    
    if (lane_id == 0) {
        s_nbr[nbr_id].d = dist;
    }

    while (g_f) {

        if (lane_id == 0) {
            s_f[f_f] = 0;
        }

        nbr_id = s_nbr[f_f].nbr;
        NBR* nbr_nbr = nbr + (nbr_id << s_bits);
        
        for (int i = 0; i < (2 * k + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < 2 * k) {
                s_nbr[M + l_id] = nbr_nbr[l_id];
                s_f[M + l_id] = 1;
            }
        }

        for (int i = 0; i < 2 * k; i++) {
            nbr_id = s_nbr[M + i].nbr;
            
            if (nbr_id >= n_d) {
                s_nbr[M + i].d = Max;
                continue;
            }
            
            dist = compute_dist(lane_id, id, nbr_id, d);
            
            if (lane_id == 0) {
                s_nbr[M + i].d = dist;
            }
        }

        binary_search(lane_id, 2 * k, M, s_nbr);

        bitonic_sort(lane_id, 2 * k, s_nbr + M, false);
        
        bitonic_merge_triplet(lane_id, 2 * k, M, s_nbr, s_f);

        for (int i = 0; i < (M + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;
            int f = 0;

            if(l_id < M){
                f = s_f[l_id];
            }

            f_f = __ballot_sync(FULL_MASK, f);

            if(f_f != 0){
                f_f = WS * i + __ffs(f_f) - 1;
                break;
            }else if(i == (M + WS - 1) / WS - 1){
                g_f = 0;
            }
        }
    }

    for (int i = 0; i < (2 * k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if(l_id < 2 * k){
            s_nbr[M + 2 * k - l_id - 1] = local_nbr_seg[l_id];
        }
    }

    bitonic_merge(lane_id, 2 * k, M, s_nbr);

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
        int e_l_id = b_id * n_warp + warp_id;
        init_value = s_nbr[l_id];
        
        if (l_id < k) {
            nbr_seg[l_id] = init_value;
            
            e[e_l_id * k + l_id].u = id;
            e[e_l_id * k + l_id].v = init_value.nbr;
            e[e_l_id * k + l_id].w = init_value.d;
        }
    }
}



__global__ 
void UpdatePrevLG(NBR* nbr, 
    Edge* e, 
    int* idx,
    int idx_sz, 
    int n_d,
    int k)
{
    int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;
    int g_id = b_id * n_warp + warp_id;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * 2 * k;
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    if (g_id < idx_sz) {
        int begin = idx[g_id];
        int end = idx[g_id + 1];
        int len = end - begin;
        int v = e[begin].v;

        for (int i = 0; i < (k + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < k) {
                s_nbr[l_id] = nbr[v * k + l_id];
            }
        }

        for (int i = 0; i < (k + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < len && l_id < k) {
                s_nbr[2 * k - l_id - 1].nbr = e[begin + l_id].u;
                s_nbr[2 * k - l_id - 1].d = e[begin + l_id].w;
            } else if (l_id >= len && l_id < k) {
                s_nbr[2 * k - l_id - 1] = init_value;
            }
        }

        bitonic_merge(lane_id, k, k, s_nbr);

        for (int i = 0; i < (k + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;
            
            if (l_id < k) {
                nbr[v * k + l_id] = s_nbr[l_id];
            }
        }
    }
}


__global__
void SearchK(float* d, 
    float* q,
    int* ann,
    int* g,
    int n_d,
    int n_q,
    int s_bits,
    int M,
    int k,
    int n_e)
{
	int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * ((1 << s_bits) + M);
    int* s_f = (int*)(sm + n_warp * ((1 << s_bits) + M)) + warp_id * ((1 << s_bits) + M);
    
    int id = b_id * n_warp + warp_id;
    int* ann_seg = ann + id * k;
    
    int g_f = 1;
    int f_f = 0;
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    if (id >= n_q) {
        return;
    }

    for (int i = 0; i < ((1 << s_bits) + M + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if (l_id < (1 << s_bits) + M) {
            s_f[l_id] = 0;
            s_nbr[l_id] = init_value;
        }
    }

    if (lane_id == 0) {
        s_nbr[0].nbr = 0;
    }

    int nbr_id = 0;
    
    float dist = compute_dist_q(lane_id, id, nbr_id, d, q);
    
    if (lane_id == 0) {
        s_nbr[0].d = dist;
    }
   	
    while (g_f) {

        if (lane_id == 0) {
            s_f[f_f] = 0;
        }
        
        nbr_id = s_nbr[f_f].nbr;
        int* nbr_nbr = g + (nbr_id << s_bits);

        for (int i = 0; i < ((1 << s_bits) + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < (1 << s_bits)) {
                s_nbr[M + l_id].nbr = nbr_nbr[l_id];
                s_f[M + l_id] = 1;
            }
        }

        for (int i = 0; i < (1 << s_bits); i++) {
            nbr_id = s_nbr[M + i].nbr;
            
            if (nbr_id >= n_d) {
                s_nbr[M + i].d = Max;
                continue;
            }
            
            dist = compute_dist_q(lane_id, id, nbr_id, d, q);
                
            if (lane_id == 0) {
                s_nbr[M + i].d = dist;
            }
        }
        
        binary_search(lane_id, (1 << s_bits), M, s_nbr);

        bitonic_sort(lane_id, (1 << s_bits), s_nbr + M, false);
        
        bitonic_merge_triplet(lane_id, (1 << s_bits), M, s_nbr, s_f);
        
        for (int i = 0; i < (n_e + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;
            int f = 0;

            if(l_id < n_e){
                f = s_f[l_id];
            }

            f_f = __ballot_sync(FULL_MASK, f);

            if(f_f != 0){
                f_f = WS * i + __ffs(f_f) - 1;
                break;
            }else if(i == (n_e + WS - 1) / WS - 1){
                g_f = 0;
            }
        }
    }

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
    
        if (l_id < k) {
            ann_seg[l_id] = s_nbr[l_id].nbr;
        }
    }
}

__global__
void SearchK_H(float* d, 
    float* q,
    int* ann,
    int** h_g,
    int n_d,
    int n_q,
    int s_bits,
    int M,
    int k,
    int n_e,
    int n_l) 
{
	int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * ((1 << s_bits) + M);
    int* s_f = (int*)(sm + n_warp * ((1 << s_bits) + M)) + warp_id * ((1 << s_bits) + M);
    
    int id = b_id * n_warp + warp_id;
    int* ann_seg = ann + id * k;
    
    int g_f = 1;
    int f_f = 0;
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    if (id >= n_q) {
        return;
    }

    for (int i = 0; i < ((1 << s_bits) + M + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if (l_id < (1 << s_bits) + M) {
            s_f[l_id] = 0;
            s_nbr[l_id] = init_value;
        }
    }

    if (lane_id == 0) {
        s_nbr[0].nbr = 0;
    }

    int nbr_id = 0;
    
    float dist = compute_dist_q(lane_id, id, nbr_id, d, q);
        
    if (lane_id == 0) {
        s_nbr[0].d = dist;
    }

    int l = 0;
   	
    while (g_f) {

        if (lane_id == 0) {
            s_f[f_f] = 0;
        }
        
        nbr_id = s_nbr[f_f].nbr;
        int* nbr_nbr = h_g[l] + (nbr_id << s_bits);

        for (int i = 0; i < ((1 << s_bits) + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < (1 << s_bits)) {
                s_nbr[M + l_id].nbr = nbr_nbr[l_id];
                s_f[M + l_id] = 1;
            }
        }

        for (int i = 0; i < (1 << s_bits); i++) {
            nbr_id = s_nbr[M + i].nbr;
            
            if (nbr_id >= n_d) {
                s_nbr[M + i].d = Max;
                continue;
            }
            
            dist = compute_dist_q(lane_id, id, nbr_id, d, q);
                
            if (lane_id == 0) {
                s_nbr[M + i].d = dist;
            }
        }
        
        binary_search(lane_id, (1 << s_bits), M, s_nbr);

        bitonic_sort(lane_id, (1 << s_bits), s_nbr + M, false);
        
        bitonic_merge_triplet(lane_id, (1 << s_bits), M, s_nbr, s_f);

        for (int i = 0; i < (n_e + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;
            int f = 0;

            if(l_id < n_e){
                f = s_f[l_id];
            }
            f_f = __ballot_sync(FULL_MASK, f);

            if(f_f != 0){
                f_f = WS * i + __ffs(f_f) - 1;
                break;
            }else if(i == (n_e + WS - 1) / WS - 1){
                if (l < n_l - 1) {
                    l++;
                    f_f = 0;

                    for (int j = 0; j < (n_e + WS - 1) / WS; j++) {
                        l_id = lane_id + WS * j;

                        if (l_id < n_e && s_nbr[l_id].d != Max) {
                            s_f[l_id] = 1;
                        }
                    }
                } else {
                    g_f = 0;
                }
            }
        }
    }

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
    
        if (l_id < k) {
            ann_seg[l_id] = s_nbr[l_id].nbr;
        }
    }
}



__global__
void UpdateNbr(float* d,
    NBR* nbr,
    int* f,
    int n_d,
    int k)
{
    int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * 2 * k;
    int* s_f = (int*)(sm + n_warp * 2 * k) + warp_id * 2 * k;

    int id = b_id * n_warp + warp_id;
    
    if (id >= n_d) {
        return;
    }

    NBR* nbr_seg = nbr + id * k;

    int g_f = 1, f_f = 0;

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if (l_id < k) {
            s_nbr[l_id] = nbr_seg[l_id];
            s_f[l_id] = f[s_nbr[l_id].nbr];
        }
    }

    while (g_f) {

        if (lane_id == 0) {
            s_f[f_f] = 0;
        }

        int nbr_id = s_nbr[f_f].nbr;
        NBR* nbr_nbr = nbr + (nbr_id * k);
        
        for (int i = 0; i < (k + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < k) {
                s_nbr[k + l_id] = nbr_nbr[l_id];
                s_f[k + l_id] = 1;
            }
        }

        for (int i = 0; i < k; i++) {
            nbr_id = s_nbr[k + i].nbr;
            
            float dist = compute_dist(lane_id, id, nbr_id, d);
            
            if (lane_id == 0) {
                s_nbr[k + i].d = dist;
            }
        }

        binary_search(lane_id, k, k, s_nbr);

        bitonic_sort(lane_id, k, s_nbr + k, false);
        
        bitonic_merge_triplet(lane_id, k, k, s_nbr, s_f);

        for (int i = 0; i < (k + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;
            int l_f = 0;

            if(l_id < k){
                l_f = s_f[l_id];
            }

            f_f = __ballot_sync(FULL_MASK, l_f);

            if(f_f != 0){
                f_f = WS * i + __ffs(f_f) - 1;
                break;
            }else if(i == (k + WS - 1) / WS - 1){
                g_f = 0;
            }
        }
    }

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
        
        if (l_id < k) {
            nbr_seg[l_id] = s_nbr[l_id];
        }
    }
}

__global__
void SearchKK(float* d, 
    float* q,
    NBR* ann,
    int* g,
    int n_d,
    int n_q,
    int s_bits,
    int M,
    int k,
    int n_e,
    unsigned long seed)
{
	int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * ((1 << s_bits) + M);
    int* s_f = (int*)(sm + n_warp * ((1 << s_bits) + M)) + warp_id * ((1 << s_bits) + M);
    
    int id = b_id * n_warp + warp_id;
    NBR* ann_seg = ann + id * k;
    
    int g_f = 1;
    int f_f = 0;
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    if (id >= n_q) {
        return;
    }

    for (int i = 0; i < ((1 << s_bits) + M + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if (l_id < (1 << s_bits) + M) {
            s_f[l_id] = 0;
            s_nbr[l_id] = init_value;
        }
    }

    curandState state;
    curand_init(seed, threadIdx.x + blockIdx.x * blockDim.x, 0, &state);

    if (lane_id == 0) {
        s_nbr[0].nbr = curand(&state) % n_d;
    }

    int nbr_id = s_nbr[0].nbr;
    
    float dist = compute_dist_q(lane_id, id, nbr_id, d, q);
    
    if (lane_id == 0) {
        s_nbr[0].d = dist;
    }
   	
    while (g_f) {

        if (lane_id == 0) {
            s_f[f_f] = 0;
        }
        
        nbr_id = s_nbr[f_f].nbr;
        int* nbr_nbr = g + (nbr_id << s_bits);

        for (int i = 0; i < ((1 << s_bits) + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;

            if (l_id < (1 << s_bits)) {
                s_nbr[M + l_id].nbr = nbr_nbr[l_id];
                s_f[M + l_id] = 1;
            }
        }

        for (int i = 0; i < (1 << s_bits); i++) {
            nbr_id = s_nbr[M + i].nbr;
            
            if (nbr_id >= n_d) {
                s_nbr[M + i].d = Max;
                continue;
            }
            
            dist = compute_dist_q(lane_id, id, nbr_id, d, q);
                
            if (lane_id == 0) {
                s_nbr[M + i].d = dist;
            }
        }
        
        binary_search(lane_id, (1 << s_bits), M, s_nbr);

        bitonic_sort(lane_id, (1 << s_bits), s_nbr + M, false);
        
        bitonic_merge_triplet(lane_id, (1 << s_bits), M, s_nbr, s_f);
        
        for (int i = 0; i < (n_e + WS - 1) / WS; i++) {
            int l_id = lane_id + WS * i;
            int f = 0;

            if(l_id < n_e){
                f = s_f[l_id];
            }

            f_f = __ballot_sync(FULL_MASK, f);

            if(f_f != 0){
                f_f = WS * i + __ffs(f_f) - 1;
                break;
            }else if(i == (n_e + WS - 1) / WS - 1){
                g_f = 0;
            }
        }
    }

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
    
        if (l_id < k) {
            ann_seg[l_id] = s_nbr[l_id];
        }
    }
}

__global__
void ANNMerge(NBR* prev_ann, 
    NBR* ann,
    int k,
    int e_k,
    int n_d)
{
    int n_warp = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int b_id = blockIdx.x;

    extern __shared__ NBR sm[];
    NBR* s_nbr = sm + warp_id * 2 * e_k;
    int* s_f = (int*)(sm + n_warp * 2 * e_k) + warp_id * 2 * e_k;

    int id = b_id * n_warp + warp_id;
    
    if (id >= n_d) {
        return;
    }

    NBR* ann_seg = ann + (id * k);
    NBR* prev_ann_seg = prev_ann + (id * k);
    NBR init_value;
    init_value.d = Max;
    init_value.nbr = n_d;

    for (int i = 0; i < (e_k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;

        if(l_id < e_k){
            if (l_id < k) {
                s_nbr[l_id] = ann_seg[l_id];
                s_nbr[2 * e_k - l_id - 1] = prev_ann_seg[l_id];
            } else {
                s_nbr[l_id] = init_value;
                s_nbr[2 * e_k - l_id - 1] = init_value;
            }
        }
    }

    binary_search(lane_id, e_k, e_k, s_nbr);

    bitonic_sort(lane_id, e_k, s_nbr + e_k, false);
    
    bitonic_merge(lane_id, e_k, e_k, s_nbr);

    for (int i = 0; i < (k + WS - 1) / WS; i++) {
        int l_id = lane_id + WS * i;
        
        if (l_id < k) {
            prev_ann_seg[l_id] = s_nbr[l_id];
        }
    }
}

#endif