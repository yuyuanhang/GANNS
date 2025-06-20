#ifndef HNSW_OP_H
#define HNSW_OP_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "type.h"
#include "kernels.h"

class HNSWOp {
public:
    static void LocalGraphMerge(float* d_d, 
		vector<int*> &h_g, 
		int n_d, 
		int d, 
		int s_bits,
		int k,
		int n_local_g, 
		int local_g_sz,
		NBR* d_nbr,
		NBR* d_local_nbr,
		int M,
		NBR* f_local_g,
        int n_l,
        int* n_v_l){
		
		int n_e = local_g_sz * k;
		Edge* d_e = allocate_device_memory<Edge>(n_e);
		int* d_f = allocate_device_memory<int>(n_e);
		int* d_sum = allocate_device_memory<int>(n_e);
		int idx_end = n_e;
		
		cudaMemcpy(d_nbr, f_local_g, (local_g_sz << s_bits) * sizeof(NBR), cudaMemcpyHostToDevice);
        int t_l = n_l - 1;
        int t_n_local_g = n_v_l[t_l] / local_g_sz;
        int t_n_d = n_v_l[t_l];

		for (int i = 1; i < n_local_g; i++) {
			if (i % 10 == 0) {
				cout << "Merging LG " << i << endl;
			}

            if (i == t_n_local_g) {
                int* d_g_seg = allocate_device_memory<int>(t_n_d << s_bits);

                int block_dim = 256;
		        int grid_dim = ((t_n_d << s_bits) + block_dim - 1) / block_dim;
		        Save<<<grid_dim, block_dim>>>(d_g_seg, d_nbr, t_n_d, s_bits);
		        cudaDeviceSynchronize();

                int* h_g_seg;
                cudaMallocHost(&h_g_seg, sizeof(int) * (t_n_d << s_bits));
                cudaMemcpy(h_g_seg, d_g_seg, (t_n_d << s_bits) * sizeof(int), cudaMemcpyDeviceToHost);

                h_g.push_back(h_g_seg);
                cudaFree(d_g_seg);

                t_l--;
                t_n_local_g = n_v_l[t_l] / local_g_sz;
                t_n_d = n_v_l[t_l];
            }

			int block_dim = 256;
			int grid_dim = (local_g_sz + block_dim / WS - 1) / (block_dim / WS);
			LocalGraphMergeK<<<grid_dim, block_dim, (2 * k + M) * (block_dim / WS) * (sizeof(NBR) + sizeof(int))>>>(
				d_nbr, 
				d_local_nbr,
				n_d,
				d_d,
				d_e,
				i,
				local_g_sz, 
				M,
				k,
				s_bits);
			cudaDeviceSynchronize();
	
			thrust::device_ptr<Edge> begin = thrust::device_pointer_cast(d_e);
			thrust::device_ptr<Edge> end = begin + n_e;
			thrust::sort(begin, end, EdgeComparator());
			
			/*// check
			Edge* h_e = copy_to_host<Edge>(d_e, n_e);
			for (int j = 0; j < n_e; j++) {
				cout << " " << h_e[j].v << " " << h_e[j].u << " " << h_e[j].w << endl;
			}*/
			
			block_dim = 256;
			grid_dim = (n_e + block_dim - 1) / block_dim;
			AssignFlag<<<grid_dim, block_dim>>>(d_e, d_f, n_e);
			cudaDeviceSynchronize();

			/*// check
			int* h_f = copy_to_host<int>(d_f, n_e);
			for (int j = n_e - 20; j < n_e; j++) {
				cout << " " << h_f[j] << endl;
			}*/

			thrust::device_ptr<int> in = thrust::device_pointer_cast(d_f);
    		thrust::device_ptr<int> out = thrust::device_pointer_cast(d_sum);
    		thrust::inclusive_scan(in, in + n_e, out);
			
			/*// check
			int* h_sum = copy_to_host<int>(d_sum, n_e + 1);
			for (int j = n_e - 20; j < n_e; j++) {
				cout << " " << h_sum[j] << endl;
			}*/

			int idx_sz = 0;
			cudaMemcpy(&idx_sz, d_sum + n_e - 1, sizeof(int), cudaMemcpyDeviceToHost);
			int* d_idx = allocate_device_memory<int>(idx_sz + 1);
			cudaMemcpy(d_idx + idx_sz, &idx_end, sizeof(int), cudaMemcpyHostToDevice);

			block_dim = 256;
			grid_dim = (n_e + block_dim - 1) / block_dim;
			WriteIdx<<<grid_dim, block_dim>>>(d_f, d_sum, d_idx, n_e);
			cudaDeviceSynchronize();

			/*// check
			int* h_idx = copy_to_host<int>(d_idx, idx_sz + 1);
			for (int j = 0; j <= idx_sz; j++) {
				cout << " " << h_idx[j] << endl;
			}*/
			
			block_dim = 256;
			grid_dim = (idx_sz + block_dim / WS - 1) / (block_dim / WS);
			UpdatePrevLG<<<grid_dim, block_dim, 4 * k * (block_dim / WS) * sizeof(NBR)>>>(
				d_nbr, 
				d_e, 
				d_idx,
				idx_sz, 
				n_d,
				2 * k);
			cudaDeviceSynchronize();

			cudaFree(d_idx);
		}

        int* d_g_seg = allocate_device_memory<int>(t_n_d << s_bits);

        int block_dim = 256;
		int grid_dim = ((t_n_d << s_bits) + block_dim - 1) / block_dim;
		Save<<<grid_dim, block_dim>>>(d_g_seg, d_nbr, t_n_d, s_bits);
		cudaDeviceSynchronize();

        int* h_g_seg;
        cudaMallocHost(&h_g_seg, sizeof(int) * (t_n_d << s_bits));
        cudaMemcpy(h_g_seg, d_g_seg, (t_n_d << s_bits) * sizeof(int), cudaMemcpyDeviceToHost);

        h_g.push_back(h_g_seg);
        cudaFree(d_g_seg);

		cudaFree(d_e);
		cudaFree(d_f);
		cudaFree(d_sum);
		cudaFree(d_nbr);
		cudaFree(d_local_nbr);
		cudaFree(d_d);
	}

    static void Search(float* h_d, 
        float* h_q, 
        int** h_g,
        int* h_ann,
        int n_q, 
        int n_d,
        int d, 
        int s_bits, 
		int k, 
		int M,
		int n_e,
        int n_l,
		int* h_n_v_l) {

		float* d_d = copy_to_device<float>(h_d, n_d * d);
        float* d_q = copy_to_device<float>(h_q, n_q * d);
        int* d_ann = allocate_device_memory<int>(n_q * k);
        
        vector<int*> h_d_g;
        for (int i = 0; i < n_l; i++) {
            int* d_g_seg = copy_to_device<int>(h_g[i], h_n_v_l[i] << s_bits);
            h_d_g.push_back(d_g_seg);
        }
        int** d_g = copy_to_device<int*>(h_d_g.data(), n_l);

        int block_dim = 256;
		int grid_dim = (n_q + block_dim / WS - 1) / (block_dim / WS);
		SearchK_H<<<grid_dim, block_dim, ((1 << s_bits) + M) * (block_dim / WS) * (sizeof(NBR) + sizeof(int))>>>(
            d_d, 
            d_q, 
            d_ann, 
            d_g, 
            n_d, 
            n_q, 
            s_bits, 
            M,
            k, 
            n_e,
            n_l);
        cudaDeviceSynchronize();
		
        cudaMemcpy(h_ann, d_ann, n_q * k * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_g);
        for (int i = 0; i < n_l; i++) {
            cudaFree(h_d_g[i]);
        }
		cudaFree(d_ann);
		cudaFree(d_q);
		cudaFree(d_d);
	}
};

#endif