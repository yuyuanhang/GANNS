#ifndef NSW_OP_H
#define NSW_OP_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "type.h"
#include "kernels.h"

__global__
void AssignFlag(Edge* e, int* f, int n) {
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (g_id < n) {
		if (g_id == 0) {
			f[g_id] = 1;
			return;
		}
		if (e[g_id - 1].v != e[g_id].v) {
			f[g_id] = 1;
		} else {
			f[g_id] = 0;
		}
	}
}

__global__
void WriteIdx(int* f, int* sum, int* idx, int n) {
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (g_id < n) {
		if (f[g_id] == 1) {
			int x = sum[g_id] - 1;
			idx[x] = g_id;
		}
	}
}

__global__
void Save(int* g, NBR* nbr, int n_d, int s_bits) {
	int g_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (g_id < (n_d << s_bits)) {
		g[g_id] = nbr[g_id].nbr;
	}
}


class NSWOp {
public:
	
	static void LocalGraphCons(float* h_d, 
		int s_bits, 
		int n_d, 
		int d, 
		int k,
		int n_local_g,
		int local_g_sz,
		float* &d_d,
		NBR* &d_nbr,
		NBR* &d_local_nbr){

		d_d = copy_to_device<float>(h_d, n_d * d);
		vector<NBR> init_nbr(n_d << s_bits, NBR(Max, n_d));
		d_nbr = copy_to_device<NBR>(init_nbr.data(), n_d << s_bits);
		d_local_nbr = copy_to_device<NBR>(init_nbr.data(), n_d << s_bits);
		
		NBR* d_dist_mat = allocate_device_memory<NBR>(n_d * k);

		for (int i = 0; i < (local_g_sz + k - 1) / k; i++) {
			int block_dim = 256;
			int grid_dim = n_local_g;
			DistMat<<<grid_dim, block_dim>>>(d_d, n_d, i * k, k, local_g_sz, d_dist_mat);
			cudaDeviceSynchronize();
		
			/*// check
			int test_id = 513;
			NBR* h_dist_mat = copy_to_host<NBR>(d_dist_mat, n_d * k);
			for (int j = 0; j < k; ++j) {
				cout << " " << h_dist_mat[test_id * k + j].d << " " << h_dist_mat[test_id * k + j].nbr << endl;
			}*/
		
			block_dim = 256;
			grid_dim = local_g_sz;
			SortNBR_LG<<<grid_dim, block_dim, 2 * k * block_dim / WS * sizeof(NBR)>>>(d_local_nbr, 
				n_d,
				d_d, 
				local_g_sz, 
				k,
				s_bits,
				i * k,
				d_dist_mat);
			cudaDeviceSynchronize();

			/*// check
			test_id = 513;
			NBR* h_local_nbr = copy_to_host<NBR>(d_local_nbr, n_d << s_bits);
			for (int j = 0; j < 2 * k; ++j) {
				cout << " " << h_local_nbr[test_id * 2 * k + j].d << " " << h_local_nbr[test_id * 2 * k + j].nbr << endl;
			}*/
		}

		cudaFree(d_dist_mat);
	}
	
	static void LocalGraphMerge(float* d_d, 
		int* &h_g, 
		int n_d, 
		int d, 
		int s_bits,
		int k,
		int n_local_g, 
		int local_g_sz,
		NBR* d_nbr,
		NBR* d_local_nbr,
		int M,
		NBR* f_local_g){

		int* d_g = allocate_device_memory<int>(n_d << s_bits);
		
		int n_e = local_g_sz * k;
		Edge* d_e = allocate_device_memory<Edge>(n_e);
		int* d_f = allocate_device_memory<int>(n_e);
		int* d_sum = allocate_device_memory<int>(n_e);
		int idx_end = n_e;
		
		cudaMemcpy(d_nbr, f_local_g, (local_g_sz << s_bits) * sizeof(NBR), cudaMemcpyHostToDevice);

		for (int i = 1; i < n_local_g; i++) {
			if (i % 10 == 0) {
				cout << "Merging LG " << i << endl;
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
			int* h_sum = copy_to_host<int>(d_sum, n_e);
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

		int block_dim = 256;
		int grid_dim = ((n_d << s_bits) + block_dim - 1) / block_dim;
		Save<<<grid_dim, block_dim>>>(d_g, d_nbr, n_d, s_bits);
		cudaDeviceSynchronize();

		cudaMemcpy(h_g, d_g, (n_d << s_bits) * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(d_e);
		cudaFree(d_f);
		cudaFree(d_sum);
		cudaFree(d_nbr);
		cudaFree(d_local_nbr);
		cudaFree(d_g);
		cudaFree(d_d);
	}
	
	static void Search(float* h_d, 
		float* h_q, 
		int* h_g, 
		int* h_ann, 
		int n_q, 
		int n_d, 
		int d, 
		int s_bits, 
		int k, 
		int M,
		int n_e) {
		
		float* d_d = copy_to_device<float>(h_d, n_d * d);
		float* d_q = copy_to_device<float>(h_q, n_q * d);
		int* d_ann = allocate_device_memory<int>(n_q * k);
		int* d_g = copy_to_device<int>(h_g, n_d << s_bits);

		int block_dim = 256;
		int grid_dim = (n_q + block_dim / WS - 1) / (block_dim / WS);
		SearchK<<<grid_dim, block_dim, ((1 << s_bits) + M) * (block_dim / WS) * (sizeof(NBR) + sizeof(int))>>>(
			d_d, 
			d_q, 
			d_ann,
			d_g, 
			n_d,
			n_q, 
			s_bits, 
			M, 
			k,
			n_e);
		cudaDeviceSynchronize();
		
		cudaMemcpy(h_ann, d_ann, n_q * k * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(d_g);
		cudaFree(d_ann);
		cudaFree(d_q);
		cudaFree(d_d);
	}
};

#endif