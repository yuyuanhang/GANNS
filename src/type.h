#ifndef TYPE_H
#define TYPE_H

#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <algorithm>

#define FULL_MASK 0xffffffff
#define Max 0x1fffffff

struct NBR {
    float d;
    int nbr;
	
    __host__ __device__
	NBR() {}

    __host__ __device__
    NBR(float d_, int nbr_) : d(d_), nbr(nbr_) {}

	__host__ __device__
    bool operator <(NBR& a) const {
        return d < a.d;
    }

	__host__ __device__
    bool operator >(NBR& a) const {
        return d > a.d;
    }
};

struct Edge {
    int u;
    int v;
    float w;
};

struct EdgeComparator {
    __host__ __device__
    bool operator()(const Edge& a, const Edge& b) const {
        if (a.v != b.v)
            return a.v < b.v;
        return a.w < b.w;
    }
};

template<typename T>
T* copy_to_device(const T* host_array, size_t num_elements) {
    T* device_array = nullptr;
    cudaError_t err;

    err = cudaMalloc(&device_array, num_elements * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    err = cudaMemcpy(device_array, host_array, num_elements * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_array);
        return nullptr;
    }

    return device_array;
}

template <typename T>
T* copy_to_host(const T* device_array, size_t num_elements) {
    T* host_array = nullptr;
    cudaError_t err;

    err = cudaMallocHost(&host_array, num_elements * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    err = cudaMemcpy(host_array, device_array, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFreeHost(host_array);
        return nullptr;
    }

    return host_array;
}

template<typename T>
T* allocate_device_memory(size_t num_elements) {
    T* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, num_elements * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return device_ptr;
}



void load_truth(int* &truth, std::string truth_path, int &k_truth){

    std::ifstream in(truth_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open file: " << truth_path << std::endl;
        exit(1);
    }

    in.read((char*)&k_truth, sizeof(int));
    in.seekg(0, std::ios::end);
    long long f_sz = in.tellg();
    int n_d = f_sz / (k_truth + 1) / sizeof(int);

    cudaMallocHost(&truth, n_d * k_truth * sizeof(int));

    in.seekg(0, std::ios::beg);

    for (int i = 0; i < n_d; i++) {
        in.seekg(sizeof(int), std::ios::cur);
        in.read((char*)(truth + i * k_truth), k_truth * sizeof(int));
    }

    in.close();
}

float compute_recall(int* ann, int* truth, int n_q, int k, int k_truth){
    int n_p = 0;

    for (int i = 0; i < n_q; i++) {

        for (int j = 0; j < k; j++) {

            int id = ann[i * k + j];

            int* pos = NULL;
            pos = std::find(truth + i * k_truth, truth + i * k_truth + k, id);
            
            if (pos != truth + i * k_truth + k) {
                n_p++;
            }
        }
    }

    return (float)n_p / (n_q * k);
}

#endif