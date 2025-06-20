#ifndef NSW_H
#define NSW_H

#include <vector>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <fstream>
#include <chrono>
#include <cmath>
#include "data.h"
#include "void_g.h"
#include "nsw_op.h"
#include "type.h"

using namespace std;

class NSW : public VoidG {

public:
    int n_;

    int k_;
    int k_bound_;
    int s_bits_;

    int local_g_sz = 2000;
    int n_local_g;

    Data* data_;
    int* graph_;

    vector<NBR> f_local_g;
    std::mt19937_64 rand_gen_ = std::mt19937_64(1234567);

    NSW(Data* data) : data_(data){
        n_ = data_->num();
        n_local_g = (n_ + local_g_sz - 1) / local_g_sz;
        local_g_sz = (n_ + n_local_g - 1) / n_local_g;
    }

    void search(float* q, int k, int* &ann, int n_q, int M, int n_e) override { // n_e <= M, the existance of n_e is due to M = 2^x 
        M = pow(2.0, ceil(log(M) / log(2)));

        cudaMallocHost(&ann, sizeof(int) * n_q * k);

        auto t1 = chrono::steady_clock::now();
    	NSWOp::Search(data_->get_vector(0), 
            q, 
            graph_, 
            ann, 
            n_q,
            data_->num(),
            data_->dim(),
            s_bits_,
            k, 
            M,
            n_e);
        auto t2 = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Query speed: " << (double)n_q / ((double)duration / pow(10, 6)) << " QPS" << endl;
    }

    void build(int k, int M) override {
        M = pow(2.0, ceil(log(M) / log(2)));
        k_ = pow(2.0, ceil(log(k) / log(2)));

        s_bits_ = log(k_) / log(2) + 1;
        k_bound_ = 1 << s_bits_;

        f_local_g.resize(local_g_sz * k_bound_, NBR(Max, n_));

        float* d_d;
        NBR* d_nbr;
        NBR* d_local_nbr;
        cudaMallocHost(&graph_, sizeof(int) * (n_ << s_bits_));

        auto t1 = chrono::steady_clock::now();
        NSWOp::LocalGraphCons(data_->get_vector(0), 
            s_bits_, 
            n_, 
            data_->dim(), 
            k, 
            n_local_g, 
            local_g_sz,
            d_d,
            d_nbr,
            d_local_nbr);

        build_f_local_g(d_local_nbr);

        NSWOp::LocalGraphMerge(d_d, 
            graph_, 
            n_, 
            data_->dim(), 
            s_bits_, 
            k, 
            n_local_g,
            local_g_sz, 
            d_nbr,
            d_local_nbr,
            M,
            f_local_g.data());
        
        auto t2 = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Running time: " << (double)duration / pow(10, 6) << " seconds" << endl;
    }

    void save(string path) override {
        ofstream os(path, ios::binary);
        
        os.write((char*)&n_, sizeof(int));
        os.write((char*)&s_bits_, sizeof(int));
        os.write((char*)graph_, sizeof(int) * (n_ << s_bits_));
        cudaFreeHost(graph_);

        os.close();
    }

    void load(string path) override {
        ifstream in(path, ios::binary);

        if (!in.is_open()) {
            std::cerr << "Error: failed to open file: " << path << std::endl;
            exit(1);
        }

        in.read((char*)&n_, sizeof(int));
        in.read((char*)&s_bits_, sizeof(int));

        cudaMallocHost(&graph_, (n_ << s_bits_) * sizeof(int));
        in.read((char*)graph_, (n_ << s_bits_) * sizeof(int));

        in.close();
    }
    
    void build_f_local_g(NBR* d_local_nbr) {
        NBR* h_local_nbr = copy_to_host<NBR>(d_local_nbr, local_g_sz * k_bound_);

        for (int i = 1; i < local_g_sz; i++) {
            vector<NBR> nbr(min(i, k_));
            
            copy(h_local_nbr + i * k_bound_, h_local_nbr + i * k_bound_ + nbr.size(), nbr.begin());
            copy(nbr.begin(), nbr.end(), f_local_g.begin() + i * k_bound_);

            for (auto u : nbr) {
                update(i, u.nbr, u.d);
            }
        }
    }

    void update(int v, int u, float d) {
        int p = -1;

        for (int i = 0; i < k_bound_; i++) {
            if (d < f_local_g[u * k_bound_ + i].d) {
                p = i;
                break;
            }
        }

        if (p != -1) {
            for (int i = k_bound_ - 2; i >= p; i--) {
                f_local_g[u * k_bound_ + i + 1] = f_local_g[u * k_bound_ + i];
            }

            f_local_g[u * k_bound_ + p] = NBR(d, v);
        }
    }
};

#endif