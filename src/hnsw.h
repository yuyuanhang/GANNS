#ifndef HNSW_H
#define HNSW_H

#include <random>
#include <cmath>
#include <chrono>
#include "data.h"
#include "hnsw_op.h"
#include "nsw.h"
#include "void_g.h"

using namespace std;

class HNSW : public NSW {

private:
    int n_l = 0;
    vector<int> n_v_l;

    vector<int*> h_graph_;
    
    void assign_vert(){

        for (int i = 0; i < n_; i++) {
            uniform_real_distribution<float> uniform_distr(0, 1);
            float r_n = uniform_distr(rand_gen_);
            
            int lvl = - log(r_n) * (1 / log(1.0 * k_));
    
            if (n_l < lvl + 1) {
                n_l = lvl + 1;
                n_v_l.resize(n_l, 0);
            } 
            for (int j = 0; j < lvl + 1; j++) {
                n_v_l[j]++;
            }
        }
    
        for (int i = n_l - 1; i >= 0; i--) {
            if (n_v_l[i] >= local_g_sz) {
                n_l = i + 1;
                break;
            }
        }
    
        for (int i = 1; i < n_l; i++) {
            n_v_l[i] = ((n_v_l[i] + local_g_sz - 1) / local_g_sz) * local_g_sz;
        }
    }
    
public:
    HNSW(Data* data) : NSW(data) {}

    void search(float* q, int k, int* &ann, int n_q, int M, int n_e) override { // n_e <= M, the existance of n_e is due to M = 2^x 
        M = pow(2.0, ceil(log(M) / log(2)));

        cudaMallocHost(&ann, sizeof(int) * n_q * k);

        auto t1 = chrono::steady_clock::now();
    	HNSWOp::Search(data_->get_vector(0), 
            q, 
            h_graph_.data(), 
            ann, 
            n_q,
            data_->num(),
            data_->dim(),
            s_bits_,
            k, 
            M,
            n_e, 
            n_l,
            n_v_l.data());
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

        assign_vert();

        float* d_d;
        NBR* d_nbr;
        NBR* d_local_nbr;

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

        HNSWOp::LocalGraphMerge(d_d, 
            h_graph_, 
            n_,
            data_->dim(), 
            s_bits_, 
            k, 
            n_local_g, 
            local_g_sz,
            d_nbr,
            d_local_nbr, 
            M,
            f_local_g.data(), 
            n_l, 
            n_v_l.data());

        auto t2 = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Running time: " << (double)duration / pow(10, 6) << " seconds" << endl;
    }

    void save(string path) override {
        ofstream os(path, ios::binary);
        
        os.write((char*)&s_bits_, sizeof(int));
        os.write((char*)&n_l, sizeof(int));
        for (int i = 0; i < n_l; i++) {
            os.write((char*)&n_v_l[n_l - 1 - i], sizeof(int));
            os.write((char*)h_graph_[i], (n_v_l[n_l - 1 - i] << s_bits_) * sizeof(int));
            cudaFreeHost(h_graph_[i]);
        }

        os.close();
    }

    void load(string path) override {
        ifstream in(path, ios::binary);

        if (!in.is_open()) {
            std::cerr << "Error: failed to open file: " << path << std::endl;
            exit(1);
        }

        in.read((char*)&s_bits_, sizeof(int));
        in.read((char*)&n_l, sizeof(int));
        n_v_l.resize(n_l);
        h_graph_.resize(n_l);

        for (int i = 0; i < n_l; i++) {
            in.read((char*)&n_v_l[i], sizeof(int));

            int* h_g_seg;
            cudaMallocHost(&h_g_seg, (n_v_l[i] << s_bits_) * sizeof(int));
            in.read((char*)h_g_seg, (n_v_l[i] << s_bits_) * sizeof(int));
            h_graph_[i] = h_g_seg;
        }

        in.close();
    }
};

#endif