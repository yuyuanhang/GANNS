#ifndef DATA_H
#define DATA_H
#include <memory>
#include <vector>
#include <string.h>
#include <fstream>
#include <cmath>

using namespace std;

class Data{

private:
    float* v_;
    int n_;
    int d_;

    void read_vectors(string path){
        ifstream in(path, std::ios::binary);
        
        if (!in.is_open()) {
            std::cerr << "Error: failed to open file: " << path << std::endl;
            exit(1);
        }

        in.read((char*)&d_, sizeof(int));

        in.seekg(0, std::ios::end);
        long long f_sz = in.tellg(); 
        n_ = f_sz / (d_ + 1) / sizeof(int);

        v_ = new float[d_ * n_];
    
        in.seekg(0, std::ios::beg);

        for (int i = 0; i < n_; i++) {
            in.seekg(sizeof(int), std::ios::cur);
            in.read((char*)(v_ + i * d_), d_ * sizeof(int));
        }

        in.close();
    }

public:
    Data(string path){
        read_vectors(path);
        printf("n: %d, d: %d\n", n_, d_);
    }
    
    float* get_vector(int id_) const{
        return v_ + id_ * d_;
    }

    int num(){
        return n_;
    }

    int dim(){
        return d_;
    }

};

#endif