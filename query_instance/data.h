#pragma once

#include <memory>
#include <vector>
#include <string.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>

using namespace std;

class Data{

private:
    float* data_;
    int num_of_points_;
    int dim_of_point_;

    void ReadVectorsFromFiles(string path){
        ifstream in_descriptor(path, std::ios::binary);
        
        if (!in_descriptor.is_open()) {
            exit(1);
        }

        in_descriptor.read((char*)&dim_of_point_, 4);

        in_descriptor.seekg(0, std::ios::end);
        long long file_size = in_descriptor.tellg(); 
        num_of_points_ = file_size / (dim_of_point_ + 1) / 4;

        data_ = new float[dim_of_point_ * num_of_points_];
        //memset(data_, 0, 4 * num_of_points_ * dim_of_point_);
    
        in_descriptor.seekg(0, std::ios::beg);

        for (int i = 0; i < num_of_points_; i++) {
            in_descriptor.seekg(4, std::ios::cur);
            in_descriptor.read((char*)(data_ + i * dim_of_point_), dim_of_point_ * 4);
        }

        in_descriptor.close();
    }

        //数据集不同，需要指定数据的维度？
        void ReadVectorsFromTxt(string path, int point_dim)
        {
            ifstream file(path);
            if(!file){
                std::cerr<<"Can not open file!"<<std::endl;
            }
            //获取数据的维度
            dim_of_point_ = point_dim;
    
            //获取文件大小
            file.seekg(0, std::ios::end);
            long long file_size = file.tellg();
            file.seekg(0, std::ios::beg);
    
            //计算每行的大小，假设每个浮点数占用12个字符
            long long estimated_line_size = point_dim * 12 + 1;
            // num_of_points_ = file_size / estimated_line_size;
            num_of_points_ = file_size / (dim_of_point_) / 4;

            //分配内存
            data_ = new float[dim_of_point_ * num_of_points_];
            
            //读取数据
            string line;
            int current_point = 0;
    
            while (getline(file, line) && current_point < num_of_points_) {
                std::stringstream ss(line);
                float value;
                int dim_count = 0;
                
                // 读取一行中的所有数值
                while (ss >> value && dim_count < dim_of_point_) {
                    data_[current_point * dim_of_point_ + dim_count] = value;
                    dim_count++;
                }
                
                // 检查维度是否匹配 
                if (dim_count != dim_of_point_) {
                    std::cerr << "Dimension mismatch at line " << current_point + 1 << std::endl;
                    exit(1);
                }
                
                current_point++;
                // std::cout<<"finish "<< current_point <<" points."<<endl;
            }
            
            //更新实际的点数
            num_of_points_ = current_point;
    
            file.close();
        }

public:
    Data(string path){
        // ReadVectorsFromFiles(path);
        ReadVectorsFromTxt(path, 1024);
    }
    
    float* GetFirstPositionofPoint(int point_id) const{
        return data_ + point_id * dim_of_point_;
    }

    float L2Distance(float* a, float* b) {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return sqrt(dist);
    }
    
    float IPDistance(float* a, float* b) {
        float dist = 0;
        for (int i = 0; i < dim_of_point_; ++i) {
            dist -= a[i] * b[i];
        }
        return dist;
    }
    
    float COSDistance(float* a, float* b) {
        float dist = 0;
        float length_a = 0, length_b = 0;

        for (int i = 0; i < dim_of_point_; ++i) {
            dist += a[i] * b[i];
            length_a += a[i] * a[i];
            length_b += b[i] * b[i];
        }

        dist = dist / sqrt(length_a) / sqrt(length_b);

        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }

        return dist;
    }

    inline int GetNumPoints(){
        return num_of_points_;
    }

    int GetDimofPoints(){
        return dim_of_point_;
    }

};