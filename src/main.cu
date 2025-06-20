#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "data.h"
#include "void_g.h"
#include "nsw.h"
#include "hnsw.h"

using namespace std;

int main(int argc,char** argv){

    string op_type = argv[1];

    if (op_type == "-b") {
        string base_path = argv[2];
        string g_type = argv[3];
        int k = atoi(argv[4]);
        int M = atoi(argv[5]);
        string g_path = argv[6];

        cout << "Loading data points..." << endl;
        Data* data = new Data(base_path);

        VoidG* graph;
        if (g_type == "nsw") {
            graph = new NSW(data);
        } else if (g_type == "hnsw") {
            graph = new HNSW(data);
        } else {
            cout << "Graph type not supported." << endl;
            exit(-1);
        }
        
        cout << "Building proximity graph " << g_type << "..." << endl;
        graph->build(k, M);
        
        cout << "Saving proximity graph " << g_type << "..." << endl;
        graph->save(g_path);
    } else if (op_type == "-q") {
        //required variables from external input
        string base_path = argv[2];
        string q_path = argv[3];
        string g_type = argv[4];
        string g_path = argv[5];
        string truth_path = argv[6];
        int k = atoi(argv[7]);
        int M = atoi(argv[8]);
        int n_e = atoi(argv[9]);

        cout << "Loading groundtruth..." << endl;
        int* truth = NULL;
        int k_truth;
        load_truth(truth, truth_path, k_truth);

        cout << "Loading data points and query points..." << endl;
        Data* data = new Data(base_path);
        Data* query = new Data(q_path);

        cout << "Loading proximity graph..." << endl;
        VoidG* graph;
        if (g_type == "nsw") {
            graph = new NSW(data);
            graph->load(g_path);
        } else if (g_type == "hnsw") {
            graph = new HNSW(data);
            graph->load(g_path);
        } else {
            cout << "Graph type not supported." << endl;
            exit(-1);
        }
	
        int* ann = NULL;
        cout << "Performing queries..." << endl;
        graph->search(query->get_vector(0), k, ann, query->num(), M, n_e);
    
        cout << "Recall: " << compute_recall(ann, truth, query->num(), k, k_truth) << endl;
    }
    
    return 0;
}
