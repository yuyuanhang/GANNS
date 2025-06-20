#ifndef VOID_G_H
#define VOID_G_H
#include <string>

using namespace std;

class VoidG{
public:
	virtual void search(float* q, int k, int* &ann, int n_q, int M, int n_e) = 0;
	virtual void build(int k, int M) = 0;
	virtual void save(string path) = 0;
	virtual void load(string path) = 0;
};

#endif