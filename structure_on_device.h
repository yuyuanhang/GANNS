#pragma once

#define FULL_MASK 0xffffffff
#define Max 0x1fffffff

template<class A,class B>
struct KernelPair{
    A first;
    B second;
	
	__device__
	KernelPair(){}


	__device__
    bool operator <(KernelPair& kp) const{
        return first < kp.first;
    }


	__device__
    bool operator >(KernelPair& kp) const{
        return first > kp.first;
    }
};

struct Edge{
    int source_point;
    int target_point;
    float distance;
};