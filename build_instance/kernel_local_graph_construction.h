#pragma once
#include "structure_on_device.h"

__global__
void DistanceMatrixComputation(float* d_data, int total_num_of_points, int num_of_points_one_batch, KernelPair<float, int>* distance_matrix){
#define DIM 1024
    int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    
    for (int i = 0; i < num_of_points_one_batch; i++) {
        int crt_point_id = b_id * num_of_points_one_batch + i;

        if (crt_point_id >= total_num_of_points) {
            break;
        }
        
        KernelPair<float, int>* crt_distance = distance_matrix + crt_point_id * num_of_points_one_batch;

#if DIM > 0
	float q1 = 0;
	if (t_id < DIM) {
		q1 = d_data[crt_point_id * DIM + t_id];
	}
#endif
#if DIM > 32
    float q2 = 0;
    if (t_id + 32 < DIM) {
        q2 = d_data[crt_point_id * DIM + t_id + 32];
    }
#endif
#if DIM > 64
    float q3 = 0;
    if (t_id + 64 < DIM) {
    	q3 = d_data[crt_point_id * DIM + t_id + 64];
   	}
#endif
#if DIM > 96
    float q4 = 0;
    if (t_id + 96 < DIM) {
    	q4 = d_data[crt_point_id * DIM + t_id + 96];
    }
#endif
#if DIM > 128
    float q5 = 0;
    if (t_id + 128 < DIM) {
        q5 = d_data[crt_point_id * DIM + t_id + 128];
    }
#endif
#if DIM > 160
    float q6 = 0;
    if (t_id + 160 < DIM) {
        q6 = d_data[crt_point_id * DIM + t_id + 160];
    }
#endif
#if DIM > 192
    float q7 = 0;
    if (t_id + 192 < DIM) {
        q7 = d_data[crt_point_id * DIM + t_id + 192];
    }
#endif
#if DIM > 224
    float q8 = 0;
    if (t_id + 224 < DIM) {
        q8 = d_data[crt_point_id * DIM + t_id + 224];
    }
#endif
#if DIM > 256
    float q9 = 0;
    if (t_id + 256 < DIM) {
        q9 = d_data[crt_point_id * DIM + t_id + 256];
    }
#endif
#if DIM > 288
    float q10 = 0;
    if (t_id + 288 < DIM) {
        q10 = d_data[crt_point_id * DIM + t_id + 288];
    }
#endif
#if DIM > 320
    float q11 = 0;
    if (t_id + 320 < DIM) {
        q11 = d_data[crt_point_id * DIM + t_id + 320];
    }
#endif
#if DIM > 352
    float q12 = 0;
    if (t_id + 352 < DIM) {
        q12 = d_data[crt_point_id * DIM + t_id + 352];
    }
#endif
#if DIM > 384
    float q13 = 0;
    if (t_id + 384 < DIM) {
        q13 = d_data[crt_point_id * DIM + t_id + 384];
    }
#endif
#if DIM > 416
    float q14 = 0;
    if (t_id + 416 < DIM) {
        q14 = d_data[crt_point_id * DIM + t_id + 416];
    }
#endif
#if DIM > 448
    float q15 = 0;
    if (t_id + 448 < DIM) {
        q15 = d_data[crt_point_id * DIM + t_id + 448];
    }
#endif
#if DIM > 480
    float q16 = 0;
    if (t_id + 480 < DIM) {
        q16 = d_data[crt_point_id * DIM + t_id + 480];
    }
#endif
#if DIM > 512
    float q17 = 0;
    if (t_id + 512 < DIM) {
        q17 = d_data[crt_point_id * DIM + t_id + 512];
    }
#endif
#if DIM > 544
    float q18 = 0;
    if (t_id + 544 < DIM) {
        q18 = d_data[crt_point_id * DIM + t_id + 544];
    }
#endif
#if DIM > 576
    float q19 = 0;
    if (t_id + 576 < DIM) {
        q19 = d_data[crt_point_id * DIM + t_id + 576];
    }
#endif
#if DIM > 608
    float q20 = 0;
    if (t_id + 608 < DIM) {
        q20 = d_data[crt_point_id * DIM + t_id + 608];
    }
#endif
#if DIM > 640
    float q21 = 0;
    if (t_id + 640 < DIM) {
        q21 = d_data[crt_point_id * DIM + t_id + 640];
    }
#endif
#if DIM > 672
    float q22 = 0;
    if (t_id + 672 < DIM) {
        q22 = d_data[crt_point_id * DIM + t_id + 672];
    }
#endif
#if DIM > 704
    float q23 = 0;
    if (t_id + 704 < DIM) {
        q23 = d_data[crt_point_id * DIM + t_id + 704];
    }
#endif
#if DIM > 736
    float q24 = 0;
    if (t_id + 736 < DIM) {
        q24 = d_data[crt_point_id * DIM + t_id + 736];
    }
#endif
#if DIM > 768
    float q25 = 0;
    if (t_id + 768 < DIM) {
        q25 = d_data[crt_point_id * DIM + t_id + 768];
    }
#endif
#if DIM > 800
    float q26 = 0;
    if (t_id + 800 < DIM) {
        q26 = d_data[crt_point_id * DIM + t_id + 800];
    }
#endif
#if DIM > 832
    float q27 = 0;
    if (t_id + 832 < DIM) {
        q27 = d_data[crt_point_id * DIM + t_id + 832];
    }
#endif
#if DIM > 864
    float q28 = 0;
    if (t_id + 864 < DIM) {
        q28 = d_data[crt_point_id * DIM + t_id + 864];
    }
#endif
#if DIM > 896
    float q29 = 0;
    if (t_id + 896 < DIM) {
        q29 = d_data[crt_point_id * DIM + t_id + 896];
    }
#endif
#if DIM > 928
    float q30 = 0;
    if (t_id + 224 < DIM) {
        q30 = d_data[crt_point_id * DIM + t_id + 928];
    }
#endif


        for (int j = i + 1; j < num_of_points_one_batch; j++) {
            
            int target_point_id = b_id * num_of_points_one_batch + j;

            if(target_point_id >= total_num_of_points){
                break;
            }
    
#if DIM > 0
	float p1 = 0;
	if (t_id < DIM) {
		p1 = d_data[target_point_id * DIM + t_id];
	}
#endif
#if DIM > 32
    float p2 = 0;
    if (t_id + 32 < DIM) {
        p2 = d_data[target_point_id * DIM + t_id + 32];
    }
#endif
#if DIM > 64
    float p3 = 0;
    if (t_id + 64 < DIM) {
    	p3 = d_data[target_point_id * DIM + t_id + 64];
   	}
#endif
#if DIM > 96
    float p4 = 0;
    if (t_id + 96 < DIM) {
    	p4 = d_data[target_point_id * DIM + t_id + 96];
    }
#endif
#if DIM > 128
    float p5 = 0;
    if (t_id + 128 < DIM) {
        p5 = d_data[target_point_id * DIM + t_id + 128];
    }
#endif
#if DIM > 160
    float p6 = 0;
    if (t_id + 160 < DIM) {
        p6 = d_data[target_point_id * DIM + t_id + 160];
    }
#endif
#if DIM > 192
    float p7 = 0;
    if (t_id + 192 < DIM) {
        p7 = d_data[target_point_id * DIM + t_id + 192];
    }
#endif
#if DIM > 224
    float p8 = 0;
    if (t_id + 224 < DIM) {
        p8 = d_data[target_point_id * DIM + t_id + 224];
    }
#endif
#if DIM > 256
    float p9 = 0;
    if (t_id + 256 < DIM) {
        p9 = d_data[target_point_id * DIM + t_id + 256];
    }
#endif
#if DIM > 288
    float p10 = 0;
    if (t_id + 288 < DIM) {
        p10 = d_data[target_point_id * DIM + t_id + 288];
    }
#endif
#if DIM > 320
    float p11 = 0;
    if (t_id + 320 < DIM) {
        p11 = d_data[target_point_id * DIM + t_id + 320];
    }
#endif
#if DIM > 352
    float p12 = 0;
    if (t_id + 352 < DIM) {
        p12 = d_data[target_point_id * DIM + t_id + 352];
    }
#endif
#if DIM > 384
    float p13 = 0;
    if (t_id + 384 < DIM) {
        p13 = d_data[target_point_id * DIM + t_id + 384];
    }
#endif
#if DIM > 416
    float p14 = 0;
    if (t_id + 416 < DIM) {
        p14 = d_data[target_point_id * DIM + t_id + 416];
    }
#endif
#if DIM > 448
    float p15 = 0;
    if (t_id + 448 < DIM) {
        p15 = d_data[target_point_id * DIM + t_id + 448];
    }
#endif
#if DIM > 480
    float p16 = 0;
    if (t_id + 480 < DIM) {
        p16 = d_data[target_point_id * DIM + t_id + 480];
    }
#endif
#if DIM > 512
    float p17 = 0;
    if (t_id + 512 < DIM) {
        p17 = d_data[target_point_id * DIM + t_id + 512];
    }
#endif
#if DIM > 544
    float p18 = 0;
    if (t_id + 544 < DIM) {
        p18 = d_data[target_point_id * DIM + t_id + 544];
    }
#endif
#if DIM > 576
    float p19 = 0;
    if (t_id + 576 < DIM) {
        p19 = d_data[target_point_id * DIM + t_id + 576];
    }
#endif
#if DIM > 608
    float p20 = 0;
    if (t_id + 608 < DIM) {
        p20 = d_data[target_point_id * DIM + t_id + 608];
    }
#endif
#if DIM > 640
    float p21 = 0;
    if (t_id + 640 < DIM) {
        p21 = d_data[target_point_id * DIM + t_id + 640];
    }
#endif
#if DIM > 672
    float p22 = 0;
    if (t_id + 672 < DIM) {
        p22 = d_data[target_point_id * DIM + t_id + 672];
    }
#endif
#if DIM > 704
    float p23 = 0;
    if (t_id + 704 < DIM) {
        p23 = d_data[target_point_id * DIM + t_id + 704];
    }
#endif
#if DIM > 736
    float p24 = 0;
    if (t_id + 736 < DIM) {
        p24 = d_data[target_point_id * DIM + t_id + 736];
    }
#endif
#if DIM > 768
    float p25 = 0;
    if (t_id + 768 < DIM) {
        p25 = d_data[target_point_id * DIM + t_id + 768];
    }
#endif
#if DIM > 800
    float p26 = 0;
    if (t_id + 800 < DIM) {
        p26 = d_data[target_point_id * DIM + t_id + 800];
    }
#endif
#if DIM > 832
    float p27 = 0;
    if (t_id + 832 < DIM) {
        p27 = d_data[target_point_id * DIM + t_id + 832];
    }
#endif
#if DIM > 864
    float p28 = 0;
    if (t_id + 864 < DIM) {
        p28 = d_data[target_point_id * DIM + t_id + 864];
    }
#endif
#if DIM > 896
    float p29 = 0;
    if (t_id + 896 < DIM) {
        p29 = d_data[target_point_id * DIM + t_id + 896];
    }
#endif
#if DIM > 928
    float p30 = 0;
    if (t_id + 224 < DIM) {
        p30 = d_data[target_point_id * DIM + t_id + 928];
    }
#endif

    
#if USE_L2_DIST_
    #if DIM > 0
		float delta1 = (p1 - q1) * (p1 - q1);
	#endif
	#if DIM > 32
        float delta2 = (p2 - q2) * (p2 - q2);
    #endif
    #if DIM > 64
        float delta3 = (p3 - q3) * (p3 - q3);
    #endif
    #if DIM > 96
        float delta4 = (p4 - q4) * (p4 - q4);
    #endif
    #if DIM > 128
        float delta5 = (p5 - q5) * (p5 - q5);
    #endif
    #if DIM > 160
        float delta6 = (p6 - q6) * (p6 - q6);
    #endif
    #if DIM > 192
        float delta7 = (p7 - q7) * (p7 - q7);
    #endif
    #if DIM > 224
        float delta8 = (p8 - q8) * (p8 - q8);
    #endif
    #if DIM > 256
        float delta9 = (p9 - q9) * (p9 - q9);
    #endif
    #if DIM > 288
        float delta10 = (p10 - q10) * (p10 - q10);
    #endif
    #if DIM > 320
        float delta11 = (p11 - q11) * (p11 - q11);
    #endif
    #if DIM > 352
        float delta12 = (p12 - q12) * (p12 - q12);
    #endif
    #if DIM > 384
        float delta13 = (p13 - q13) * (p13 - q13);
    #endif
    #if DIM > 416
        float delta14 = (p14 - q14) * (p14 - q14);
    #endif
    #if DIM > 448
        float delta15 = (p15 - q15) * (p15 - q15);
    #endif
    #if DIM > 480
        float delta16 = (p16 - q16) * (p16 - q16);
    #endif
    #if DIM > 512
        float delta17 = (p17 - q17) * (p17 - q17);
    #endif
    #if DIM > 544
        float delta18 = (p18 - q18) * (p18 - q18);
    #endif
    #if DIM > 576
        float delta19 = (p19 - q19) * (p19 - q19);
    #endif
    #if DIM > 608
        float delta20 = (p20 - q20) * (p20 - q20);
    #endif
    #if DIM > 640
        float delta21 = (p21 - q21) * (p21 - q21);
    #endif
    #if DIM > 672
        float delta22 = (p22 - q22) * (p22 - q22);
    #endif
    #if DIM > 704
        float delta23 = (p23 - q23) * (p23 - q23);
    #endif
    #if DIM > 736
        float delta24 = (p24 - q24) * (p24 - q24);
    #endif
    #if DIM > 768
        float delta25 = (p25 - q25) * (p25 - q25);
    #endif
    #if DIM > 800
        float delta26 = (p26 - q26) * (p26 - q26);
    #endif
    #if DIM > 832
        float delta27 = (p27 - q27) * (p27 - q27);
    #endif
    #if DIM > 864
        float delta28 = (p28 - q28) * (p28 - q28);
    #endif
    #if DIM > 896
        float delta29 = (p29 - q29) * (p29 - q29);
    #endif
    #if DIM > 928
        float delta30 = (p30 - q30) * (p30 - q30);
    #endif

#elif USE_IP_DIST_
    #if DIM > 0
		float delta1 = p1 * q1;
	#endif
	#if DIM > 32
        float delta2 = p2 * q2;
    #endif
    #if DIM > 64
        float delta3 = p3 * q3;
    #endif
    #if DIM > 96
        float delta4 = p4 * q4;
    #endif
    #if DIM > 128
        float delta5 = p5 * q5;
    #endif
    #if DIM > 160
        float delta6 = p6 * q6;
    #endif
    #if DIM > 192
        float delta7 = p7 * q7;
    #endif
    #if DIM > 224
        float delta8 = p8 * q8;
    #endif
    #if DIM > 256
        float delta9 = p9 * q9;
    #endif
    #if DIM > 288
        float delta10 = p10 * q10;
    #endif
    #if DIM > 320
        float delta11 = p11 * q11;
    #endif
    #if DIM > 352
        float delta12 = p12 * q12;
    #endif
    #if DIM > 384
        float delta13 = p13 * q13;
    #endif
    #if DIM > 416
        float delta14 = p14 * q14;
    #endif
    #if DIM > 448
        float delta15 = p15 * q15;
    #endif
    #if DIM > 480
        float delta16 = p16 * q16;
    #endif
    #if DIM > 512
        float delta17 = p17 * q17;
    #endif
    #if DIM > 544
        float delta18 = p18 * q18;
    #endif
    #if DIM > 576
        float delta19 = p19 * q19;
    #endif
    #if DIM > 608
        float delta20 = p20 * q20;
    #endif
    #if DIM > 640
        float delta21 = p21 * q21;
    #endif
    #if DIM > 672
        float delta22 = p22 * q22;
    #endif
    #if DIM > 704
        float delta23 = p23 * q23;
    #endif
    #if DIM > 736
        float delta24 = p24 * q24;
    #endif
    #if DIM > 768
        float delta25 = p25 * q25;
    #endif
    #if DIM > 800
        float delta26 = p26 * q26;
    #endif
    #if DIM > 832
        float delta27 = p27 * q27;
    #endif
    #if DIM > 864
        float delta28 = p28 * q28;
    #endif
    #if DIM > 896
        float delta29 = p29 * q29;
    #endif
    #if DIM > 928
        float delta30 = p30 * q30;
    #endif

#elif USE_COS_DIST_
    #if DIM > 0
		float delta1 = p1 * q1;
		float p_l2_1 = p1 * p1;
		float q_l2_1 = q1 * q1;
	#endif
	#if DIM > 32
        float delta2 = p2 * q2;
        float p_l2_2 = p2 * p2;
        float q_l2_2 = q2 * q2;
    #endif
    #if DIM > 64
        float delta3 = p3 * q3;
        float p_l2_3 = p3 * p3;
        float q_l2_3 = q3 * q3;
    #endif
    #if DIM > 96
        float delta4 = p4 * q4;
        float p_l2_4 = p4 * p4;
        float q_l2_4 = q4 * q4;
    #endif
    #if DIM > 128
        float delta5 = p5 * q5;
        float p_l2_5 = p5 * p5;
        float q_l2_5 = q5 * q5;
    #endif
    #if DIM > 160
        float delta6 = p6 * q6;
        float p_l2_6 = p6 * p6;
        float q_l2_6 = q6 * q6;
    #endif
    #if DIM > 192
        float delta7 = p7 * q7;
        float p_l2_7 = p7 * p7;
        float q_l2_7 = q7 * q7;
    #endif
    #if DIM > 224
        float delta8 = p8 * q8;
        float p_l2_8 = p8 * p8;
        float q_l2_8 = q8 * q8;
    #endif
    #if DIM > 256
        float delta9 = p9 * q9;
        float p_l2_9 = p9 * p9;
        float q_l2_9 = q9 * q9;
    #endif
    #if DIM > 288
        float delta10 = p10 * q10;
        float p_l2_10 = p10 * p10;
        float q_l2_10 = q10 * q10;
    #endif
    #if DIM > 320
        float delta11 = p11 * q11;
        float p_l2_11 = p11 * p11;
        float q_l2_11 = q11 * q11;
    #endif
    #if DIM > 352
        float delta12 = p12 * q12;
        float p_l2_12 = p12 * p12;
        float q_l2_12 = q12 * q12;
    #endif
    #if DIM > 384
        float delta13 = p13 * q13;
        float p_l2_13 = p13 * p13;
        float q_l2_13 = q13 * q13;
    #endif
    #if DIM > 416
        float delta14 = p14 * q14;
        float p_l2_14 = p14 * p14;
        float q_l2_14 = q14 * q14;
    #endif
    #if DIM > 448
        float delta15 = p15 * q15;
        float p_l2_15 = p15 * p15;
        float q_l2_15 = q15 * q15;
    #endif
    #if DIM > 480
        float delta16 = p16 * q16;
        float p_l2_16 = p16 * p16;
        float q_l2_16 = q16 * q16;
    #endif
    #if DIM > 512
        float delta17 = p17 * q17;
        float p_l2_17 = p17 * p17;
        float q_l2_17 = q17 * q17;
    #endif
    #if DIM > 544
        float delta18 = p18 * q18;
        float p_l2_18 = p18 * p18;
        float q_l2_18 = q18 * q18;
    #endif
    #if DIM > 576
        float delta19 = p19 * q19;
        float p_l2_19 = p19 * p19;
        float q_l2_19 = q19 * q19;
    #endif
    #if DIM > 608
        float delta20 = p20 * q20;
        float p_l2_20 = p20 * p20;
        float q_l2_20 = q20 * q20;
    #endif
    #if DIM > 640
        float delta21 = p21 * q21;
        float p_l2_21 = p21 * p21;
        float q_l2_21 = q21 * q21;
    #endif
    #if DIM > 672
        float delta22 = p22 * q22;
        float p_l2_22 = p22 * p22;
        float q_l2_22 = q22 * q22;
    #endif
    #if DIM > 704
        float delta23 = p23 * q23;
        float p_l2_23 = p23 * p23;
        float q_l2_23 = q23 * q23;
    #endif
    #if DIM > 736
        float delta24 = p24 * q24;
        float p_l2_24 = p24 * p24;
        float q_l2_24 = q24 * q24;
    #endif
    #if DIM > 768
        float delta25 = p25 * q25;
        float p_l2_25 = p25 * p25;
        float q_l2_25 = q25 * q25;
    #endif
    #if DIM > 800
        float delta26 = p26 * q26;
        float p_l2_26 = p26 * p26;
        float q_l2_26 = q26 * q26;
    #endif
    #if DIM > 832
        float delta27 = p27 * q27;
        float p_l2_27 = p27 * p27;
        float q_l2_27 = q27 * q27;
    #endif
    #if DIM > 864
        float delta28 = p28 * q28;
        float p_l2_28 = p28 * p28;
        float q_l2_28 = q28 * q28;
    #endif
    #if DIM > 896
        float delta29 = p29 * q29;
        float p_l2_29 = p29 * p29;
        float q_l2_29 = q29 * q29;
    #endif
    #if DIM > 928
        float delta30 = p30 * q30;
        float p_l2_30 = p30 * p30;
        float q_l2_30 = q30 * q30;
    #endif
#endif

            
#if USE_L2_DIST_
    	float dist = 0;
    #if DIM > 0
    	dist += delta1;
    #endif
    #if DIM > 32
        dist += delta2;
    #endif
    #if DIM > 64
        dist += delta3;
    #endif
    #if DIM > 96
        dist += delta4;
    #endif
    #if DIM > 128
        dist += delta5;
    #endif
    #if DIM > 160
        dist += delta6;
    #endif
    #if DIM > 192
        dist += delta7;
    #endif
    #if DIM > 224
        dist += delta8;
    #endif
    #if DIM > 256
        dist += delta9;
    #endif
    #if DIM > 288
        dist += delta10;
    #endif
    #if DIM > 320
        dist += delta11;
    #endif
    #if DIM > 352
        dist += delta12;
    #endif
    #if DIM > 384
        dist += delta13;
    #endif
    #if DIM > 416
        dist += delta14;
    #endif
    #if DIM > 448
        dist += delta15;
    #endif
    #if DIM > 480
        dist += delta16;
    #endif
    #if DIM > 512
        dist += delta17;
    #endif
    #if DIM > 544
        dist += delta18;
    #endif
    #if DIM > 576
        dist += delta19;
    #endif
    #if DIM > 608
        dist += delta20;
    #endif
    #if DIM > 640
        dist += delta21;
    #endif
    #if DIM > 672
        dist += delta22;
    #endif
    #if DIM > 704
        dist += delta23;
    #endif
    #if DIM > 736
        dist += delta24;
    #endif
    #if DIM > 768
        dist += delta25;
    #endif
    #if DIM > 800
        dist += delta26;
    #endif
    #if DIM > 832
        dist += delta27;
    #endif
    #if DIM > 864
        dist += delta28;
    #endif
    #if DIM > 896
        dist += delta29;
    #endif
    #if DIM > 928
        dist += delta30;
    #endif

#elif USE_IP_DIST_
    	float dist = 0;
    #if DIM > 0
    	dist += delta1;
    #endif
    #if DIM > 32
        dist += delta2;
    #endif
    #if DIM > 64
        dist += delta3;
    #endif
    #if DIM > 96
        dist += delta4;
    #endif
    #if DIM > 128
        dist += delta5;
    #endif
    #if DIM > 160
        dist += delta6;
    #endif
    #if DIM > 192
        dist += delta7;
    #endif
    #if DIM > 224
        dist += delta8;
    #endif
    #if DIM > 256
        dist += delta9;
    #endif
    #if DIM > 288
        dist += delta10;
    #endif
    #if DIM > 320
        dist += delta11;
    #endif
    #if DIM > 352
        dist += delta12;
    #endif
    #if DIM > 384
        dist += delta13;
    #endif
    #if DIM > 416
        dist += delta14;
    #endif
    #if DIM > 448
        dist += delta15;
    #endif
    #if DIM > 480
        dist += delta16;
    #endif
    #if DIM > 512
        dist += delta17;
    #endif
    #if DIM > 544
        dist += delta18;
    #endif
    #if DIM > 576
        dist += delta19;
    #endif
    #if DIM > 608
        dist += delta20;
    #endif
    #if DIM > 640
        dist += delta21;
    #endif
    #if DIM > 672
        dist += delta22;
    #endif
    #if DIM > 704
        dist += delta23;
    #endif
    #if DIM > 736
        dist += delta24;
    #endif
    #if DIM > 768
        dist += delta25;
    #endif
    #if DIM > 800
        dist += delta26;
    #endif
    #if DIM > 832
        dist += delta27;
    #endif
    #if DIM > 864
        dist += delta28;
    #endif
    #if DIM > 896
        dist += delta29;
    #endif
    #if DIM > 928
        dist += delta30;
    #endif

#elif USE_COS_DIST_
    	float dist = 0;
    	float p_l2 = 0;
    	float q_l2 = 0;
    #if DIM > 0
    	dist += delta1;
    	p_l2 += p_l2_1;
    	q_l2 += q_l2_1;
    #endif
    #if DIM > 32
        dist += delta2;
        p_l2 += p_l2_2;
    	q_l2 += q_l2_2;
    #endif
    #if DIM > 64
        dist += delta3;
        p_l2 += p_l2_3;
    	q_l2 += q_l2_3;
    #endif
    #if DIM > 96
        dist += delta4;
        p_l2 += p_l2_4;
    	q_l2 += q_l2_4;
    #endif
    #if DIM > 128
        dist += delta5;
        p_l2 += p_l2_5;
    	q_l2 += q_l2_5;
    #endif
    #if DIM > 160
        dist += delta6;
        p_l2 += p_l2_6;
    	q_l2 += q_l2_6;
    #endif
    #if DIM > 192
        dist += delta7;
        p_l2 += p_l2_7;
    	q_l2 += q_l2_7;
    #endif
    #if DIM > 224
        dist += delta8;
        p_l2 += p_l2_8;
    	q_l2 += q_l2_8;
    #endif
    #if DIM > 256
        dist += delta9;
        p_l2 += p_l2_9;
    	q_l2 += q_l2_9;
    #endif
    #if DIM > 288
        dist += delta10;
        p_l2 += p_l2_10;
    	q_l2 += q_l2_10;
    #endif
    #if DIM > 320
        dist += delta11;
        p_l2 += p_l2_11;
    	q_l2 += q_l2_11;
    #endif
    #if DIM > 352
        dist += delta12;
        p_l2 += p_l2_12;
    	q_l2 += q_l2_12;
    #endif
    #if DIM > 384
        dist += delta13;
        p_l2 += p_l2_13;
    	q_l2 += q_l2_13;
    #endif
    #if DIM > 416
        dist += delta14;
        p_l2 += p_l2_14;
    	q_l2 += q_l2_14;
    #endif
    #if DIM > 448
        dist += delta15;
        p_l2 += p_l2_15;
    	q_l2 += q_l2_15;
    #endif
    #if DIM > 480
        dist += delta16;
        p_l2 += p_l2_16;
    	q_l2 += q_l2_16;
    #endif
    #if DIM > 512
        dist += delta17;
        p_l2 += p_l2_17;
    	q_l2 += q_l2_17;
    #endif
    #if DIM > 544
        dist += delta18;
        p_l2 += p_l2_18;
    	q_l2 += q_l2_18;
    #endif
    #if DIM > 576
        dist += delta19;
        p_l2 += p_l2_19;
    	q_l2 += q_l2_19;
    #endif
    #if DIM > 608
        dist += delta20;
        p_l2 += p_l2_20;
    	q_l2 += q_l2_20;
    #endif
    #if DIM > 640
        dist += delta21;
        p_l2 += p_l2_21;
    	q_l2 += q_l2_21;
    #endif
    #if DIM > 672
        dist += delta22;
        p_l2 += p_l2_22;
    	q_l2 += q_l2_22;
    #endif
    #if DIM > 704
        dist += delta23;
        p_l2 += p_l2_23;
    	q_l2 += q_l2_23;
    #endif
    #if DIM > 736
        dist += delta24;
        p_l2 += p_l2_24;
    	q_l2 += q_l2_24;
    #endif
    #if DIM > 768
        dist += delta25;
        p_l2 += p_l2_25;
    	q_l2 += q_l2_25;
    #endif
    #if DIM > 800
        dist += delta26;
        p_l2 += p_l2_26;
    	q_l2 += q_l2_26;
    #endif
    #if DIM > 832
        dist += delta27;
        p_l2 += p_l2_27;
    	q_l2 += q_l2_27;
    #endif
    #if DIM > 864
        dist += delta28;
        p_l2 += p_l2_28;
    	q_l2 += q_l2_28;
    #endif
    #if DIM > 896
        dist += delta29;
        p_l2 += p_l2_29;
    	q_l2 += q_l2_29;
    #endif
    #if DIM > 928
        dist += delta30;
        p_l2 += p_l2_30;
    	q_l2 += q_l2_30;
    #endif
#endif

        
#if USE_L2_DIST_
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
#elif USE_IP_DIST_
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);
#elif USE_COS_DIST_
    dist += __shfl_down_sync(FULL_MASK, dist, 16);
    dist += __shfl_down_sync(FULL_MASK, dist, 8);
    dist += __shfl_down_sync(FULL_MASK, dist, 4);
    dist += __shfl_down_sync(FULL_MASK, dist, 2);
    dist += __shfl_down_sync(FULL_MASK, dist, 1);

    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 16);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 8);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 4);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 2);
    p_l2 += __shfl_down_sync(FULL_MASK, p_l2, 1);

    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 16);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 8);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 4);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 2);
    q_l2 += __shfl_down_sync(FULL_MASK, q_l2, 1);
#endif

#if USE_L2_DIST_
    dist = sqrt(dist);
#elif USE_IP_DIST_
    dist = -dist;
#elif USE_COS_DIST_
    p_l2 = sqrt(p_l2);
    q_l2 = sqrt(q_l2);
    dist = dist / (p_l2 * q_l2);
    if (t_id == 0) {
        if(!(dist == dist)){
            dist = 2;
        } else {
            dist = 1 - dist;
        }
    }
#endif


            if(t_id == 0){
                crt_distance[j].first = dist;
                crt_distance[j].second = target_point_id;

                (distance_matrix + (b_id * num_of_points_one_batch + j) * num_of_points_one_batch)[i].first = dist;
                (distance_matrix + (b_id * num_of_points_one_batch + j) * num_of_points_one_batch)[i].second = crt_point_id;
            }
    
        }

        if(t_id == 0){
            crt_distance[i].first = Max;
            crt_distance[i].second = crt_point_id;
        }

    }
    
}
