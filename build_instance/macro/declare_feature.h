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