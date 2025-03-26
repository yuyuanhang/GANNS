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