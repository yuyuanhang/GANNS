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