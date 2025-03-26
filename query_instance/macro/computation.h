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