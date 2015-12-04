__kernel void multiply_matrix(__global float * m1,
                __global float * m2,
                __global float * res,
                unsigned long int wm1, unsigned long int wm2) {
    unsigned long int i = get_group_id(0);
    unsigned long int j = get_local_id(0);
    unsigned long int k;
    float sum = 0.0f;
    for(k=0;k<wm1;++k)
    	sum += m1[i*wm1+k] * m2[k*wm2+j];
    res[i*wm2+j] = sum;
}
