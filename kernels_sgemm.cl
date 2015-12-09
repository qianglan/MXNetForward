// First naive implementation
__kernel void myGEMM(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,int transA) {

    /*
    // Thread identifiers
    const int group_index = get_group_id(0);
    const int local_index = get_local_id(0);

    // Compute a single element (loop over K)
    float acc = 0.0f;
    //for (int k=0; k<K; k++) {
    //    acc += A[k*M + local_index] * B[group_index*K + k];
    //}

      // Store the result
    //C[group_index*M + local_index] = acc;
    for (int n=0;n<N;n++){
      for (int m=0;m<M;m++){
        float acc=0.0f;
        for (int k=0;k<K;k++){
          acc+=A[m+k*M]*B[n*K+k];
        }
        C[n*M+m]=acc;
      }
    }
    */

    const int group_index_0 = get_group_id(0);
    const int group_index_1 = get_group_id(1);


    float acc = 0.0f;
    for (int k =0;k<K;k++){
      acc+=A[k*M+group_index_0]*B[group_index_1*K+k];
    }

    C[group_index_1*M+group_index_0] = acc;



}
