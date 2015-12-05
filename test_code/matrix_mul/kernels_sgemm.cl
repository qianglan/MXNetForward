// First naive implementation
__kernel void myGEMM(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,int transA) {

    // Thread identifiers
    const int group_index = get_group_id(0);
    const int local_index = get_local_id(0);

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k*M + local_index] * B[group_index*K + k];
    }
      // Store the result
    C[group_index*M + local_index] += acc;
}
