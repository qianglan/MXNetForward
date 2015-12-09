// First naive implementation
__kernel void myGEMM(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,int transA) {
    /*
    const int group_index_0 = get_group_id(0);
    const int group_index_1 = get_group_id(1);


    float acc = 0.0f;
    for (int k =0;k<K;k++){
      acc+=A[k*M+group_index_0]*B[group_index_1*K+k];
    }

    C[group_index_1*M+group_index_0] = acc;
    */

    //try to use float4
    const int group_index_0 = get_group_id(0);
    const int group_index_1 = get_group_id(1);


    float4 sum = (float4)0.0f;
    float4 matrixARow = (float4)1.0f;
    uint aOffset = group_index_0;
    uint fillFlag = K%4;
    B += group_index_1*K;
    if (fillFlag==0){
    //K can be divied by 4
      for(int i = 0; i < K; i+=4)
      {
        /* code */
        //get 4 elements of A in a row
        matrixARow.x = A[aOffset];
        aOffset += M;
        //fillFlag =0;

        matrixARow.y = A[aOffset];
        aOffset += M;

        matrixARow.z = A[aOffset];
        aOffset += M;

        matrixARow.w = A[aOffset];
        aOffset += M;

        sum += vload4(0,B)*matrixARow;
        B += 4;
      }
      C[group_index_1*M+group_index_0] = sum.x + sum.y +sum.z +sum.w;
    }
    else{
    //K can't be divied by 4 , left fillFlag nums behind
      for(int i = 0; i < K - fillFlag; i+=4)
      {
        /* code */
        //get 4 elements of A in a row
        matrixARow.x = A[aOffset];
        aOffset += M;
        //fillFlag =0;

        matrixARow.y = A[aOffset];
        aOffset += M;

        matrixARow.z = A[aOffset];
        aOffset += M;

        matrixARow.w = A[aOffset];
        aOffset += M;

        sum += vload4(0,B)*matrixARow;
        B += 4;
      }
      //compute the rest part
      for (int m =0;m<fillFlag;m++)
      {
        sum.x += B[m]*A[aOffset+m];
      }
      C[group_index_1*M+group_index_0] = sum.x + sum.y +sum.z +sum.w;
    }
}
