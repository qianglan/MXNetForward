// First naive implementation
__kernel void myGEMM(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,int transA) {

    const int group_index = get_group_id(0);
    const int local_index = get_local_id(0);
    const int l0size = get_local_size(0);

    float4 sum = (float4)0.0f;
    float4 matrixARow = (float4)1.0f;
    uint aOffset = 0;
    uint fillFlag = K%4;


    // the col of this work item
    int local_cols = group_index*l0size + local_index;
    int local_cols_B = local_cols*K;
    int local_cols_C = local_cols*M;
    //get the woring item
    if ( local_cols < N){
      //for every element in clo local_cols
      for (int i = 0;i < M;i++){
        sum = (float4)0.0f;
        matrixARow = (float4)0.0f;
        aOffset = i;
        //K can be divied by 4
        if (fillFlag == 0){
          for (int j=0;j<K;j+=4){
            //get 4 elements of A in a row
            matrixARow.x = A[aOffset];
            aOffset+=M;
            matrixARow.y = A[aOffset];
            aOffset+=M;
            matrixARow.z = A[aOffset];
            aOffset+=M;
            matrixARow.w = A[aOffset];
            aOffset+=M;
            sum += vload4(0,B+local_cols_B+j)*matrixARow;
          }
          C[i + local_cols_C] = sum.x + sum.y +sum.z+sum.w;
        }
        //K can't be divied by 4 , left fillFlag nums behind
        else
        {
          for (int j=0;j<K - fillFlag;j+=4){
            //get 4 elements of A in a row
            matrixARow.x = A[aOffset];
            aOffset+=M;
            matrixARow.y = A[aOffset];
            aOffset+=M;
            matrixARow.z = A[aOffset];
            aOffset+=M;
            matrixARow.w = A[aOffset];
            aOffset+=M;
            sum += vload4(0,B+local_cols_B+j)*matrixARow;
          }
          //compute the rest part
          for (int r=0;r<fillFlag;r++){
            sum.x += B[local_cols_B + K - fillFlag +r]*A[aOffset+r*M];
          }
          C[i + local_cols_C] = sum.x + sum.y +sum.z+sum.w;
        }
      }
    }

}
