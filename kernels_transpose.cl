__kernel void MatrixTranspose(const int rows,
                              const int cols,
                              __global float* matrix,
                              __global float* matrixTranspose)
{
    /*
    int gid = get_global_id(0);
    int indexSrc = cols*gid;
    int iters = cols >> 2;
    int offset = 0;

    for(int i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        float4 tmp1 = (*((__global float4*)&matrix[indexSrc]));

        matrixTranspose[gid+offset] = tmp1.x;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.y;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.z;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.w;
        offset += rows;

        indexSrc += 4;
    }

    */
    int gid = get_global_id(0);
    int indexSrc = rows*gid;
    int offset =0;
    int leftFlag = rows%4;
    int iters=0;

    //M(rows can be divied by 4)
    if (leftFlag == 0){
      iters = rows >> 2;
      for (int i=0;i<iters;i++){
          float4 tmp1 = (*((__global float4*)&matrix[indexSrc]));
          matrixTranspose[gid+offset] = tmp1.x;
          offset += cols;
          matrixTranspose[gid+offset] = tmp1.y;
          offset += cols;
          matrixTranspose[gid+offset] = tmp1.z;
          offset += cols;
          matrixTranspose[gid+offset] = tmp1.w;
          offset += cols;
          indexSrc += 4;
      }
    }
    //M(rows can't be divied by 4)
    else{
      iters = (rows - leftFlag) >> 2;
      for (int i=0;i<iters;i++){
          float4 tmp1 = (*((__global float4*)&matrix[indexSrc]));
          matrixTranspose[gid+offset] = tmp1.x;
          offset += cols;
          matrixTranspose[gid+offset] = tmp1.y;
          offset += cols;
          matrixTranspose[gid+offset] = tmp1.z;
          offset += cols;
          matrixTranspose[gid+offset] = tmp1.w;
          offset += cols;
          indexSrc += 4;
      }
      for (int j = 0;j<leftFlag;j++){
          matrixTranspose[gid+offset] = matrix[indexSrc+j];
          offset += cols;
      }
    }
}
