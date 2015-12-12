// First naive implementation
__kernel void myGEMM(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C,int transA,
                      __local float* dataCacheA) {

    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    int resultIndex = gid*M;
    int iters = 0;
    int fillFlag = K%4;

    if (gid < N){
      for(int j=0;j<M;j++){
        //use local Memory to cache a_trans's entire col
        int offset = j*K;
        for (int k = lid;k < K;k += lsize ){
          //dataCacheA[k] = A[k+offset];
          *((__local float*)&dataCacheA[k]) = *((const __global float*)&A[k+offset]);
        }
        barrier( CLK_LOCAL_MEM_FENCE );


        int indexB = gid*K;
        int indexA = 0;
        float sum =0.0f;
        if (fillFlag==0){
          for (int h =0;h<K;h+=4){
            //sum +=dot(vload4(0,indexB),vload4(0,indexA));
            float4 tmpb = (*((__global float4*)&B[indexB]));
            float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));
            sum += tmpa.x * tmpb.x;
            sum += tmpa.y * tmpb.y;
            sum += tmpa.z * tmpb.z;
            sum += tmpa.w * tmpb.w;
            //sum += dot(tmpa,tmpb);
            indexA+=4;
            indexB+=4;
          }
        }
        else{
          for(int h=0;h<K-fillFlag;h+=4){
            float4 tmpb = (*((__global float4*)&B[indexB]));
            float4 tmpa = (*((__local float4*)&dataCacheA[indexA]));
            //sum += dot(tmpa,tmpb);
            sum += tmpa.x * tmpb.x;
            sum += tmpa.y * tmpb.y;
            sum += tmpa.z * tmpb.z;
            sum += tmpa.w * tmpb.w;
            indexA+=4;
            indexB+=4;
          }
          for (int r=0;r<fillFlag;r++){
            sum += B[indexB+r]*dataCacheA[indexA+r];
          }
        }
        C[resultIndex+j]=sum;
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
}
