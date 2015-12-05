

void my_sgemm( bool TransA, bool TransB,int M, int N, int K, float Alpha,const float* A, int lda,const float* B, int ldb, float beta, float* C, int ldc){
  if (TransA==true){
    for (int n=0;n<N;n++){
      for (int m=0;m<M;m++){
        float acc=0.0f;
        for (int k=0;k<K;k++){
          acc+=Alpha*A[m*K+k]*B[n*K+k];
        }
        C[n*M+m]=acc+beta*C[n*M+m];
      }
    }
  }

  else {
    for (int n=0;n<N;n++){
      for (int m=0;m<M;m++){
        float acc=0.0f;
        for (int k=0;k<K;k++){
          acc+=Alpha*A[m+k*M]*B[n*K+k];
        }
        C[n*M+m]=acc+beta*C[n*M+m];
      }
    }
  }
}
