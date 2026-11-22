#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdint>
int main(int argc,char**argv){
  const int M=2048, N=15360, K=3840;           // the video FFN up-projection
  hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
  int8_t *A,*B; int32_t *C; hipMalloc(&A,(size_t)M*K); hipMalloc(&B,(size_t)K*N); hipMalloc(&C,(size_t)M*N*4);
  hipMemset(A,1,(size_t)M*K); hipMemset(B,1,(size_t)K*N);
  rocblas_handle h; rocblas_create_handle(&h);
  int32_t al=1,be=0;
  auto run=[&]{ return rocblas_gemm_ex(h,rocblas_operation_none,rocblas_operation_none,N,M,K,&al,B,rocblas_datatype_i8_r,N,A,rocblas_datatype_i8_r,K,&be,C,rocblas_datatype_i32_r,N,C,rocblas_datatype_i32_r,N,rocblas_datatype_i32_r,rocblas_gemm_algo_standard,0,0); };
  if(run()!=rocblas_status_success){ printf("%s: rocblas i8 gemm refused\n",p.gcnArchName); return 1; }
  hipDeviceSynchronize();
  hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
  const int reps=50; hipEventRecord(a); for(int i=0;i<reps;++i) run(); hipEventRecord(b); hipEventSynchronize(b);
  float ms; hipEventElapsedTime(&ms,a,b); ms/=reps;
  printf("%s CUs=%d clock=%d MHz: int8 GEMM %dx%dx%d %.3f ms = %.1f TOPS\n",p.gcnArchName,p.multiProcessorCount,p.clockRate/1000,M,N,K,ms,2.0*M*N*K/ms/1e9);
}
