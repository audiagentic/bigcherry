#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void rd(const uint4* p, size_t n, unsigned* out){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; unsigned a=0; for(;i<n;i+=s){uint4 v=p[i]; a^=v.x^v.y^v.z^v.w;} if(a==0x12345678u) out[0]=a; }
int main(){ for (size_t mb : {64,300,1200}) { size_t bytes=mb<<20, n=bytes/16; uint4* d; unsigned* o; hipMalloc(&d,bytes); hipMalloc(&o,4); hipMemset(d,1,bytes);
 hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); for(int w=0;w<3;++w) rd<<<4096,256>>>(d,n,o);
 hipEventRecord(a); for(int k=0;k<20;++k) rd<<<4096,256>>>(d,n,o); hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms,a,b);
 printf("%5zu MB read: %.0f GB/s (%.0f us per pass)\n",mb,20.0*bytes/(ms*1e-3)/1e9,ms*1000/20); hipFree(d); hipFree(o);} }
