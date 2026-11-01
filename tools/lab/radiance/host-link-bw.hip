// RAD23: what one card's PCIe link gives the remote-MoE path, in seconds: a kernel reading and writing pinned, mapped host memory (how the
// engine reads helper results and helpers write them; no copies) and hipMemcpy both ways. 21 MiB is two helpers' result planes for a
// 2048-row step. The card's own port reports x16 even in a x4 slot; this measures the slot.
// Build: hipcc -O3 --offload-arch=gfx1030,gfx1100,gfx1201 tools/lab/radiance/host-link-bw.hip -o host-link-bw
// Run:   for g in 0 1 2 3; do HIP_VISIBLE_DEVICES=$g ./host-link-bw 21; done
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
__global__ void rd(const uint4* p, size_t n, unsigned* out){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; unsigned a=0; for(;i<n;i+=s){uint4 v=p[i]; a^=v.x^v.y^v.z^v.w;} if(a==0x12345678u) out[0]=a; }
__global__ void wr(uint4* p, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; for(;i<n;i+=s) p[i]=make_uint4(i,i,i,i); }
int main(int c,char**v){ size_t mb=c>1?atoi(v[1]):21, bytes=mb<<20, n=bytes/16; int dev; hipGetDevice(&dev);
 void* h; hipHostMalloc(&h,bytes,hipHostMallocMapped|hipHostMallocCoherent); void* dh; hipHostGetDevicePointer(&dh,h,0);
 void* d; hipMalloc(&d,bytes); unsigned* o; hipMalloc(&o,4); hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b); float ms;
 auto t=[&](auto f){ for(int w=0;w<2;++w) f(); hipEventRecord(a); for(int k=0;k<10;++k) f(); hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&ms,a,b); return ms*100.0f; };
 float r=t([&]{ rd<<<1024,256>>>((const uint4*)dh,n,o); }), w=t([&]{ wr<<<1024,256>>>((uint4*)dh,n); }),
       h2d=t([&]{ hipMemcpyAsync(d,h,bytes,hipMemcpyHostToDevice,0); }), d2h=t([&]{ hipMemcpyAsync(h,d,bytes,hipMemcpyDeviceToHost,0); });
 printf("%zu MiB: kernel read host %.1f GB/s (%.0f us), kernel write host %.1f GB/s (%.0f us), copy H2D %.1f GB/s, D2H %.1f GB/s\n", mb,
   bytes/(r*1e3), r, bytes/(w*1e3), w, bytes/(h2d*1e3), bytes/(d2h*1e3)); }
