// HL02: measure host-map attribute effects on GPU-initiated PCIe reads and writes.
// Laboratory-only. Acquiring the exclusive GPU lease is the caller's responsibility.
// Usage: host_attributes DEVICE coherent|noncoherent|writecombined read|write SECONDS [MIB]
// The "writecombined" flag is a CPU mapping attribute; not proof of a PCIe No-Snoop bit.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
static volatile std::sig_atomic_t requested_stop=0;
static void handle_stop(int){requested_stop=1;}
#include <string>
#include <vector>
#define HIP(x) do {hipError_t e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,hipGetErrorString(e)); return 2;}}while(0)
__global__ void read_host(const uint4* p, size_t n, unsigned* sink) {
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
  unsigned a=0;
  for(;i<n;i+=stride){uint4 x=p[i];a+=x.x^x.y^x.z^x.w;}
  if(a==0xdeadbeefu) atomicAdd(sink,a);
}
__global__ void write_host(uint4* p,size_t n,unsigned seed) {
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=stride) p[i]=make_uint4((unsigned)i^seed,(unsigned)i+seed,(unsigned)i*3u,seed);
}
__global__ void check_host(const uint4* p,size_t n,unsigned seed,unsigned* errors) {
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, stride=(size_t)gridDim.x*blockDim.x;
  unsigned bad=0;
  for(;i<n;i+=stride) {
    uint4 v=p[i];
    bad+=(v.x!=((unsigned)i^seed) || v.y!=(unsigned)i+seed || v.z!=(unsigned)i*3u || v.w!=seed);
  }
  if(bad) atomicAdd(errors,bad);
}
int main(int argc,char**argv) {
  if(argc<5 || argc>6) {
    std::fprintf(stderr,"usage: host_attributes DEVICE coherent|noncoherent|writecombined read|write SECONDS [MIB]\n");return 2;
  }
  int device=std::atoi(argv[1]);
  std::string mode=argv[2],direction=argv[3];
  double seconds=std::atof(argv[4]);
  size_t mib=argc==6?std::strtoul(argv[5],nullptr,10):64;
  if((mode!="coherent"&&mode!="noncoherent"&&mode!="writecombined") ||
      (direction!="read"&&direction!="write") ||seconds<=0 || seconds>25||mib<1||mib>256){return 2;}
  std::signal(SIGTERM, handle_stop);
  std::signal(SIGINT, handle_stop);
  HIP(hipSetDevice(device));
  unsigned flags=hipHostMallocMapped | (mode=="coherent"?hipHostMallocCoherent:mode=="noncoherent"?hipHostMallocNonCoherent:hipHostMallocWriteCombined);
  size_t bytes=mib<<20, n=bytes/sizeof(uint4);
  void* h=nullptr; hipError_t alloc=hipHostMalloc(&h,bytes,flags);
  if(alloc!=hipSuccess){std::fprintf(stderr,"ALLOC_UNSUPPORTED mode=%s error=%s\n",mode.c_str(),hipGetErrorString(alloc));return 3;}
  uint4 *p=nullptr; HIP(hipHostGetDevicePointer((void**)&p,h,0));
  unsigned* derrors=nullptr; HIP(hipMalloc(&derrors,sizeof(unsigned)));
  const unsigned seed=0x91a2b3c4u;
  // Start with full deterministic values for both source and verifier.
  write_host<<<1024,256>>>(p,n,seed);
  HIP(hipDeviceSynchronize());
  HIP(hipMemset(derrors,0,sizeof(unsigned)));
  check_host<<<1024,256>>>(p,n,seed,derrors);
  HIP(hipDeviceSynchronize());
  unsigned errors=~0u; HIP(hipMemcpy(&errors,derrors,sizeof(unsigned),hipMemcpyDeviceToHost));
  if(errors){std::fprintf(stderr,"FAIL initial_GPU_verify=%u mode=%s\n",errors,mode.c_str());return 4;}
  hipEvent_t a,b; HIP(hipEventCreate(&a)); HIP(hipEventCreate(&b));
  double moved=0, device_ms=0;
  auto start=std::chrono::steady_clock::now();
  int rounds=0;
  // Separate launch/serialization from timed device work; fixed batch of four sweeps.
  while(!requested_stop && std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<seconds) {
    HIP(hipEventRecord(a));
    for(int k=0;k<4;k++) {
      if(direction=="read") read_host<<<1024,256>>>(p,n,derrors);
      else write_host<<<1024,256>>>(p,n,seed);
    }
    HIP(hipEventRecord(b));HIP(hipEventSynchronize(b));
    float ms=0;HIP(hipEventElapsedTime(&ms,a,b));
    moved+=4.0*bytes;device_ms+=ms;rounds++;
    hipError_t status=hipGetLastError();
    if(status!=hipSuccess){std::fprintf(stderr,"FAIL kernel: %s\n",hipGetErrorString(status));return 5;}
  }
  HIP(hipMemset(derrors,0,sizeof(unsigned)));
  check_host<<<1024,256>>>(p,n,seed,derrors);
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(&errors,derrors,sizeof(unsigned),hipMemcpyDeviceToHost));
  double wall_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::printf("HL02_RESULT device=%d mode=%s direction=%s flags=%u mib=%zu rounds=%d device_gbps=%.3f wall_gbps=%.3f device_ms=%.2f wall_s=%.3f gpu_word_errors=%u status=%s\n",
      device,mode.c_str(),direction.c_str(),flags,mib,rounds,moved/(device_ms*1e6),moved/(wall_s*1e9),device_ms,wall_s,errors,errors?"FAIL":"PASS");
  HIP(hipEventDestroy(a));HIP(hipEventDestroy(b));HIP(hipFree(derrors));HIP(hipHostFree(h));
  return errors?6:0;
}
