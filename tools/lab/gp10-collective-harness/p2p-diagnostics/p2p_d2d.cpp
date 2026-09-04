#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#define CHK(c) do { hipError_t r=(c); if(r!=hipSuccess) printf("  !! %d: %s\n", __LINE__, hipGetErrorString(r)); } while(0)
int main(){
  const int D0=0,D1=1; const int n=262144; size_t b=n*4;
  CHK(hipSetDevice(D0)); CHK(hipDeviceEnablePeerAccess(D1,0));
  CHK(hipSetDevice(D1)); CHK(hipDeviceEnablePeerAccess(D0,0));
  std::vector<float> h(n), back(n);
  for(int i=0;i<n;i++) h[i]=i*0.25f+1.0f;
  CHK(hipSetDevice(D0)); float*p0; CHK(hipMalloc(&p0,b));
  CHK(hipSetDevice(D1)); float*p1; CHK(hipMalloc(&p1,b));
  auto chk=[&](const char*tag,int dstdev,float*dst){
    CHK(hipSetDevice(dstdev)); CHK(hipDeviceSynchronize());
    CHK(hipMemcpy(back.data(),dst,b,hipMemcpyDeviceToHost));
    int bad=0; for(int i=0;i<n;i++) if(fabsf(back[i]-h[i])>1e-6f) bad++;
    printf("  %-34s bad=%d/%d first=%.3f (want %.3f) %s\n",tag,bad,n,back[0],h[0],bad?"*** CORRUPT ***":"OK");
  };
  // seed p0 on D0
  CHK(hipSetDevice(D0)); CHK(hipMemcpy(p0,h.data(),b,hipMemcpyHostToDevice));
  CHK(hipSetDevice(D1)); CHK(hipMemset(p1,0,b)); CHK(hipDeviceSynchronize());
  CHK(hipMemcpyPeer(p1,D1,p0,D0,b)); CHK(hipDeviceSynchronize());
  chk("hipMemcpyPeer  dev0->dev1 (push)",D1,p1);
  // reverse
  CHK(hipSetDevice(D1)); CHK(hipMemcpy(p1,h.data(),b,hipMemcpyHostToDevice));
  CHK(hipSetDevice(D0)); CHK(hipMemset(p0,0,b)); CHK(hipDeviceSynchronize());
  CHK(hipMemcpyPeer(p0,D0,p1,D1,b)); CHK(hipDeviceSynchronize());
  chk("hipMemcpyPeer  dev1->dev0 (pull)",D0,p0);
  // D2D via hipMemcpy
  CHK(hipSetDevice(D0)); CHK(hipMemcpy(p0,h.data(),b,hipMemcpyHostToDevice));
  CHK(hipSetDevice(D1)); CHK(hipMemset(p1,0,b)); CHK(hipDeviceSynchronize());
  CHK(hipSetDevice(D0)); CHK(hipMemcpy(p1,p0,b,hipMemcpyDeviceToDevice)); CHK(hipDeviceSynchronize());
  chk("hipMemcpy D2D  dev0->dev1",D1,p1);
  // hipMemcpyDefault (what p2p_bw.cpp timed)
  CHK(hipSetDevice(D0)); CHK(hipMemcpy(p0,h.data(),b,hipMemcpyHostToDevice));
  CHK(hipSetDevice(D1)); CHK(hipMemset(p1,0,b)); CHK(hipDeviceSynchronize());
  CHK(hipSetDevice(D0)); CHK(hipMemcpy(p1,p0,b,hipMemcpyDefault)); CHK(hipDeviceSynchronize());
  chk("hipMemcpy Default dev0->dev1",D1,p1);
  return 0;
}
