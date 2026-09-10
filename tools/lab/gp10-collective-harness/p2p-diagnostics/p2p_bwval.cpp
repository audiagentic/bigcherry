#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#define CHK(c) do { hipError_t r=(c); if(r!=hipSuccess) printf("  !! %d: %s\n", __LINE__, hipGetErrorString(r)); } while(0)
int main(){
  const int devA=0, devB=1; const size_t bytes=1ull<<20; const int n=bytes/4;
  int ca=0,cb=0; hipDeviceCanAccessPeer(&ca,devA,devB); hipDeviceCanAccessPeer(&cb,devB,devA);
  hipSetDevice(devA); hipDeviceEnablePeerAccess(devB,0);
  hipSetDevice(devB); hipDeviceEnablePeerAccess(devA,0);
  std::vector<float> h(n), back(n);
  for(int i=0;i<n;i++) h[i]=i*0.125f+3.0f;
  hipSetDevice(devA); void*dA; CHK(hipMalloc(&dA,bytes));
  hipSetDevice(devB); void*dB; CHK(hipMalloc(&dB,bytes));
  // seed dA, zero dB  (exact p2p_bw.cpp state: current device is devB here)
  hipSetDevice(devA); CHK(hipMemcpy(dA,h.data(),bytes,hipMemcpyHostToDevice));
  hipSetDevice(devB); CHK(hipMemset(dB,0,bytes)); CHK(hipDeviceSynchronize());
  // >>> EXACT p2p_bw.cpp call: current device devB, hipMemcpyDefault, dB<-dA
  CHK(hipMemcpy(dB,dA,bytes,hipMemcpyDefault));
  CHK(hipDeviceSynchronize());
  CHK(hipMemcpy(back.data(),dB,bytes,hipMemcpyDeviceToHost));
  int bad=0; for(int i=0;i<n;i++) if(fabsf(back[i]-h[i])>1e-6f) bad++;
  printf("p2p_bw.cpp exact config (cur=devB, dB<-dA, Default): bad=%d/%d first=%.3f want=%.3f %s\n",
         bad,n,back[0],h[0], bad? "*** CORRUPT -- benchmark measured garbage ***":"OK -- number is real");
  // and the mirrored case: current device devA
  hipSetDevice(devB); CHK(hipMemset(dB,0,bytes)); CHK(hipDeviceSynchronize());
  hipSetDevice(devA); CHK(hipMemcpy(dB,dA,bytes,hipMemcpyDefault)); CHK(hipDeviceSynchronize());
  CHK(hipMemcpy(back.data(),dB,bytes,hipMemcpyDeviceToHost));
  bad=0; for(int i=0;i<n;i++) if(fabsf(back[i]-h[i])>1e-6f) bad++;
  printf("same copy but current device = devA:                 bad=%d/%d first=%.3f want=%.3f %s\n",
         bad,n,back[0],h[0], bad?"*** CORRUPT ***":"OK");
  return 0;
}
