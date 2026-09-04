#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
int main(){
  const int n=262144; size_t b=n*4;
  hipSetDevice(0); hipDeviceEnablePeerAccess(1,0);
  hipSetDevice(1); hipDeviceEnablePeerAccess(0,0);
  std::vector<float> h(n), back(n);
  for(int i=0;i<n;i++) h[i]=i*0.125f+3.0f;
  hipSetDevice(0); float*p0; hipMalloc(&p0,b); hipMemcpy(p0,h.data(),b,hipMemcpyHostToDevice);
  hipSetDevice(1); float*p1; hipMalloc(&p1,b);
  hipStream_t s0; hipSetDevice(0); hipStreamCreateWithFlags(&s0,hipStreamNonBlocking);
  auto reset=[&]{ hipSetDevice(1); hipMemset(p1,0,b); hipDeviceSynchronize(); };
  auto check=[&](const char*tag,hipError_t e){
    hipSetDevice(1); hipDeviceSynchronize();
    hipMemcpy(back.data(),p1,b,hipMemcpyDeviceToHost);
    int bad=0; for(int i=0;i<n;i++) if(fabsf(back[i]-h[i])>1e-6f) bad++;
    printf("%-46s call=%-16s data=%s\n", tag, hipGetErrorString(e), bad? "CORRUPT":"OK"); };
  reset(); hipSetDevice(0);
  check("hipMemcpyAsync D2D  stream, cur=src", hipMemcpyAsync(p1,p0,b,hipMemcpyDeviceToDevice,s0));
  reset(); hipSetDevice(0);
  check("hipMemcpyAsync Default stream, cur=src", hipMemcpyAsync(p1,p0,b,hipMemcpyDefault,s0));
  reset(); hipSetDevice(0);
  check("hipMemcpyAsync D2D  NULL stream, cur=src", hipMemcpyAsync(p1,p0,b,hipMemcpyDeviceToDevice,0));
  reset(); hipSetDevice(0);
  check("hipMemcpyPeerAsync stream", hipMemcpyPeerAsync(p1,1,p0,0,b,s0));
  reset(); hipSetDevice(0);
  check("hipMemcpy (sync) D2D cur=src [known-good]", hipMemcpy(p1,p0,b,hipMemcpyDeviceToDevice));
  return 0;
}
