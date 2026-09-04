#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
__global__ void k_reduce(const float*s,const float*e,float*r,int n){
  int g=blockIdx.x*blockDim.x+threadIdx.x,st=gridDim.x*blockDim.x;
  for(int i=g;i<n;i+=st) r[i]=s[i]+e[i];
}
__global__ void k_busy(float*p,int n,int it){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){float v=p[i];for(int k=0;k<it;++k)v=v*1.0000001f+1e-8f;p[i]=v;}
}
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**argv){
  int reps=argc>1?atoi(argv[1]):100;
  std::vector<int> sizes={7680,15360,30720,61440,122880,262144};
  std::vector<double> A={25.48,29.52,42.84,63.36,105.44,302.40};
  float *snd[2],*exch[2],*rcv[2],*busy[2]; hipStream_t s[2],c[2];
  printf("%-9s %-9s %10s %10s %9s %11s %10s %9s\n","elements","bytes","A host","C dma","C vs A","busy-alone","D ovlp","hidden");
  for(size_t si=0;si<sizes.size();++si){
    int n=sizes[si]; size_t b=(size_t)n*4;
    std::vector<float> h0(n),h1(n),bk(n);
    for(int i=0;i<n;i++){h0[i]=i*0.001f+1.0f;h1[i]=i*0.002f-0.5f;}
    for(int r=0;r<2;++r){hipSetDevice(r);hipDeviceEnablePeerAccess(1-r,0);
      hipStreamCreateWithFlags(&s[r],hipStreamNonBlocking);
      hipStreamCreateWithFlags(&c[r],hipStreamNonBlocking);
      hipMalloc(&snd[r],b);hipMalloc(&exch[r],b);hipMalloc(&rcv[r],b);hipMalloc(&busy[r],b);
      hipMemset(exch[r],0,b);hipMemset(busy[r],0,b);}
    hipSetDevice(0);hipMemcpy(snd[0],h0.data(),b,hipMemcpyHostToDevice);
    hipSetDevice(1);hipMemcpy(snd[1],h1.data(),b,hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    int TH=256, BL=(n+TH-1)/TH; if(BL>256) BL=256;
    auto verify=[&](const char*t){int bad=0;
      for(int r=0;r<2;++r){hipSetDevice(r);hipDeviceSynchronize();
        hipMemcpy(bk.data(),rcv[r],b,hipMemcpyDeviceToHost);
        for(int i=0;i<n;i++) if(fabsf(bk[i]-(h0[i]+h1[i]))>2e-4f) bad++;}
      if(bad) printf("   [%s] INCORRECT bad=%d\n",t,bad); return bad==0;};
    // C: async D2D push (cur=src, explicit stream) then local reduce
    std::vector<double> tc;
    for(int it=0;it<reps+1;++it){
      auto t0=std::chrono::high_resolution_clock::now();
      for(int r=0;r<2;++r){hipSetDevice(r);hipMemcpyAsync(exch[1-r],snd[r],b,hipMemcpyDeviceToDevice,s[r]);}
      for(int r=0;r<2;++r){hipSetDevice(r);hipStreamSynchronize(s[r]);}
      for(int r=0;r<2;++r){hipSetDevice(r);k_reduce<<<BL,TH,0,s[r]>>>(snd[r],exch[r],rcv[r],n);}
      for(int r=0;r<2;++r){hipSetDevice(r);hipStreamSynchronize(s[r]);}
      auto t1=std::chrono::high_resolution_clock::now();
      if(it) tc.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());}
    bool okC=verify("C");
    // busy alone
    const int BUSY=1500; std::vector<double> tb;
    for(int it=0;it<reps/4+1;++it){
      auto t0=std::chrono::high_resolution_clock::now();
      for(int r=0;r<2;++r){hipSetDevice(r);k_busy<<<BL,TH,0,s[r]>>>(busy[r],n,BUSY);}
      for(int r=0;r<2;++r){hipSetDevice(r);hipStreamSynchronize(s[r]);}
      auto t1=std::chrono::high_resolution_clock::now();
      if(it) tb.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());}
    // D: copy on separate stream concurrent with busy
    for(int r=0;r<2;++r){hipSetDevice(r);hipMemset(exch[r],0,b);} hipDeviceSynchronize();
    std::vector<double> td;
    for(int it=0;it<reps/4+1;++it){
      auto t0=std::chrono::high_resolution_clock::now();
      for(int r=0;r<2;++r){hipSetDevice(r);
        hipMemcpyAsync(exch[1-r],snd[r],b,hipMemcpyDeviceToDevice,c[r]);
        k_busy<<<BL,TH,0,s[r]>>>(busy[r],n,BUSY);}
      for(int r=0;r<2;++r){hipSetDevice(r);hipStreamSynchronize(c[r]);hipStreamSynchronize(s[r]);}
      for(int r=0;r<2;++r){hipSetDevice(r);k_reduce<<<BL,TH,0,s[r]>>>(snd[r],exch[r],rcv[r],n);}
      for(int r=0;r<2;++r){hipSetDevice(r);hipStreamSynchronize(s[r]);}
      auto t1=std::chrono::high_resolution_clock::now();
      if(it) td.push_back(std::chrono::duration<double,std::micro>(t1-t0).count());}
    bool okD=verify("D");
    double C=med(tc),B=med(tb),Dm=med(td);
    double serial=B+C, hidden=serial-Dm;
    printf("%-9d %-9zu %10.2f %10.2f %8.1f%% %11.2f %10.2f %8.2f  %s%s\n",
      n,b,A[si],C,100.0*(C-A[si])/A[si],B,Dm,hidden, okC?"":"C_BAD ", okD?"":"D_BAD ");
    for(int r=0;r<2;++r){hipSetDevice(r);hipFree(snd[r]);hipFree(exch[r]);hipFree(rcv[r]);hipFree(busy[r]);
      hipStreamDestroy(s[r]);hipStreamDestroy(c[r]);}
  }
  return 0;
}
