// VTP14: exact Qwen decode GDN math, partial original BF16 checkpoint:
// real depthwise-conv weights, A_log, dt_bias and RMS norm gains.
// Preprojected Q|K|V, a/b and z inputs are deterministic BF16 synthetic.
// One head-local concurrent HIP worker per physical GPU (2x gfx1100, 1x
// gfx1201), 4/4/8 key and 12/12/24 value heads. Not full model serving,
// not the original QKV GEMM, not a prefill-KKT implementation.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#define HC(op) do { hipError_t e_=(op); if(e_!=hipSuccess) { \
 fprintf(stderr,"HIP %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e_));exit(2); \
} } while(0)
constexpr int H=128;
constexpr float EPS=1.e-6f;
constexpr float QSCALE=0.08838834764831845f; // 1/sqrt(128)
constexpr std::array<int,3> KEYS{4,4,8};
constexpr std::array<int,3> VALUES{12,12,24};
constexpr std::array<int,3> KEY_OFF{0,4,8};
constexpr std::array<int,3> VALUE_OFF{0,12,24};
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
float percentile(std::vector<double> v,float p) {
  std::sort(v.begin(),v.end());
  const size_t rank=std::max(size_t(1),std::min(v.size(),
                                    static_cast<size_t>(std::ceil(p*v.size()))));
  return static_cast<float>(v[rank-1]);
}
static const char* arg(int argc,char** argv,const char* name) {
  for(int i=1;i<argc;++i) if(strcmp(argv[i],name)==0) {
    if(i+1==argc)throw std::runtime_error(std::string("missing ")+name);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing required ")+name);
}
static int bounded(int argc,char** argv,const char* name,int maxv) {
  char* end=nullptr;
  const long v=strtol(arg(argc,argv,name),&end,10);
  if(!end || *end || v<1 || v>maxv)throw std::runtime_error("invalid bounded argument");
  return static_cast<int>(v);
}
template<typename T>std::vector<T> read_exact(const char* path,size_t count) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=sizeof(T)*count)
    throw std::runtime_error(std::string("wrong binary size: ")+path);
  f.seekg(0);
  std::vector<T> v(count);
  f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));
  if(!f) throw std::runtime_error(std::string("truncated binary: ")+path);
  return v;
}
std::array<int,3> devices() {
  int n=0;HC(hipGetDeviceCount(&n));std::vector<int> xtx,rdna4;
  for(int i=0;i<n;++i) {
    hipDeviceProp_t p{};HC(hipGetDeviceProperties(&p,i));
    if(strncmp(p.gcnArchName,"gfx1100",7)==0)xtx.push_back(i);
    else if(strncmp(p.gcnArchName,"gfx1201",7)==0)rdna4.push_back(i);
  }
  if(xtx.size()!=2 || rdna4.size()!=1)
    throw std::runtime_error("require exactly 2 visible gfx1100 and one gfx1201");
  return {xtx[0],xtx[1],rdna4[0]};
}
__host__ __device__ static float input_value(int row,int global_c) {
  return static_cast<float>((row*7 + global_c*11)%23 - 11)*(1.0f/64.0f);
}
__host__ __device__ static float scalar_value(int row,int index,int mode) {
  const int offset = mode==0?5:(mode==1?11:19);
  return static_cast<float>((row*13+index*17+offset)%31-15)*(1.0f/64.0f);
}
__device__ static float fbf16(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x)<<16);
}
__device__ static uint16_t bf16rn(float x) {
  const uint32_t bits=__float_as_uint(x);
  return static_cast<uint16_t>((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__device__ static float sigmoid(float x) {
  return x>=0 ? 1.0f/(1.0f+expf(-x)) : expf(x)/(1.0f+expf(x));
}
__device__ static float softplus(float x) {
  return x>20 ? x : log1pf(expf(-fabsf(x)))+fmaxf(x,0.0f);
}
__device__ static float silu(float x) {
  return x*sigmoid(x);
}
// Preprojected x is BF16-representable, already grouped Q|K|V per rank.
// Every thread owns one convolution channel and its entire causal history.
__global__ void conv4(const float* x,const uint16_t* w,uint16_t* conv,
                      int channels,int rows) {
  for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<channels;c+=gridDim.x*blockDim.x)
    for(int t=0;t<rows;++t) {
      float y=0.0f;
      for(int tap=0;tap<4;++tap) {
        const int prev=t+tap-3;
        if(prev>=0)
          y+=x[static_cast<size_t>(prev)*channels+c]*
              fbf16(w[static_cast<size_t>(c)*4+tap]);
      }
      conv[static_cast<size_t>(t)*channels+c]=bf16rn(silu(y));
    }
}
// Exact ref: norm over 128 channels (L2, not RMS), q scaled 1/sqrt(K).
__global__ void qk_norm(const uint16_t* conv,float* qnorm,float* knorm,
                        int channels,int keys,int rows) {
  const int d=threadIdx.x;
  const int head=blockIdx.x%keys;
  const int kind=(blockIdx.x/keys)%2;
  const int row=blockIdx.x/(2*keys);
  if(row>=rows || d>=H)return;
  __shared__ float ss[H];
  const int base=kind*keys*H+head*H;
  const float x=fbf16(conv[static_cast<size_t>(row)*channels+base+d]);
  ss[d]=x*x;__syncthreads();
  for(int s=H/2;s>0;s>>=1) {
    if(d<s)ss[d]+=ss[d+s];
    __syncthreads();
  }
  const float norm=rsqrtf(ss[0]+EPS)*(kind==0?QSCALE:1.0f);
  float* out=kind==0?qnorm:knorm;
  out[(static_cast<size_t>(row)*keys+head)*H+d]=x*norm;
}
// One block per VALUE head (24 concurrent blocks on R9700). Each thread
// owns one value coordinate and its 128-wide state row, no cross-block
// communication. Gate/decay and beta are computed from original BF16
// A_log/dt_bias and synthetic projected a/b; z is synthetic. The
// per-value-head RMS norm is a workgroup reduction.
__global__ void delta_qwen(const uint16_t* conv,const float* q,const float* k,
                           const float* coeff,float* state,float* output,
                           int keys,int value_heads,int value_offset,
                           int channels,int rows) {
  const int h=blockIdx.x,e=threadIdx.x,hk=h/3;
  if(h>=value_heads || hk>=keys || e>=H)return;
  __shared__ float norm[H];
  const int global_h=value_offset+h;
  for(int t=0;t<rows;++t) {
    const float a=scalar_value(t,global_h,0);
    const float b=scalar_value(t,global_h,1);
    const float alog=coeff[global_h],dt=coeff[48+global_h];
    const float g=-expf(alog)*softplus(a+dt);
    const float dec=expf(g),beta=sigmoid(b);
    const size_t stbase=(static_cast<size_t>(h)*H+e)*H;
    const size_t keybase=(static_cast<size_t>(t)*keys+hk)*H;
    float predicted=0.0f;
    for(int d=0;d<H;++d) {
      float val=state[stbase+d]*dec;
      state[stbase+d]=val;
      predicted+=val*k[keybase+d];
    }
    const size_t vch=static_cast<size_t>(t)*channels+2*keys*H+h*H+e;
    const float v=fbf16(conv[vch]);
    const float delta=beta*(v-predicted);
    float raw=0.0f;
    for(int d=0;d<H;++d) {
      const float updated=state[stbase+d]+delta*k[keybase+d];
      state[stbase+d]=updated;
      raw+=updated*q[keybase+d];
    }
    norm[e]=raw*raw;
    __syncthreads();
    for(int stride=H/2;stride>0;stride>>=1) {
      if(e<stride)norm[e]+=norm[e+stride];
      __syncthreads();
    }
    const float rms=rsqrtf(norm[0]*(1.0f/H)+EPS);
    const float z=scalar_value(t,global_h*H+e,2);
    output[(static_cast<size_t>(h)*rows+t)*H+e]=raw*rms*coeff[96+e]*silu(z);
    __syncthreads();
  }
}
struct Rank {
  int id=-1,keys=0,values=0,keyoff=0,valoff=0,channels=0,rows=0;
  hipStream_t stream=nullptr;hipEvent_t start=nullptr,end=nullptr;
  float *x=nullptr,*q=nullptr,*k=nullptr,*state=nullptr,*out=nullptr,*coef=nullptr;
  uint16_t *w=nullptr,*conv=nullptr;
  std::vector<float> actual,expected;
  std::vector<double> latency;
  double max_abs=0.0,max_rel=0.0;
};
void setup(Rank& r,const char* wpath,const char* refpath,
           const std::vector<float>& coefficients) {
  HC(hipSetDevice(r.id));HC(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&r.start));HC(hipEventCreate(&r.end));
  const auto weights=read_exact<uint16_t>(wpath,static_cast<size_t>(r.channels)*4);
  const size_t refcount=static_cast<size_t>(r.values)*r.rows*H;
  r.expected=read_exact<float>(refpath,refcount);
  r.actual.resize(refcount);
  std::vector<float> input(static_cast<size_t>(r.rows)*r.channels);
  const std::array<int,3> starts{r.keyoff*H,2048+r.keyoff*H,4096+r.valoff*H};
  const std::array<int,3> sizes{r.keys*H,r.keys*H,r.values*H};
  std::vector<int> indices;
  for(int i=0;i<3;++i)for(int k=0;k<sizes[i];++k)indices.push_back(starts[i]+k);
  if(indices.size()!=static_cast<size_t>(r.channels))
    throw std::runtime_error("rank-local QKV offset coverage mismatch");
  for(int t=0;t<r.rows;++t)
    for(int c=0;c<r.channels;++c)
      input[static_cast<size_t>(t)*r.channels+c]=input_value(t,indices[c]);
  const size_t conv_count=static_cast<size_t>(r.rows)*r.channels;
  const size_t key_count=static_cast<size_t>(r.rows)*r.keys*H;
  const size_t state_count=static_cast<size_t>(r.values)*H*H;
  HC(hipMalloc(reinterpret_cast<void**>(&r.w),weights.size()*sizeof(uint16_t)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.x),input.size()*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.conv),conv_count*sizeof(uint16_t)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.q),key_count*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.k),key_count*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.state),state_count*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.out),refcount*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.coef),coefficients.size()*sizeof(float)));
  HC(hipMemcpyAsync(r.w,weights.data(),weights.size()*2,hipMemcpyHostToDevice,r.stream));
  HC(hipMemcpyAsync(r.x,input.data(),input.size()*4,hipMemcpyHostToDevice,r.stream));
  HC(hipMemcpyAsync(r.coef,coefficients.data(),coefficients.size()*4,
                    hipMemcpyHostToDevice,r.stream));
  HC(hipStreamSynchronize(r.stream));
}
void launch(Rank& r) {
  HC(hipSetDevice(r.id));
  HC(hipEventRecord(r.start,r.stream));
  HC(hipMemsetAsync(r.state,0,static_cast<size_t>(r.values)*H*H*4,r.stream));
  conv4<<<std::min(1024,(r.channels+255)/256),256,0,r.stream>>>(
      r.x,r.w,r.conv,r.channels,r.rows);
  qk_norm<<<r.rows*2*r.keys,H,0,r.stream>>>(
      r.conv,r.q,r.k,r.channels,r.keys,r.rows);
  delta_qwen<<<r.values,H,0,r.stream>>>(
      r.conv,r.q,r.k,r.coef,r.state,r.out,
      r.keys,r.values,r.valoff,r.channels,r.rows);
  HC(hipGetLastError());
  HC(hipEventRecord(r.end,r.stream));
}
void check(Rank& r) {
  HC(hipSetDevice(r.id));
  HC(hipMemcpy(r.actual.data(),r.out,r.actual.size()*sizeof(float),
               hipMemcpyDeviceToHost));
  for(size_t i=0;i<r.actual.size();++i) {
    const double got=r.actual[i],wanted=r.expected[i];
    const double diff=std::abs(got-wanted);
    r.max_abs=std::max(r.max_abs,diff);
    r.max_rel=std::max(r.max_rel,diff/(1.+std::abs(wanted)));
    // A generous rounding tolerance: sources have BF16 storage boundaries
    // while kernel uses native FP32 expf/rsqrtf versus CPU libm double.
    if(!std::isfinite(got) || diff > 0.002 + 0.02*std::abs(wanted))
      throw std::runtime_error("Qwen GDN original-conv/decay rank-local GPU oracle FAILED");
  }
}
void destroy(Rank& r) {
  HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
  HC(hipFree(r.w));HC(hipFree(r.x));HC(hipFree(r.conv));
  HC(hipFree(r.q));HC(hipFree(r.k));HC(hipFree(r.state));
  HC(hipFree(r.out));HC(hipFree(r.coef));
  HC(hipEventDestroy(r.start));HC(hipEventDestroy(r.end));HC(hipStreamDestroy(r.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const int reps=bounded(argc,argv,"--iterations",30);
    const auto coefficients=read_exact<float>(arg(argc,argv,"--coefficients"),224);
    const auto ids=devices();
    constexpr const char* C[]={"--conv-xtx0","--conv-xtx1","--conv-r9700"};
    constexpr const char* E[]={"--oracle-xtx0","--oracle-xtx1","--oracle-r9700"};
    std::array<Rank,3> ranks{};
    for(int i=0;i<3;++i) {
      Rank& r=ranks[i];r.id=ids[i];r.keys=KEYS[i];r.values=VALUES[i];
      r.keyoff=KEY_OFF[i];r.valoff=VALUE_OFF[i];r.rows=rows;
      r.channels=(2*r.keys+r.values)*H;
      setup(r,arg(argc,argv,C[i]),arg(argc,argv,E[i]),coefficients);
    }
    std::vector<double> wall;
    for(int it=-2;it<reps;++it) {
      const auto start=Clock::now();
      for(int i=0;i<3;++i)launch(ranks[i]);
      for(auto& r:ranks){HC(hipSetDevice(r.id));HC(hipEventSynchronize(r.end));}
      if(it<0)continue;
      wall.push_back(elapsed(start));
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));float ms=0;
        HC(hipEventElapsedTime(&ms,r.start,r.end));
        r.latency.push_back(static_cast<double>(ms)*1000.);
      }
    }
    for(auto& r:ranks)check(r);
    printf("mode,rows,wall_us_p50,wall_us_p95,xtx0_us_p50,xtx1_us_p50,r9700_us_p50,xtx0_max_abs,xtx1_max_abs,r9700_max_abs,oracle\n");
    printf("partial_original_bf16_qwen_gdn,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.8g,%.8g,%.8g,PASS\n",
           rows,percentile(wall,.5),percentile(wall,.95),
           percentile(ranks[0].latency,.5),percentile(ranks[1].latency,.5),
           percentile(ranks[2].latency,.5),
           ranks[0].max_abs,ranks[1].max_abs,ranks[2].max_abs);
    printf("physical_ids=%d,%d,%d; inputs=synthetic_preprojected_BF16; conv_weights=real_BF16; A_log_dt_bias_norm=real_BF16; qwen_recurrent_math=YES; qkv_projection=NO; prefill_kkt=NO; no_p2p=YES\n",
           ids[0],ids[1],ids[2]);
    for(auto& r:ranks)destroy(r);
    return 0;
  } catch(const std::exception& e) {
    fprintf(stderr,"VTP14 Qwen GDN GPU oracle FAILED: %s\n",e.what());
    return 1;
  }
}
