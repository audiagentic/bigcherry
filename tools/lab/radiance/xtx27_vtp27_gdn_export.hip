// VTP27 VTP20-based BF16 export; original VTP20 remains untouched.
// VTP20 experimental one-execution-chain original BF16 Qwen3.8 GDN block:
// input RMSNorm from original checkpoint, true original BF16
// QKV/a/b/z GEMVs, depthwise conv, Q/K norm, learned GDN delta,
// gated RMSNorm, on-device head-major -> token-major BF16 conversion,
// true original BF16 output projection (column-sharded 1536/1536/3072),
// three concurrent GPU streams, pinned async D2H, ordered FP32 host sum,
// BF16 residual add, and H2D broadcast. No P2P.
// This does NOT implement 64-layer model execution, cached state,
// chunk-prefill KKT, native optimized GEMM or actual token generation.
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
// Original checkpoint projection GEMV produces rank-local BF16 Q|K|V;
// each thread convolves one channel across its causal sequence.
__global__ void conv4(const uint16_t* x,const uint16_t* w,uint16_t* conv,
                      int channels,int rows) {
  for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<channels;c+=gridDim.x*blockDim.x)
    for(int t=0;t<rows;++t) {
      float y=0.0f;
      for(int tap=0;tap<4;++tap) {
        const int prev=t+tap-3;
        if(prev>=0)
          y+=fbf16(x[static_cast<size_t>(prev)*channels+c])*
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
// communication. Gate/decay/beta/z use actual BF16 projected outputs
// and original checkpoint A_log/dt_bias/norm gain. The
// per-value-head RMS norm is a workgroup reduction.
__global__ void delta_qwen(const uint16_t* conv,const float* q,const float* k,
                           const uint16_t* proj_a,const uint16_t* proj_b,
                           const uint16_t* proj_z,const float* coeff,
                           float* state,float* output,
                           int keys,int value_heads,int value_offset,
                           int channels,int rows) {
  const int h=blockIdx.x,e=threadIdx.x,hk=h/3;
  if(h>=value_heads || hk>=keys || e>=H)return;
  __shared__ float norm[H];
  const int global_h=value_offset+h;
  for(int t=0;t<rows;++t) {
    const float a=fbf16(proj_a[static_cast<size_t>(t)*value_heads+h]);
    const float b=fbf16(proj_b[static_cast<size_t>(t)*value_heads+h]);
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
    const float z=fbf16(proj_z[(static_cast<size_t>(t)*value_heads+h)*H+e]);
    output[(static_cast<size_t>(h)*rows+t)*H+e]=raw*rms*coeff[96+e]*silu(z);
    __syncthreads();
  }
}

// Real original BF16 [N,5120] weight matrix times one shared BF16
// weighted-RMSNorm input [M,5120], row-major. Simple 8-wave GEMV.
__global__ void project_original_bf16(const uint16_t* x,const uint16_t* w,
                                       float* fp32,uint16_t* bf16,
                                       int n,int rows) {
  const int wave=warpSize,lane=threadIdx.x%wave;
  const int out=blockIdx.x*(blockDim.x/wave)+threadIdx.x/wave;
  const int row=blockIdx.y;
  float acc=0.0f;
  if(out<n && row<rows) {
    const uint16_t* weight=w+static_cast<size_t>(out)*5120;
    const uint16_t* input=x+static_cast<size_t>(row)*5120;
    for(int k=lane;k<5120;k+=wave)
      acc+=fbf16(weight[k])*fbf16(input[k]);
  }
  for(int shift=wave/2;shift>=1;shift/=2)
    acc+=__shfl_down(acc,shift);
  if(lane==0 && out<n && row<rows) {
    const size_t idx=static_cast<size_t>(row)*n+out;
    fp32[idx]=acc;
    bf16[idx]=bf16rn(acc);
  }
}

// GDN recurrence emits [value_head, token, 128] F32. Out_proj owns
// [token, local_value_head, 128] BF16. Avoid CPU re-packing: round and
// transpose the independently validated result directly on each GPU.
__global__ void gdn_rank_to_out_bf16(const float* in,uint16_t* out,
                                      int rows,int value_heads) {
  const int n=value_heads*rows*H;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
    const int d=i%H;
    const int h=(i/H)%value_heads;
    const int t=i/(H*value_heads);
    out[i]=bf16rn(in[(static_cast<size_t>(h)*rows+t)*H+d]);
  }
}
// Original checkpoint BF16 [5120, local GDN width] column-shard GEMV.
// Independent 32-lane GPU wave per output neuron, FP32 sum per token.
__global__ void gdn_out_projection(const uint16_t* in,const uint16_t* w,
                                    float* partial,int rows,int width) {
  const int wave=warpSize,lane=threadIdx.x%wave;
  const int neuron=blockIdx.x*(blockDim.x/wave)+threadIdx.x/wave;
  const int token=blockIdx.y;
  float sum=0.0f;
  if(neuron<5120 && token<rows) {
    const auto* ww=w+static_cast<size_t>(neuron)*width;
    const auto* x=in+static_cast<size_t>(token)*width;
    for(int k=lane;k<width;k+=wave)sum+=fbf16(ww[k])*fbf16(x[k]);
  }
  for(int shift=wave/2;shift>0;shift/=2)
    sum+=__shfl_down(sum,shift);
  if(lane==0 && neuron<5120 && token<rows)
    partial[static_cast<size_t>(token)*5120+neuron]=sum;
}
struct Projection {
  int n=0;
  uint16_t* weights=nullptr,*output=nullptr;
  float* fp32=nullptr;
  std::vector<uint16_t> expected_bf16;
  std::vector<float> expected_f32;
  double max_error=0.0;
};
struct Rank {
  int id=-1,keys=0,values=0,keyoff=0,valoff=0,channels=0,rows=0;
  hipStream_t stream=nullptr;hipEvent_t start=nullptr,core_ready=nullptr,end=nullptr;
  uint16_t *normalized_input=nullptr,*conv=nullptr,*w=nullptr;
  float *q=nullptr,*k=nullptr,*state=nullptr,*out=nullptr,*coef=nullptr;
  uint16_t *out_weights=nullptr,*out_bf16=nullptr,*broadcast=nullptr;
  float *out_partial=nullptr,*pinned_partial=nullptr;
  std::vector<float> output_reference;
  std::vector<uint16_t> received;
  std::vector<double> core_latency;
  double out_max_abs=0.0;
  std::array<Projection,4> projections{};
  std::vector<float> actual,expected;
  std::vector<double> latency;
  double max_abs=0.0,max_rel=0.0,max_projection_error=0.0;
};
static std::string file_path(const std::string& directory,
                             const std::string& prefix,
                             const char* rank,const char* ext) {
  return directory+"/"+prefix+"-"+rank+"."+ext;
}
void setup(Rank& r,const std::string& conv_dir,const std::string& pack_dir,
           const std::string& reference_dir,const std::string& core_dir,
           const std::string& output_pack,const std::string& output_reference,
           const std::vector<float>& coefficients,int role) {
  const char* tag=role==0?"xtx0":role==1?"xtx1":"r9700";
  HC(hipSetDevice(r.id));HC(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&r.start));HC(hipEventCreate(&r.core_ready));HC(hipEventCreate(&r.end));
  const std::string conv_path=file_path(conv_dir,"conv1d",tag,"bf16");
  const auto weights=read_exact<uint16_t>(conv_path.c_str(),
                              static_cast<size_t>(r.channels)*4);
  const auto input=read_exact<uint16_t>((pack_dir+"/normed_input.bf16").c_str(),
                              static_cast<size_t>(r.rows)*5120);
  const auto expected_path=file_path(core_dir,"gdn-core",tag,"f32");
  const size_t outcount=static_cast<size_t>(r.values)*r.rows*H;
  r.expected=read_exact<float>(expected_path.c_str(),outcount);
  r.actual.resize(outcount);
  HC(hipMalloc(reinterpret_cast<void**>(&r.normalized_input),input.size()*2));
  HC(hipMalloc(reinterpret_cast<void**>(&r.w),weights.size()*2));
  HC(hipMemcpyAsync(r.normalized_input,input.data(),input.size()*2,
                    hipMemcpyHostToDevice,r.stream));
  HC(hipMemcpyAsync(r.w,weights.data(),weights.size()*2,
                    hipMemcpyHostToDevice,r.stream));
  constexpr const char* ops[]={"qkv","a","b","z"};
  for(int i=0;i<4;++i) {
    auto& p=r.projections[i];
    p.n=i==0?r.channels:(i==3?r.values*128:r.values);
    const std::string name=ops[i];
    const std::string wpath=file_path(i==0?conv_dir:pack_dir,
                                      i==0?"in_proj_qkv":name,tag,"bf16");
    const auto source=read_exact<uint16_t>(wpath.c_str(),
                                static_cast<size_t>(p.n)*5120);
    const auto fpath=file_path(reference_dir,name,tag,"f32");
    const auto bpath=file_path(reference_dir,name,tag,"bf16");
    p.expected_f32=read_exact<float>(fpath.c_str(),static_cast<size_t>(p.n)*r.rows);
    p.expected_bf16=read_exact<uint16_t>(bpath.c_str(),static_cast<size_t>(p.n)*r.rows);
    HC(hipMalloc(reinterpret_cast<void**>(&p.weights),source.size()*2));
    HC(hipMalloc(reinterpret_cast<void**>(&p.output),
                 static_cast<size_t>(p.n)*r.rows*2));
    HC(hipMalloc(reinterpret_cast<void**>(&p.fp32),
                 static_cast<size_t>(p.n)*r.rows*4));
    HC(hipMemcpyAsync(p.weights,source.data(),source.size()*2,
                      hipMemcpyHostToDevice,r.stream));
    // Host staging only during setup, never part of timed GPU kernels.
    HC(hipStreamSynchronize(r.stream));
  }
  const size_t convcount=static_cast<size_t>(r.rows)*r.channels;
  const size_t keycount=static_cast<size_t>(r.rows)*r.keys*H;
  HC(hipMalloc(reinterpret_cast<void**>(&r.conv),convcount*2));
  HC(hipMalloc(reinterpret_cast<void**>(&r.q),keycount*4));
  HC(hipMalloc(reinterpret_cast<void**>(&r.k),keycount*4));
  HC(hipMalloc(reinterpret_cast<void**>(&r.state),
               static_cast<size_t>(r.values)*H*H*4));
  HC(hipMalloc(reinterpret_cast<void**>(&r.out),outcount*4));
  HC(hipMalloc(reinterpret_cast<void**>(&r.coef),coefficients.size()*4));
  HC(hipMemcpyAsync(r.coef,coefficients.data(),coefficients.size()*4,
                    hipMemcpyHostToDevice,r.stream));
  const size_t nout=static_cast<size_t>(r.rows)*5120;
  const size_t local_width=static_cast<size_t>(r.values)*H;
  const std::string out_path=file_path(output_pack,"out",tag,"bf16");
  const auto out_weights=read_exact<uint16_t>(
      out_path.c_str(),5120*local_width);
  const auto partial_path=file_path(output_reference,"partial",tag,"f32");
  r.output_reference=read_exact<float>(partial_path.c_str(),nout);
  r.received.resize(nout);
  HC(hipMalloc(reinterpret_cast<void**>(&r.out_weights),
               out_weights.size()*sizeof(uint16_t)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.out_bf16),
               static_cast<size_t>(r.rows)*local_width*2));
  HC(hipMalloc(reinterpret_cast<void**>(&r.out_partial),nout*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.broadcast),nout*sizeof(uint16_t)));
  HC(hipHostMalloc(reinterpret_cast<void**>(&r.pinned_partial),
                   nout*sizeof(float),hipHostMallocPortable));
  HC(hipMemcpyAsync(r.out_weights,out_weights.data(),
                    out_weights.size()*sizeof(uint16_t),
                    hipMemcpyHostToDevice,r.stream));
  HC(hipStreamSynchronize(r.stream));
}
void launch(Rank& r) {
  HC(hipSetDevice(r.id));
  HC(hipEventRecord(r.start,r.stream));
  for(auto& p:r.projections) {
    const int waves=256/32;
    project_original_bf16<<<dim3((p.n+waves-1)/waves,r.rows),256,0,r.stream>>>(
      r.normalized_input,p.weights,p.fp32,p.output,p.n,r.rows);
  }
  HC(hipMemsetAsync(r.state,0,static_cast<size_t>(r.values)*H*H*4,r.stream));
  conv4<<<std::min(1024,(r.channels+255)/256),256,0,r.stream>>>(
    r.projections[0].output,r.w,r.conv,r.channels,r.rows);
  qk_norm<<<r.rows*2*r.keys,H,0,r.stream>>>(
    r.conv,r.q,r.k,r.channels,r.keys,r.rows);
  delta_qwen<<<r.values,H,0,r.stream>>>(
    r.conv,r.q,r.k,r.projections[1].output,r.projections[2].output,
    r.projections[3].output,r.coef,r.state,r.out,
    r.keys,r.values,r.valoff,r.channels,r.rows);
  HC(hipEventRecord(r.core_ready,r.stream));
  const int nlocal=r.values*H;
  gdn_rank_to_out_bf16<<<(r.rows*nlocal+255)/256,256,0,r.stream>>>(
      r.out,r.out_bf16,r.rows,r.values);
  gdn_out_projection<<<dim3((5120+7)/8,r.rows),256,0,r.stream>>>(
      r.out_bf16,r.out_weights,r.out_partial,r.rows,nlocal);
  HC(hipMemcpyAsync(r.pinned_partial,r.out_partial,
                    static_cast<size_t>(r.rows)*5120*sizeof(float),
                    hipMemcpyDeviceToHost,r.stream));
  HC(hipGetLastError());
  HC(hipEventRecord(r.end,r.stream));
}
static float fb16host(uint16_t bits) {
  const uint32_t b=static_cast<uint32_t>(bits)<<16;
  float f;std::memcpy(&f,&b,sizeof(float));return f;
}
void check(Rank& r) {
  HC(hipSetDevice(r.id));
  // Verify every actual BF16 projection before its effect on Qwen GDN.
  for(auto& p:r.projections) {
    std::vector<float> actual(p.expected_f32.size());
    std::vector<uint16_t> actual_bf16(p.expected_bf16.size());
    HC(hipMemcpy(actual.data(),p.fp32,actual.size()*4,hipMemcpyDeviceToHost));
    HC(hipMemcpy(actual_bf16.data(),p.output,actual_bf16.size()*2,
                 hipMemcpyDeviceToHost));
    for(size_t i=0;i<actual.size();++i) {
      const double delta=std::abs(static_cast<double>(actual[i])-p.expected_f32[i]);
      r.max_projection_error=std::max(r.max_projection_error,delta);
      if(!std::isfinite(actual[i]) ||
         delta>0.0005+0.002*std::abs(p.expected_f32[i]))
        throw std::runtime_error("original Qwen projection FP32 numeric oracle mismatch");
      // Exact RNE BF16 is expected on reference M1, but allow an
      // independent FP32 tree to land on the other side of a BF16 tie.
      const float bfloat=fb16host(actual_bf16[i]);
      const float want=fb16host(p.expected_bf16[i]);
      if(std::abs(bfloat-want)>0.02+0.02*std::abs(want))
        throw std::runtime_error("original Qwen projection BF16 rounding divergence");
    }
  }
  HC(hipMemcpy(r.actual.data(),r.out,r.actual.size()*4,hipMemcpyDeviceToHost));
  for(size_t i=0;i<r.actual.size();++i) {
    const double got=r.actual[i],wanted=r.expected[i];
    const double diff=std::abs(got-wanted);
    r.max_abs=std::max(r.max_abs,diff);
    r.max_rel=std::max(r.max_rel,diff/(1.+std::abs(wanted)));
    if(!std::isfinite(got) || diff > 0.002 + 0.02*std::abs(wanted))
      throw std::runtime_error("original BF16 Qwen projection-to-GDN GPU core oracle mismatch");
  }
  for(size_t i=0;i<r.output_reference.size();++i) {
    const double got=r.pinned_partial[i],expected=r.output_reference[i];
    const double diff=std::abs(got-expected);
    r.out_max_abs=std::max(r.out_max_abs,diff);
    if(!std::isfinite(got) || diff>0.0005+0.008*std::abs(expected))
      throw std::runtime_error("VTP20 original checkpoint out projection partial differs from CPU oracle");
  }
}
void destroy(Rank& r) {
  HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
  HC(hipFree(r.normalized_input));HC(hipFree(r.w));HC(hipFree(r.conv));
  HC(hipFree(r.q));HC(hipFree(r.k));HC(hipFree(r.state));
  HC(hipFree(r.out));HC(hipFree(r.coef));
  HC(hipFree(r.out_weights));HC(hipFree(r.out_bf16));
  HC(hipFree(r.out_partial));HC(hipFree(r.broadcast));
  HC(hipHostFree(r.pinned_partial));
  for(auto& p:r.projections) {
    HC(hipFree(p.weights));HC(hipFree(p.output));HC(hipFree(p.fp32));
  }
  HC(hipEventDestroy(r.start));HC(hipEventDestroy(r.core_ready));HC(hipEventDestroy(r.end));
  HC(hipStreamDestroy(r.stream));
}
static uint16_t round_host(float x) {
  uint32_t u=0;std::memcpy(&u,&x,sizeof(u));
  return static_cast<uint16_t>((u+0x7fffu+((u>>16)&1u))>>16);
}

// An optional, immutable host boundary after exact GPU BF16 numerical
// acceptance. Never overwrites existing output or silently creates
// model-wide state. File handoff is for numerical integration only.
static const char* vtp27_opt(int argc,char** argv,const char* key) {
  for(int i=1;i<argc;++i)if(std::strcmp(argv[i],key)==0) {
    if(i+1>=argc)throw std::runtime_error("VTP27 required option value");
    return argv[i+1];
  }
  return nullptr;
}
static void vtp27_export(const char* path,const uint16_t* data,size_t count) {
  if(path==nullptr)return;
  if(std::ifstream(path,std::ios::binary).good())
    throw std::runtime_error("VTP27 refuses existing BF16 output path");
  std::ofstream f(path,std::ios::binary|std::ios::out);
  if(!f)throw std::runtime_error("VTP27 cannot create handoff file");
  f.write(reinterpret_cast<const char*>(data),
          static_cast<std::streamsize>(count*sizeof(uint16_t)));
  if(!f)throw std::runtime_error("VTP27 BF16 handoff write incomplete");
}

int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const int reps=bounded(argc,argv,"--iterations",30);
    const auto coefficients=read_exact<float>(arg(argc,argv,"--coefficients"),224);
    const std::string conv_dir=arg(argc,argv,"--conv-dir");
    const std::string pack_dir=arg(argc,argv,"--pack-dir");
    const std::string reference_dir=arg(argc,argv,"--reference-dir");
    const std::string core_dir=arg(argc,argv,"--core-dir");
    const std::string out_pack=arg(argc,argv,"--output-pack-dir");
    const std::string out_reference=arg(argc,argv,"--output-reference-dir");
    const auto residual=read_exact<uint16_t>(arg(argc,argv,"--residual"),
                                            static_cast<size_t>(rows)*5120);
    const auto expected_full=read_exact<float>(
        (out_reference+"/ordered-three-rank.f32").c_str(),
        static_cast<size_t>(rows)*5120);
    const auto expected_residual=read_exact<uint16_t>(
        (out_reference+"/ordered-three-rank-residual.bf16").c_str(),
        static_cast<size_t>(rows)*5120);
    const auto ids=devices();
    std::array<Rank,3> ranks{};
    for(int i=0;i<3;++i) {
      Rank& r=ranks[i];
      r.id=ids[i];r.keys=KEYS[i];r.values=VALUES[i];
      r.keyoff=KEY_OFF[i];r.valoff=VALUE_OFF[i];r.rows=rows;
      r.channels=(2*r.keys+r.values)*H;
      setup(r,conv_dir,pack_dir,reference_dir,core_dir,
            out_pack,out_reference,coefficients,i);
    }
    const size_t count=static_cast<size_t>(rows)*5120;
    uint16_t* pinned_final=nullptr;
    HC(hipSetDevice(ids[0]));
    HC(hipHostMalloc(reinterpret_cast<void**>(&pinned_final),
                     count*sizeof(uint16_t),hipHostMallocPortable));
    std::vector<float> reduced(count);
    std::vector<double> wall;
    for(int it=-2;it<reps;++it) {
      const auto start=Clock::now();
      for(int i=0;i<3;++i)launch(ranks[i]);  // all 3 enqueue before ANY wait
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));HC(hipEventSynchronize(r.end));
      }
      for(size_t i=0;i<count;++i) {
        float v=ranks[0].pinned_partial[i];
        v+=ranks[1].pinned_partial[i];
        v+=ranks[2].pinned_partial[i];
        reduced[i]=v;
        pinned_final[i]=round_host(fb16host(round_host(v))+fb16host(residual[i]));
      }
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        HC(hipMemcpyAsync(r.broadcast,pinned_final,count*2,
                          hipMemcpyHostToDevice,r.stream));
      }
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
      }
      if(it<0)continue;
      wall.push_back(elapsed(start));
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        float core_ms=0,total_ms=0;
        HC(hipEventElapsedTime(&core_ms,r.start,r.core_ready));
        HC(hipEventElapsedTime(&total_ms,r.start,r.end));
        r.core_latency.push_back(static_cast<double>(core_ms)*1000.0);
        r.latency.push_back(static_cast<double>(total_ms)*1000.0);
      }
    }
    for(auto& r:ranks)check(r);
    size_t exact=0;
    double max_reduced_abs=0,max_final_abs=0;
    for(size_t i=0;i<count;++i) {
      const double diff=std::abs(reduced[i]-expected_full[i]);
      max_reduced_abs=std::max(max_reduced_abs,diff);
      if(!std::isfinite(reduced[i]) ||
          diff>0.001+0.008*std::abs(expected_full[i]))
        throw std::runtime_error("VTP20 full checkpoint-GDN output differs from independent CPU oracle");
      const double err=std::abs(fb16host(pinned_final[i])-fb16host(expected_residual[i]));
      max_final_abs=std::max(max_final_abs,err);
      if(pinned_final[i]==expected_residual[i])++exact;
      if(err>0.015+0.015*std::abs(fb16host(expected_residual[i])))
        throw std::runtime_error("VTP20 final BF16 residual exceeds independent CPU tolerance");
    }
    for(auto& r:ranks) {
      HC(hipSetDevice(r.id));
      HC(hipMemcpy(r.received.data(),r.broadcast,count*2,hipMemcpyDeviceToHost));
      if(!std::equal(r.received.begin(),r.received.end(),pinned_final))
        throw std::runtime_error("VTP20 three GPU broadcast did not agree bitwise");
    }
    printf("mode,rows,wall_us_p50,wall_us_p95,"
           "xtx0_total_us_p50,xtx1_total_us_p50,r9700_total_us_p50,"
           "xtx0_core_us_p50,xtx1_core_us_p50,r9700_core_us_p50,"
           "max_input_proj_abs,max_core_abs,max_output_partial_abs,"
           "max_reduced_abs,max_final_bf16_abs,final_exact,total,oracle\n");
    printf("original_bf16_GDN_full_block_three_GPU_host_residual,%d,"
           "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
           "%.8g,%.8g,%.8g,%.8g,%.8g,%zu,%zu,PASS\n",
           rows,percentile(wall,.5),percentile(wall,.95),
           percentile(ranks[0].latency,.5),percentile(ranks[1].latency,.5),
           percentile(ranks[2].latency,.5),
           percentile(ranks[0].core_latency,.5),percentile(ranks[1].core_latency,.5),
           percentile(ranks[2].core_latency,.5),
           std::max({ranks[0].max_projection_error,ranks[1].max_projection_error,
                     ranks[2].max_projection_error}),
           std::max({ranks[0].max_abs,ranks[1].max_abs,ranks[2].max_abs}),
           std::max({ranks[0].out_max_abs,ranks[1].out_max_abs,ranks[2].out_max_abs}),
           max_reduced_abs,max_final_abs,exact,count);
    printf("physical_ids=%d,%d,%d; true_original_BF16_checkpoint_GDN_projection_core_outproj_residual=YES;"
           " one_GPU_stream_per_physical_card=YES; GDN_core_output_transposed_on_gpu=YES;"
           " host_F32_3rank_ordered_reduce_H2D=YES; no_p2p=YES;"
           " synthetic_incoming_hidden=YES; full_model=NO; cache_continuation=NO; native_Radiance=NO\n",
           ids[0],ids[1],ids[2]);
    vtp27_export(vtp27_opt(argc,argv,"--export-bf16"),pinned_final,count);
    for(auto& r:ranks)destroy(r);
    HC(hipSetDevice(ids[0]));HC(hipHostFree(pinned_final));
    return 0;
  } catch(const std::exception& e) {
    fprintf(stderr,"VTP20 whole original BF16 GDN block FAILED: %s\n",e.what());
    return 1;
  }
}
