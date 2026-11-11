// VTP23: original Qwen3.8-27B BF16 *full-attention* GPU chain,
// at full model dimensions for layer 3 (a 24Q/4KV, 256-dim layer).
// 6/6/12 original query heads and 1/1/2 KV heads on XTX0/XTX1/R9700.
// Unlike VTP13's synthetic head-local proof, every projection,
// normalization, causal softmax, output gate and output column shard
// is loaded from the original checkpoint. One stream per GPU.
//
// This first step starts from a synthetic incoming hidden state
// normalized using the original checkpoint input RMSNorm. It is a
// GPU layer proof, NOT a native Radiance engine, full 64-layer
// generation, long context, chunk-prefill attention, or real tok/s.
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

#define HC(op) do {const auto e_=(op);if(e_!=hipSuccess){ \
  fprintf(stderr,"VTP23 HIP %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e_));exit(2); \
}}while(0)
constexpr int N=5120,HD=256,ROT=64;
constexpr std::array<int,3> Q{6,6,12};
constexpr std::array<int,3> KV{1,1,2};
constexpr std::array<const char*,3> ROLES{"xtx0","xtx1","r9700"};
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point now) {
  return std::chrono::duration<double,std::micro>(Clock::now()-now).count();
}
double percentile(std::vector<double> x,double p) {
  std::sort(x.begin(),x.end());
  return x[std::clamp(static_cast<size_t>(std::ceil(x.size()*p)),
                      size_t(1),x.size())-1];
}
const char* argument(int argc,char** argv,const char* name) {
  for(int i=1;i<argc;++i)if(std::strcmp(argv[i],name)==0) {
    if(i+1==argc)throw std::runtime_error(std::string("no value ")+name);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing ")+name);
}
int bounded(int argc,char** argv,const char* name,int high) {
  const char* x=argument(argc,argv,name);
  char* end=nullptr;long v=std::strtol(x,&end,10);
  if(!*x || !end || *end || v<1 || v>high)
    throw std::runtime_error("bounded argument invalid");
  return static_cast<int>(v);
}
template<class T>std::vector<T> read_exact(const std::string& file,size_t n) {
  std::ifstream f(file,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=n*sizeof(T))
    throw std::runtime_error("source control tensor extent wrong "+file);
  f.seekg(0);std::vector<T> values(n);
  f.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(n*sizeof(T)));
  if(!f)throw std::runtime_error("truncated original source "+file);
  return values;
}
std::string path(const std::string& dir,const std::string& leaf) {
  return dir+"/"+leaf;
}
std::array<int,3> physical_gpus() {
  int count=0;HC(hipGetDeviceCount(&count));
  std::vector<int> xtx,rdna4;
  for(int i=0;i<count;++i) {
    hipDeviceProp_t p{};HC(hipGetDeviceProperties(&p,i));
    if(std::strncmp(p.gcnArchName,"gfx1100",7)==0)xtx.push_back(i);
    else if(std::strncmp(p.gcnArchName,"gfx1201",7)==0)rdna4.push_back(i);
  }
  if(xtx.size()!=2 || rdna4.size()!=1)
    throw std::runtime_error("expected exactly two gfx1100 and one gfx1201 physical GPUs");
  return {xtx[0],xtx[1],rdna4[0]};
}
__device__ float bf16float(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits)<<16);
}
__device__ uint16_t bf16round(float value) {
  const uint32_t bits=__float_as_uint(value);
  return static_cast<uint16_t>((bits+0x7fffu+((bits>>16)&1u))>>16);
}
__device__ float sigmoid(float x) {
  return x>=0 ? 1.0f/(1.0f+expf(-x)) : expf(x)/(1.0f+expf(x));
}
// One wave owns one output neuron for one token, BF16 x BF16 -> F32.
__global__ void original_bf16_gemv(const uint16_t* input,const uint16_t* weight,
                                   uint16_t* output,int rows,int neurons,
                                   int input_width) {
  const int lane=threadIdx.x%warpSize;
  const int neuron=blockIdx.x*(blockDim.x/warpSize)+threadIdx.x/warpSize;
  const int row=blockIdx.y;
  float sum=0.0f;
  if(row<rows && neuron<neurons) {
    const auto* w=weight+static_cast<size_t>(neuron)*input_width;
    const auto* x=input+static_cast<size_t>(row)*input_width;
    for(int k=lane;k<input_width;k+=warpSize)
      sum+=bf16float(w[k])*bf16float(x[k]);
  }
  for(int delta=warpSize/2;delta>0;delta>>=1)
    sum+=__shfl_down(sum,delta);
  if(lane==0 && row<rows && neuron<neurons)
    output[static_cast<size_t>(row)*neurons+neuron]=bf16round(sum);
}
__global__ void zero_centered_rms(const uint16_t* in,uint16_t* out,
                                  const uint16_t* weight,int rows,int heads,
                                  int stride) {
  const int head=blockIdx.x,token=blockIdx.y,d=threadIdx.x;
  if(head>=heads || token>=rows || d>=HD)return;
  __shared__ float sq[HD];
  const size_t base=static_cast<size_t>(token)*heads*stride+head*stride;
  const float x=bf16float(in[base+d]);
  sq[d]=x*x;__syncthreads();
  for(int delta=HD/2;delta>0;delta>>=1) {
    if(d<delta)sq[d]+=sq[d+delta];
    __syncthreads();
  }
  const float scale=rsqrtf(sq[0]/HD+1.e-6f);
  out[base+d]=bf16round(x*scale*(1.0f+bf16float(weight[d])));
  if(stride==HD*2)
    out[base+HD+d]=in[base+HD+d];  // unnormalised query output gate
}
__global__ void text_rope64(const uint16_t* in,uint16_t* out,
                            int rows,int heads,int stride) {
  const int token=blockIdx.y,head=blockIdx.x,d=threadIdx.x;
  if(token>=rows || head>=heads || d>=HD)return;
  const size_t base=static_cast<size_t>(token)*heads*stride+head*stride;
  if(d<ROT) {
    const int k=d%32;
    const float a=bf16float(in[base+k]);
    const float b=bf16float(in[base+k+32]);
    const float freq=powf(1.e7f,-2.0f*k/ROT);
    const float angle=static_cast<float>(token)*freq;
    const float co=cosf(angle),si=sinf(angle);
    out[base+d]=bf16round(d<32?a*co-b*si:b*co+a*si);
  } else {
    out[base+d]=in[base+d];
  }
  if(stride==HD*2)
    out[base+HD+d]=in[base+HD+d];
}
// One block per query head and query token. Six query heads share
// each physical KV head. One R9700 kernel has 12 independent Q heads,
// and both of its original checkpoint KV groups are live concurrently.
//
// Online stable causal softmax. Every lane owns one output coordinate,
// and all lanes participate in the per-step 256-d dot reduction.
__global__ void original_causal_gated_gqa(
    const uint16_t* q,const uint16_t* k,const uint16_t* v,
    uint16_t* gated,int rows,int qheads,int kvheads) {
  const int head=blockIdx.x,token=blockIdx.y,d=threadIdx.x;
  const int kv=head/6;
  if(head>=qheads || kv>=kvheads || token>=rows || d>=HD)return;
  __shared__ float products[HD];
  __shared__ float factors[2];
  const size_t qb=static_cast<size_t>(token)*qheads*HD*2+head*HD*2;
  float weighted=0.0f,denom=0.0f;
  float mx=-1e30f;
  for(int prev=0;prev<=token;++prev) {
    const size_t kb=static_cast<size_t>(prev)*kvheads*HD+kv*HD;
    products[d]=bf16float(q[qb+d])*bf16float(k[kb+d]);
    __syncthreads();
    for(int step=HD/2;step>0;step>>=1) {
      if(d<step)products[d]+=products[d+step];
      __syncthreads();
    }
    if(d==0) {
      const float score=products[0]*(1.0f/16.0f);
      const float next=fmaxf(mx,score);
      factors[0]=expf(mx-next);
      factors[1]=expf(score-next);
      mx=next;
    }
    __syncthreads();
    denom=denom*factors[0]+factors[1];
    weighted=weighted*factors[0]+factors[1]*bf16float(v[kb+d]);
    __syncthreads();
  }
  const float value=bf16float(bf16round(weighted/denom));
  const float gate=bf16float(q[qb+HD+d]);
  gated[(static_cast<size_t>(token)*qheads+head)*HD+d]=
      bf16round(value*sigmoid(gate));
}
__global__ void original_output_col_gemv(
    const uint16_t* gated,const uint16_t* weight,
    float* partial,int rows,int width) {
  const int lane=threadIdx.x%warpSize;
  const int neuron=blockIdx.x*(blockDim.x/warpSize)+threadIdx.x/warpSize;
  const int token=blockIdx.y;
  float sum=0.0f;
  if(token<rows && neuron<N) {
    const auto* w=weight+static_cast<size_t>(neuron)*width;
    const auto* x=gated+static_cast<size_t>(token)*width;
    for(int j=lane;j<width;j+=warpSize)
      sum+=bf16float(w[j])*bf16float(x[j]);
  }
  for(int d=warpSize/2;d>0;d>>=1)sum+=__shfl_down(sum,d);
  if(lane==0 && token<rows && neuron<N)
    partial[static_cast<size_t>(token)*N+neuron]=sum;
}
static float hostbf(uint16_t value) {
  uint32_t bits=static_cast<uint32_t>(value)<<16;
  float f=0;std::memcpy(&f,&bits,sizeof(f));return f;
}
static uint16_t hostrn(float value) {
  uint32_t bits=0;std::memcpy(&bits,&value,sizeof(bits));
  return static_cast<uint16_t>((bits+0x7fffu+((bits>>16)&1u))>>16);
}
struct Tensor {
  uint16_t* gpu=nullptr;
  std::vector<uint16_t> reference;
  std::vector<uint16_t> observed;
  double max_difference=0.0;
  size_t exact=0;
};
struct Rank {
  int device=-1,rows=0,qheads=0,kvheads=0;
  hipStream_t stream=nullptr;
  hipEvent_t start=nullptr,done=nullptr;
  uint16_t *input=nullptr,*qweight=nullptr,*kweight=nullptr,*vweight=nullptr,*oweight=nullptr;
  uint16_t *qnorm=nullptr,*knorm=nullptr,*qnormed=nullptr,*knormed=nullptr;
  Tensor qraw,kraw,vraw,qrope,krope,gated;
  float* partial=nullptr,*pinned_partial=nullptr;
  uint16_t* broadcast=nullptr;
  std::vector<float> partial_ref;
  std::vector<uint16_t> broadcast_cpu;
  std::vector<double> timings;
  double max_partial_difference=0.0;
};
void h2d_alloc(uint16_t** ptr,const std::vector<uint16_t>& source,
               hipStream_t stream) {
  HC(hipMalloc(reinterpret_cast<void**>(ptr),source.size()*2));
  HC(hipMemcpyAsync(*ptr,source.data(),source.size()*2,hipMemcpyHostToDevice,stream));
}
void temp_alloc(Tensor& a,int rows,int width,const std::string& filename) {
  a.reference=read_exact<uint16_t>(filename,static_cast<size_t>(rows)*width);
  a.observed.resize(a.reference.size());
  HC(hipMalloc(reinterpret_cast<void**>(&a.gpu),a.reference.size()*2));
}
void setup(Rank& r,const std::string& pack,const std::string& cpu,int index) {
  const std::string role=ROLES[index];
  HC(hipSetDevice(r.device));
  HC(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&r.start));HC(hipEventCreate(&r.done));
  const int qw=r.qheads*HD*2,kw=r.kvheads*HD,ow=r.qheads*HD;
  const auto input=read_exact<uint16_t>(path(pack,"normed_input.bf16"),
                                         static_cast<size_t>(r.rows)*N);
  const auto q=read_exact<uint16_t>(path(pack,"q-"+role+".bf16"),
                                     static_cast<size_t>(qw)*N);
  const auto k=read_exact<uint16_t>(path(pack,"k-"+role+".bf16"),
                                     static_cast<size_t>(kw)*N);
  const auto v=read_exact<uint16_t>(path(pack,"v-"+role+".bf16"),
                                     static_cast<size_t>(kw)*N);
  const auto o=read_exact<uint16_t>(path(pack,"o-"+role+".bf16"),
                                     static_cast<size_t>(N)*ow);
  const auto qn=read_exact<uint16_t>(path(pack,"q_norm.bf16"),HD);
  const auto kn=read_exact<uint16_t>(path(pack,"k_norm.bf16"),HD);
  h2d_alloc(&r.input,input,r.stream);
  h2d_alloc(&r.qweight,q,r.stream);
  h2d_alloc(&r.kweight,k,r.stream);
  h2d_alloc(&r.vweight,v,r.stream);
  h2d_alloc(&r.oweight,o,r.stream);
  h2d_alloc(&r.qnorm,qn,r.stream);
  h2d_alloc(&r.knorm,kn,r.stream);
  temp_alloc(r.qraw,r.rows,qw,path(cpu,"q-gate-"+role+".bf16"));
  temp_alloc(r.kraw,r.rows,kw,path(cpu,"k-raw-"+role+".bf16"));
  temp_alloc(r.vraw,r.rows,kw,path(cpu,"v-"+role+".bf16"));
  temp_alloc(r.qrope,r.rows,qw,path(cpu,"q-rope-"+role+".bf16"));
  temp_alloc(r.krope,r.rows,kw,path(cpu,"k-rope-"+role+".bf16"));
  temp_alloc(r.gated,r.rows,ow,path(cpu,"attended-gated-"+role+".bf16"));
  HC(hipMalloc(reinterpret_cast<void**>(&r.qnormed),r.qraw.reference.size()*2));
  HC(hipMalloc(reinterpret_cast<void**>(&r.knormed),r.kraw.reference.size()*2));
  const size_t count=static_cast<size_t>(r.rows)*N;
  r.partial_ref=read_exact<float>(path(cpu,"partial-"+role+".f32"),count);
  HC(hipMalloc(reinterpret_cast<void**>(&r.partial),count*4));
  HC(hipMalloc(reinterpret_cast<void**>(&r.broadcast),count*2));
  HC(hipHostMalloc(reinterpret_cast<void**>(&r.pinned_partial),
                   count*4,hipHostMallocPortable));
  r.broadcast_cpu.resize(count);
  HC(hipStreamSynchronize(r.stream));
}
void launch(Rank& r) {
  HC(hipSetDevice(r.device));
  HC(hipEventRecord(r.start,r.stream));
  constexpr int WARPS=8;
  original_bf16_gemv<<<dim3((r.qheads*HD*2+WARPS-1)/WARPS,r.rows),
                       256,0,r.stream>>>(
    r.input,r.qweight,r.qraw.gpu,r.rows,r.qheads*HD*2,N);
  original_bf16_gemv<<<dim3((r.kvheads*HD+WARPS-1)/WARPS,r.rows),
                       256,0,r.stream>>>(
    r.input,r.kweight,r.kraw.gpu,r.rows,r.kvheads*HD,N);
  original_bf16_gemv<<<dim3((r.kvheads*HD+WARPS-1)/WARPS,r.rows),
                       256,0,r.stream>>>(
    r.input,r.vweight,r.vraw.gpu,r.rows,r.kvheads*HD,N);
  zero_centered_rms<<<dim3(r.qheads,r.rows),HD,0,r.stream>>>(
    r.qraw.gpu,r.qnormed,r.qnorm,r.rows,r.qheads,HD*2);
  zero_centered_rms<<<dim3(r.kvheads,r.rows),HD,0,r.stream>>>(
    r.kraw.gpu,r.knormed,r.knorm,r.rows,r.kvheads,HD);
  text_rope64<<<dim3(r.qheads,r.rows),HD,0,r.stream>>>(
    r.qnormed,r.qrope.gpu,r.rows,r.qheads,HD*2);
  text_rope64<<<dim3(r.kvheads,r.rows),HD,0,r.stream>>>(
    r.knormed,r.krope.gpu,r.rows,r.kvheads,HD);
  original_causal_gated_gqa<<<dim3(r.qheads,r.rows),HD,0,r.stream>>>(
    r.qrope.gpu,r.krope.gpu,r.vraw.gpu,r.gated.gpu,
    r.rows,r.qheads,r.kvheads);
  original_output_col_gemv<<<dim3((N+WARPS-1)/WARPS,r.rows),
                              256,0,r.stream>>>(
    r.gated.gpu,r.oweight,r.partial,r.rows,r.qheads*HD);
  HC(hipMemcpyAsync(r.pinned_partial,r.partial,
                    static_cast<size_t>(r.rows)*N*4,
                    hipMemcpyDeviceToHost,r.stream));
  HC(hipGetLastError());
  HC(hipEventRecord(r.done,r.stream));
}
void check(Tensor& tensor,int device,double abs_bound,double rel_bound) {
  HC(hipSetDevice(device));
  HC(hipMemcpy(tensor.observed.data(),tensor.gpu,
               tensor.observed.size()*2,hipMemcpyDeviceToHost));
  for(size_t i=0;i<tensor.observed.size();++i) {
    const double x=hostbf(tensor.observed[i]),y=hostbf(tensor.reference[i]);
    const double error=std::abs(x-y);
    tensor.max_difference=std::max(tensor.max_difference,error);
    if(tensor.observed[i]==tensor.reference[i])++tensor.exact;
    if(!std::isfinite(x) || error>abs_bound+rel_bound*std::abs(y))
      throw std::runtime_error("VTP23 original BF16 attention stage differs from CPU oracle");
  }
}
void check_rank(Rank& r) {
  check(r.qraw,r.device,.03,.03);
  check(r.kraw,r.device,.03,.03);
  check(r.vraw,r.device,.03,.03);
  check(r.qrope,r.device,.03,.04);
  check(r.krope,r.device,.03,.04);
  check(r.gated,r.device,.08,.06);
  for(size_t i=0;i<r.partial_ref.size();++i) {
    const double x=r.pinned_partial[i],y=r.partial_ref[i];
    const double error=std::abs(x-y);
    r.max_partial_difference=std::max(r.max_partial_difference,error);
    if(!std::isfinite(x) || error>.3+.06*std::abs(y))
      throw std::runtime_error("VTP23 original BF16 output shard differs from CPU");
  }
}
void destroy(Rank& r) {
  HC(hipSetDevice(r.device));HC(hipStreamSynchronize(r.stream));
  for(auto* ptr:{r.input,r.qweight,r.kweight,r.vweight,r.oweight,
                 r.qnorm,r.knorm,r.qnormed,r.knormed,
                 r.qraw.gpu,r.kraw.gpu,r.vraw.gpu,r.qrope.gpu,
                 r.krope.gpu,r.gated.gpu})
    HC(hipFree(ptr));
  HC(hipFree(r.partial));HC(hipFree(r.broadcast));
  HC(hipHostFree(r.pinned_partial));
  HC(hipEventDestroy(r.start));HC(hipEventDestroy(r.done));
  HC(hipStreamDestroy(r.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const int iterations=bounded(argc,argv,"--iterations",30);
    const std::string pack=argument(argc,argv,"--pack-dir");
    const std::string cpu=argument(argc,argv,"--reference-dir");
    const auto residual=read_exact<uint16_t>(
      path(pack,"hidden_input.bf16"),static_cast<size_t>(rows)*N);
    const auto full_reference=read_exact<float>(
      path(cpu,"original-attention-reduced.f32"),static_cast<size_t>(rows)*N);
    const auto final_reference=read_exact<uint16_t>(
      path(cpu,"original-attention-post-residual.bf16"),static_cast<size_t>(rows)*N);
    const auto devices=physical_gpus();
    std::array<Rank,3> rank{};
    for(int i=0;i<3;++i) {
      rank[i].device=devices[i];rank[i].rows=rows;
      rank[i].qheads=Q[i];rank[i].kvheads=KV[i];
      setup(rank[i],pack,cpu,i);
    }
    const size_t n=static_cast<size_t>(rows)*N;
    uint16_t* pinned_final=nullptr;
    HC(hipSetDevice(devices[0]));
    HC(hipHostMalloc(reinterpret_cast<void**>(&pinned_final),
                     n*2,hipHostMallocPortable));
    std::vector<float> reduced(n);
    std::vector<double> wall;
    for(int rep=-2;rep<iterations;++rep) {
      const auto t=Clock::now();
      for(auto& r:rank)launch(r); // all 3 run before any completion barrier
      for(auto& r:rank) {
        HC(hipSetDevice(r.device));HC(hipEventSynchronize(r.done));
      }
      for(size_t i=0;i<n;++i) {
        float sum=rank[0].pinned_partial[i];
        sum+=rank[1].pinned_partial[i];
        sum+=rank[2].pinned_partial[i];
        reduced[i]=sum;
        pinned_final[i]=hostrn(hostbf(hostrn(sum))+hostbf(residual[i]));
      }
      for(auto& r:rank) {
        HC(hipSetDevice(r.device));
        HC(hipMemcpyAsync(r.broadcast,pinned_final,n*2,
                          hipMemcpyHostToDevice,r.stream));
      }
      for(auto& r:rank) {
        HC(hipSetDevice(r.device));HC(hipStreamSynchronize(r.stream));
      }
      if(rep<0)continue;
      wall.push_back(elapsed(t));
      for(auto& r:rank) {
        HC(hipSetDevice(r.device));
        float ms=0;HC(hipEventElapsedTime(&ms,r.start,r.done));
        r.timings.push_back(static_cast<double>(ms)*1000.0);
      }
    }
    for(auto& r:rank)check_rank(r);
    size_t exact_final=0;
    double max_reduced=0,max_residual=0,max_partial=0,max_stage=0;
    for(auto& r:rank) {
      max_partial=std::max(max_partial,r.max_partial_difference);
      max_stage=std::max({max_stage,r.qraw.max_difference,
                          r.kraw.max_difference,r.vraw.max_difference,
                          r.qrope.max_difference,r.krope.max_difference,
                          r.gated.max_difference});
      HC(hipSetDevice(r.device));
      HC(hipMemcpy(r.broadcast_cpu.data(),r.broadcast,n*2,hipMemcpyDeviceToHost));
      if(!std::equal(r.broadcast_cpu.begin(),r.broadcast_cpu.end(),pinned_final))
        throw std::runtime_error("VTP23 original full attention broadcast mismatch");
    }
    for(size_t i=0;i<n;++i) {
      const double a=reduced[i],b=full_reference[i];
      const double err=std::abs(a-b);
      max_reduced=std::max(max_reduced,err);
      if(!std::isfinite(a) || err>.3+.06*std::abs(b))
        throw std::runtime_error("VTP23 reduced full original attention CPU oracle mismatch");
      if(pinned_final[i]==final_reference[i])++exact_final;
      const double fb=std::abs(hostbf(pinned_final[i])-hostbf(final_reference[i]));
      max_residual=std::max(max_residual,fb);
      if(fb>.4+.06*std::abs(hostbf(final_reference[i])))
        throw std::runtime_error("VTP23 full attention BF16 residual exceeds CPU oracle");
    }
    printf("mode,rows,wall_us_p50,wall_us_p95,xtx0_gpu_us_p50,xtx1_gpu_us_p50,"
           "r9700_gpu_us_p50,max_stage_abs,max_partial_abs,max_reduced_abs,"
           "max_final_bf16_abs,final_exact,total,oracle\n");
    printf("original_BF16_full_attention_3physical_gpu,%d,"
           "%.2f,%.2f,%.2f,%.2f,%.2f,%.8g,%.8g,%.8g,%.8g,%zu,%zu,PASS\n",
           rows,percentile(wall,.5),percentile(wall,.95),
           percentile(rank[0].timings,.5),percentile(rank[1].timings,.5),
           percentile(rank[2].timings,.5),
           max_stage,max_partial,max_reduced,max_residual,exact_final,n);
    printf("physical_devices=%d,%d,%d; original_layer3_BF16=YES;"
           " q_gate_interleaved=YES; Q_per_rank=6/6/12; KV_per_rank=1/1/2;"
           " partial_rope_64=YES; causal_softmax=YES; sigmoid_gate=YES;"
           " host_staged_ordered_reduce=YES; no_p2p=YES;"
           " synthetic_incoming_hidden=YES; full_model=NO\n",
           devices[0],devices[1],devices[2]);
    for(auto& r:rank)destroy(r);
    HC(hipSetDevice(devices[0]));HC(hipHostFree(pinned_final));
    return 0;
  } catch(const std::exception& e) {
    fprintf(stderr,"VTP23 original BF16 GPU full-attention FAILED: %s\n",e.what());
    return 1;
  }
}
