// VTP18: row-fused output BF16 GEMV with batch-aware weight reuse, compared to VTP17 pinned:
// persistent HIP portable pinned host packets and asynchronous ordered
// D2H (enqueued immediately behind each GPU GEMV), CPU F32 reduce,
// H2D broadcast; three independent physical contributions; no P2P.
// All input projection weights and reference outputs come from
// checkpoint-backed SHA256-validated immutable VTP16 packets.
//
// This tests pinned asynchronous output GEMVs and the no-P2P collector; it does
// NOT yet attach to the previous VTP15 on-device GDN core, nor does
// it execute native Radiance GEMM or full model inference.
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

#define HC(op) do{const hipError_t e_=(op);if(e_!=hipSuccess){ \
 fprintf(stderr,"VTP16 HIP failure %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e_));exit(2); \
}}while(0)
constexpr int N=5120;
constexpr std::array<int,3> WIDTHS{1536,1536,3072};
constexpr std::array<const char*,3> ROLES{"xtx0","xtx1","r9700"};
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point begin){
  return std::chrono::duration<double,std::micro>(Clock::now()-begin).count();
}
double percentile(std::vector<double> values,double fraction) {
  std::sort(values.begin(),values.end());
  const size_t rank=std::clamp(static_cast<size_t>(std::ceil(fraction*values.size())),
                               size_t(1),values.size());
  return values[rank-1];
}
const char* arg(int argc,char** argv,const char* key) {
  for(int i=1;i<argc;++i) if(std::strcmp(argv[i],key)==0) {
    if(i+1==argc) throw std::runtime_error(std::string("missing ")+key);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing ")+key);
}
int bounded(int argc,char** argv,const char* key,int limit) {
  const char* s=arg(argc,argv,key);
  char* end=nullptr;const long value=std::strtol(s,&end,10);
  if(!*s || !end || *end || value<1 || value>limit)
    throw std::runtime_error(std::string("invalid bounded ")+key);
  return static_cast<int>(value);
}
template<class T>std::vector<T> read_exact(const std::string& path,size_t n) {
  std::ifstream input(path,std::ios::binary|std::ios::ate);
  if(!input || input.tellg()<0 || static_cast<size_t>(input.tellg())!=n*sizeof(T))
    throw std::runtime_error("invalid original BF16 input extent "+path);
  input.seekg(0);
  std::vector<T> result(n);
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(n*sizeof(T)));
  if(!input)throw std::runtime_error("truncated original BF16 reference "+path);
  return result;
}
std::array<int,3> devices() {
  int count=0;HC(hipGetDeviceCount(&count));
  std::vector<int> xtx,r9700;
  for(int i=0;i<count;++i) {
    hipDeviceProp_t prop{};HC(hipGetDeviceProperties(&prop,i));
    if(std::strncmp(prop.gcnArchName,"gfx1100",7)==0 ||
       std::strncmp(prop.gcnArchName,"gfx1201",7)==0) {
      if(prop.warpSize!=32)
        throw std::runtime_error("VTP16 requires 32-lane waves on selected GPUs");
      if(std::strncmp(prop.gcnArchName,"gfx1100",7)==0)xtx.push_back(i);
      else r9700.push_back(i);
    }
  }
  if(xtx.size()!=2 || r9700.size()!=1)
    throw std::runtime_error("need 2 gfx1100 and 1 gfx1201 GPUs");
  return {xtx[0],xtx[1],r9700[0]};
}
__device__ float bf16value(uint16_t x) {
  return __uint_as_float(static_cast<uint32_t>(x)<<16);
}
__global__ void output_col_gemv(const uint16_t* x,const uint16_t* weight,
                                float* partial,int m,int width) {
  const int wave=warpSize;
  const int lane=threadIdx.x%wave;
  const int neuron=blockIdx.x*(blockDim.x/wave)+threadIdx.x/wave;
  const int row=blockIdx.y;
  float sum=0.f;
  if(neuron<N && row<m) {
    const uint16_t* w=weight+static_cast<size_t>(neuron)*width;
    const uint16_t* a=x+static_cast<size_t>(row)*width;
    for(int k=lane;k<width;k+=wave)
      sum+=bf16value(w[k])*bf16value(a[k]);
  }
  for(int delta=wave/2;delta>0;delta/=2)
    sum+=__shfl_down(sum,delta);
  if(lane==0 && neuron<N && row<m)
    partial[static_cast<size_t>(row)*N+neuron]=sum;
}
// Specialised M=2/4/8 row-fused GEMV. Each physical GPU wave owns
// one output neuron and reads its original BF16 weight once for all M
// independent batch rows; the original VTP17 code reloads the whole
// weight matrix for each row. Every row retains its own accumulation
// order and BF16 input/output contract.
template<int M>
__global__ void output_col_gemv_reuse(const uint16_t* x,
                                      const uint16_t* weight,
                                      float* partial,int width) {
  const int wave=warpSize;
  const int lane=threadIdx.x%wave;
  const int neuron=blockIdx.x*(blockDim.x/wave)+threadIdx.x/wave;
  float acc[M]{};
  if(neuron<N) {
    const uint16_t* w=weight+static_cast<size_t>(neuron)*width;
    for(int k=lane;k<width;k+=wave) {
      const float value=bf16value(w[k]);
#pragma unroll
      for(int row=0;row<M;++row)
        acc[row]+=value*bf16value(x[static_cast<size_t>(row)*width+k]);
    }
  }
#pragma unroll
  for(int row=0;row<M;++row) {
    float sum=acc[row];
    for(int delta=wave/2;delta>0;delta/=2)
      sum+=__shfl_down(sum,delta);
    if(lane==0 && neuron<N)
      partial[static_cast<size_t>(row)*N+neuron]=sum;
  }
}
static inline float f32(uint16_t x) {
  const uint32_t u=static_cast<uint32_t>(x)<<16;
  float f;std::memcpy(&f,&u,sizeof(f));return f;
}
static inline uint16_t roundbf16(float f) {
  uint32_t u;std::memcpy(&u,&f,sizeof(u));
  return static_cast<uint16_t>((u+0x7fffu+((u>>16)&1u))>>16);
}
struct Rank {
  int id=-1,width=0,rows=0;
  hipStream_t stream=nullptr;
  hipEvent_t begin=nullptr,end=nullptr;
  uint16_t* weights=nullptr,*inputs=nullptr,*broadcast=nullptr;
  float* partial=nullptr;
  float* pinned_partial=nullptr;
  std::vector<float> expected_partial;
  std::vector<uint16_t> cpu_broadcast;
  std::vector<double> latency;
  double max_partial_error=0.0;
};
std::string path(const std::string& dir,const std::string& name) {
  return dir+"/"+name;
}
void setup(Rank& r,const std::string& packet,const std::string& ref,int i) {
  HC(hipSetDevice(r.id));
  HC(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&r.begin));
  HC(hipEventCreate(&r.end));
  const std::string role=ROLES[i];
  const auto weights=read_exact<uint16_t>(path(packet,"out-"+role+".bf16"),
                                         static_cast<size_t>(r.width)*N);
  const auto input=read_exact<uint16_t>(path(packet,"gdn-"+role+".bf16"),
                                       static_cast<size_t>(r.width)*r.rows);
  r.expected_partial=read_exact<float>(path(ref,"partial-"+role+".f32"),
                                       static_cast<size_t>(r.rows)*N);
  HC(hipHostMalloc(reinterpret_cast<void**>(&r.pinned_partial),
                   r.expected_partial.size()*sizeof(float),
                   hipHostMallocPortable));
  r.cpu_broadcast.resize(static_cast<size_t>(r.rows)*N);
  HC(hipMalloc(reinterpret_cast<void**>(&r.weights),weights.size()*2));
  HC(hipMalloc(reinterpret_cast<void**>(&r.inputs),input.size()*2));
  HC(hipMalloc(reinterpret_cast<void**>(&r.partial),r.expected_partial.size()*4));
  HC(hipMalloc(reinterpret_cast<void**>(&r.broadcast),r.cpu_broadcast.size()*2));
  HC(hipMemcpyAsync(r.weights,weights.data(),weights.size()*2,
                    hipMemcpyHostToDevice,r.stream));
  HC(hipMemcpyAsync(r.inputs,input.data(),input.size()*2,
                    hipMemcpyHostToDevice,r.stream));
  HC(hipStreamSynchronize(r.stream));
}
void launch(Rank& r) {
  HC(hipSetDevice(r.id));
  HC(hipEventRecord(r.begin,r.stream));
  if(r.rows==2)
    output_col_gemv_reuse<2><<<dim3((N+7)/8),256,0,r.stream>>>(
        r.inputs,r.weights,r.partial,r.width);
  else if(r.rows==4)
    output_col_gemv_reuse<4><<<dim3((N+7)/8),256,0,r.stream>>>(
        r.inputs,r.weights,r.partial,r.width);
  else if(r.rows==8)
    output_col_gemv_reuse<8><<<dim3((N+7)/8),256,0,r.stream>>>(
        r.inputs,r.weights,r.partial,r.width);
  else
    output_col_gemv<<<dim3((N+7)/8,r.rows),256,0,r.stream>>>(
        r.inputs,r.weights,r.partial,r.rows,r.width);
  HC(hipGetLastError());
  HC(hipEventRecord(r.end,r.stream));
}
void destroy(Rank& r) {
  HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
  HC(hipFree(r.weights));HC(hipFree(r.inputs));
  HC(hipFree(r.partial));HC(hipFree(r.broadcast));
  HC(hipHostFree(r.pinned_partial));
  HC(hipEventDestroy(r.begin));HC(hipEventDestroy(r.end));
  HC(hipStreamDestroy(r.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const int reps=bounded(argc,argv,"--iterations",30);
    const std::string packet=arg(argc,argv,"--pack-dir");
    const std::string reference=arg(argc,argv,"--reference-dir");
    const auto resid=read_exact<uint16_t>(arg(argc,argv,"--residual"),
                                         static_cast<size_t>(rows)*N);
    const auto expected=read_exact<float>(path(reference,"ordered-three-rank.f32"),
                                          static_cast<size_t>(rows)*N);
    const auto expected_final=read_exact<uint16_t>(
      path(reference,"ordered-three-rank-residual.bf16"),
      static_cast<size_t>(rows)*N);
    const auto ids=devices();
    std::array<Rank,3> ranks{};
    for(int r=0;r<3;++r) {
      ranks[r].id=ids[r];ranks[r].width=WIDTHS[r];ranks[r].rows=rows;
      setup(ranks[r],packet,reference,r);
    }
    std::vector<float> host_total(static_cast<size_t>(rows)*N);
    uint16_t* host_final=nullptr;
    HC(hipSetDevice(ids[0]));
    HC(hipHostMalloc(reinterpret_cast<void**>(&host_final),
                     host_total.size()*sizeof(uint16_t),hipHostMallocPortable));
    std::vector<double> latency, d2h_stage, host_reduce_stage, h2d_stage;
    double max_reduce_error=0.0,max_final_error=0.0,max_partial_error=0.0;
    size_t exact_bf16=0;
    for(int iter=-2;iter<reps;++iter) {
      const auto wall=Clock::now();
      for(auto& r:ranks)launch(r);
      // D2H transfers are enqueued immediately after each GEMV in
      // the same stream, so they overlap across the three GPUs.
      // Portable pinned buffers are stable throughout the experiment.
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        HC(hipMemcpyAsync(r.pinned_partial,r.partial,
                          r.expected_partial.size()*sizeof(float),
                          hipMemcpyDeviceToHost,r.stream));
      }
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
      }
      const auto after_d2h=Clock::now();
      // Exactly one physical host contribution from each GPU, in the
      // documented deterministic order. No P2P assumptions.
      for(size_t j=0;j<host_total.size();++j) {
        float v=ranks[0].pinned_partial[j];
        v+=ranks[1].pinned_partial[j];
        v+=ranks[2].pinned_partial[j];
        host_total[j]=v;
        host_final[j]=roundbf16(f32(roundbf16(v))+f32(resid[j]));
      }
      const auto after_reduce=Clock::now();
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        HC(hipMemcpyAsync(r.broadcast,host_final,host_total.size()*2,
                          hipMemcpyHostToDevice,r.stream));
      }
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
      }
      if(iter<0)continue;
      latency.push_back(elapsed(wall));
      d2h_stage.push_back(std::chrono::duration<double,std::micro>(after_d2h-wall).count());
      host_reduce_stage.push_back(std::chrono::duration<double,std::micro>(after_reduce-after_d2h).count());
      h2d_stage.push_back(std::chrono::duration<double,std::micro>(Clock::now()-after_reduce).count());
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        float time_ms=0;
        HC(hipEventElapsedTime(&time_ms,r.begin,r.end));
        r.latency.push_back(static_cast<double>(time_ms)*1000.0);
      }
    }
    for(auto& r:ranks) {
      for(size_t j=0;j<r.expected_partial.size();++j) {
        const double x=r.pinned_partial[j],y=r.expected_partial[j];
        const double e=std::abs(x-y);
        max_partial_error=std::max(max_partial_error,e);
        if(!std::isfinite(x) || e>0.0005+0.002*std::abs(y))
          throw std::runtime_error("original BF16 physical output shard numeric oracle failed");
      }
      HC(hipSetDevice(r.id));
      HC(hipMemcpy(r.cpu_broadcast.data(),r.broadcast,
                   r.cpu_broadcast.size()*2,hipMemcpyDeviceToHost));
      if(!std::equal(r.cpu_broadcast.begin(),r.cpu_broadcast.end(),host_final))
        throw std::runtime_error("rank did not receive identical BF16 host broadcast");
    }
    for(size_t j=0;j<host_total.size();++j) {
      const double f32error=std::abs(static_cast<double>(host_total[j])-expected[j]);
      max_reduce_error=std::max(max_reduce_error,f32error);
      if(!std::isfinite(host_total[j]) || f32error>0.001+0.002*std::abs(expected[j]))
        throw std::runtime_error("no-P2P ordered host reduction differs from independent CPU oracle");
      const double ferr=std::abs(f32(host_final[j])-f32(expected_final[j]));
      max_final_error=std::max(max_final_error,ferr);
      if(host_final[j]==expected_final[j])++exact_bf16;
      if(ferr>0.01+0.005*std::abs(f32(expected_final[j])))
        throw std::runtime_error("original BF16 residual result differs numerically from CPU oracle");
    }
    std::printf("mode,rows,wall_us_p50,wall_us_p95,xtx0_kernel_us_p50,"
                "xtx1_kernel_us_p50,r9700_kernel_us_p50,"
                "d2h_stage_us_p50,host_reduce_us_p50,h2d_stage_us_p50,"
                "max_partial_abs,max_reduce_abs,max_final_bf16_abs,"
                "final_bf16_exact,total,oracle\n");
    std::printf("original_BF16_out_proj_row_reuse_pinned_async_three_GPU,%d,"
                "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.8g,%.8g,%.8g,%zu,%zu,PASS\n",
                rows,percentile(latency,.5),percentile(latency,.95),
                percentile(ranks[0].latency,.5),
                percentile(ranks[1].latency,.5),
                percentile(ranks[2].latency,.5),
                percentile(d2h_stage,.5),percentile(host_reduce_stage,.5),
                percentile(h2d_stage,.5),
                max_partial_error,max_reduce_error,max_final_error,
                exact_bf16,host_total.size());
    std::printf("physical_ids=%d,%d,%d; original_out_proj_BF16=YES;"
                " input=CPU_original_weight_GDN_core; host_D2H_F32_reduce_H2D=YES;"
                " no_p2p=YES; full_on_GPU_GDN_block=NO; engine=NO\n",
                ids[0],ids[1],ids[2]);
    for(auto& r:ranks)destroy(r);
    HC(hipHostFree(host_final));
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"VTP18 row-reuse output-projection GPU oracle FAILED: %s\n",e.what());
    return 1;
  }
}
