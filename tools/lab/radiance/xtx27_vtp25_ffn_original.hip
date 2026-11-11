// VTP25 original BF16 Qwen3.8-27B layer-0 FFN on physical TP3.
// Actual original checkpoint gate/up and down projection shards
// (5120/4864/7424) on two gfx1100 XTX and one gfx1201 R9700.
// Original GDN-after-attention residual -> checkpoint original
// post_attention_layernorm -> FFN gate/up BF16 GEMV -> BF16 SwiGLU
// -> original BF16 down projection F32 physical partials
// -> pinned host F32 reduce -> BF16 residual -> 3 H2D broadcasts.
// GPU stage/outputs independently compared to original CPU oracle.
// NOT the full 64-layer model, native Radiance, or real decoding.
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
#define HC(x) do {const auto code_=(x);if(code_!=hipSuccess){ \
  fprintf(stderr,"VTP25 HIP %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(code_));exit(2); \
}}while(0)
constexpr int H=5120;
constexpr std::array<int,3> WIDTHS{5120,4864,7424};
constexpr std::array<const char*,3> ROLES{"xtx0","xtx1","r9700"};
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
double percentile(std::vector<double> x,double p) {
  std::sort(x.begin(),x.end());
  return x[std::clamp(static_cast<size_t>(std::ceil(x.size()*p)),
                      size_t(1),x.size())-1];
}
const char* arg(int argc,char** argv,const char* name) {
  for(int i=1;i<argc;++i)if(std::strcmp(argv[i],name)==0) {
    if(i+1==argc)throw std::runtime_error(std::string("missing ")+name);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing "+std::string(name)));
}
int bounded(int argc,char** argv,const char* name,int maxval) {
  const char* x=arg(argc,argv,name);
  char* end=nullptr;long value=std::strtol(x,&end,10);
  if(!*x || !end || *end || value<1 || value>maxval)
    throw std::runtime_error(std::string("invalid bounded ")+name);
  return static_cast<int>(value);
}
template<class T>std::vector<T> read_exact(const std::string& path,size_t n) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=n*sizeof(T))
    throw std::runtime_error("original BF16 source extent invalid: "+path);
  f.seekg(0);std::vector<T> data(n);
  f.read(reinterpret_cast<char*>(data.data()),
         static_cast<std::streamsize>(n*sizeof(T)));
  if(!f)throw std::runtime_error("truncated source: "+path);
  return data;
}
std::string file(const std::string& dir,const std::string& stem) {
  return dir+"/"+stem;
}
std::array<int,3> devices() {
  int count=0;HC(hipGetDeviceCount(&count));
  std::vector<int> xtx,r9700;
  for(int i=0;i<count;++i) {
    hipDeviceProp_t p{};HC(hipGetDeviceProperties(&p,i));
    if(std::strncmp(p.gcnArchName,"gfx1100",7)==0)xtx.push_back(i);
    if(std::strncmp(p.gcnArchName,"gfx1201",7)==0)r9700.push_back(i);
  }
  if(xtx.size()!=2 || r9700.size()!=1)
    throw std::runtime_error("VTP25 requires 2 gfx1100 and one gfx1201 GPU");
  return {xtx[0],xtx[1],r9700[0]};
}
__device__ float fb16(uint16_t val) {
  return __uint_as_float(static_cast<uint32_t>(val)<<16);
}
__device__ uint16_t bfrn(float x) {
  const uint32_t v=__float_as_uint(x);
  return static_cast<uint16_t>((v+0x7fffu+((v>>16)&1u))>>16);
}
// 8 wave32 groups per 256-thread block; each wave reads one
// checkpoint weight output row BF16 and its BF16 input token.
// F32 lane reduction, once BF16 rounding at GEMV output.
__global__ void original_ffn_bf16_projection(
    const uint16_t* w,const uint16_t* x,uint16_t* y,
    int rows,int neurons,int width) {
  const int lane=threadIdx.x%warpSize;
  const int neuron=blockIdx.x*(blockDim.x/warpSize)+threadIdx.x/warpSize;
  const int row=blockIdx.y;
  float total=0.0f;
  if(neuron<neurons && row<rows) {
    const auto* weight=w+static_cast<size_t>(neuron)*width;
    const auto* input=x+static_cast<size_t>(row)*width;
    for(int k=lane;k<width;k+=warpSize)
      total+=fb16(weight[k])*fb16(input[k]);
  }
  for(int d=warpSize/2;d>0;d>>=1)
    total+=__shfl_down(total,d);
  if(lane==0 && neuron<neurons && row<rows)
    y[static_cast<size_t>(row)*neurons+neuron]=bfrn(total);
}
__global__ void original_ffn_swiglu(
    const uint16_t* gate,const uint16_t* up,uint16_t* mid,int total) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;
      i<total;i+=blockDim.x*gridDim.x) {
    const float g=fb16(gate[i]),u=fb16(up[i]);
    const float act=g>=0?g/(1.0f+expf(-g)) :
        g*expf(g)/(1.0f+expf(g));
    mid[i]=bfrn(act*u);
  }
}
__global__ void original_ffn_bf16_down(
    const uint16_t* w,const uint16_t* mid,float* out,int rows,int width) {
  const int lane=threadIdx.x%warpSize;
  const int neuron=blockIdx.x*(blockDim.x/warpSize)+threadIdx.x/warpSize;
  const int row=blockIdx.y;
  float sum=0;
  if(row<rows && neuron<H) {
    const auto* weights=w+static_cast<size_t>(neuron)*width;
    const auto* x=mid+static_cast<size_t>(row)*width;
    for(int k=lane;k<width;k+=warpSize)
      sum+=fb16(weights[k])*fb16(x[k]);
  }
  for(int d=warpSize/2;d>0;d>>=1)sum+=__shfl_down(sum,d);
  if(lane==0 && neuron<H && row<rows)
    out[static_cast<size_t>(row)*H+neuron]=sum;
}
float hostbf(uint16_t value) {
  uint32_t raw=static_cast<uint32_t>(value)<<16;
  float out=0;std::memcpy(&out,&raw,sizeof(out));return out;
}
uint16_t hostround(float x) {
  uint32_t bits=0;std::memcpy(&bits,&x,sizeof(bits));
  return static_cast<uint16_t>((bits+0x7fffu+((bits>>16)&1u))>>16);
}
struct Check {
  uint16_t* gpu=nullptr;
  std::vector<uint16_t> reference;
  std::vector<uint16_t> returned;
  double max_abs=0;
  size_t exact=0;
};
struct Rank {
  int id=-1,rows=0,width=0;
  hipStream_t stream=nullptr;
  hipEvent_t begin=nullptr,end=nullptr;
  uint16_t *input=nullptr,*gatew=nullptr,*upw=nullptr,*downw=nullptr,*broadcast=nullptr;
  Check gate,up,mid;
  float* partial=nullptr,*pinned_partial=nullptr;
  std::vector<float> partial_ref;
  std::vector<uint16_t> broadcast_cpu;
  std::vector<double> gpu_times;
  double max_partial_abs=0;
};
void upload(uint16_t** out,const std::vector<uint16_t>& data,hipStream_t stream) {
  HC(hipMalloc(reinterpret_cast<void**>(out),data.size()*2));
  HC(hipMemcpyAsync(*out,data.data(),data.size()*2,hipMemcpyHostToDevice,stream));
}
void prep(Check& target,const std::string& filepath,size_t n) {
  target.reference=read_exact<uint16_t>(filepath,n);
  target.returned.resize(n);
  HC(hipMalloc(reinterpret_cast<void**>(&target.gpu),n*2));
}
void setup(Rank& rank,const std::string& packet,const std::string& cpu,int id) {
  const std::string tag=ROLES[id];
  const size_t width=static_cast<size_t>(rank.width);
  HC(hipSetDevice(rank.id));
  HC(hipStreamCreateWithFlags(&rank.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&rank.begin));HC(hipEventCreate(&rank.end));
  const auto input=read_exact<uint16_t>(
      file(packet,"normed.bf16"),static_cast<size_t>(rank.rows)*H);
  const auto gate=read_exact<uint16_t>(
      file(packet,"gate-"+tag+".bf16"),width*H);
  const auto up=read_exact<uint16_t>(
      file(packet,"up-"+tag+".bf16"),width*H);
  const auto down=read_exact<uint16_t>(
      file(packet,"down-"+tag+".bf16"),width*H);
  upload(&rank.input,input,rank.stream);
  upload(&rank.gatew,gate,rank.stream);
  upload(&rank.upw,up,rank.stream);
  upload(&rank.downw,down,rank.stream);
  const size_t n=static_cast<size_t>(rank.rows)*width;
  prep(rank.gate,file(cpu,"gate-"+tag+".bf16"),n);
  prep(rank.up,file(cpu,"up-"+tag+".bf16"),n);
  prep(rank.mid,file(cpu,"mid-"+tag+".bf16"),n);
  const size_t h=static_cast<size_t>(rank.rows)*H;
  rank.partial_ref=read_exact<float>(file(cpu,"partial-"+tag+".f32"),h);
  HC(hipMalloc(reinterpret_cast<void**>(&rank.partial),h*4));
  HC(hipMalloc(reinterpret_cast<void**>(&rank.broadcast),h*2));
  HC(hipHostMalloc(reinterpret_cast<void**>(&rank.pinned_partial),
                   h*4,hipHostMallocPortable));
  rank.broadcast_cpu.resize(h);
  HC(hipStreamSynchronize(rank.stream));
}
void launch(Rank& r) {
  HC(hipSetDevice(r.id));
  HC(hipEventRecord(r.begin,r.stream));
  const int warps=8;
  original_ffn_bf16_projection<<<dim3((r.width+warps-1)/warps,r.rows),
                                  256,0,r.stream>>>(
      r.gatew,r.input,r.gate.gpu,r.rows,r.width,H);
  original_ffn_bf16_projection<<<dim3((r.width+warps-1)/warps,r.rows),
                                  256,0,r.stream>>>(
      r.upw,r.input,r.up.gpu,r.rows,r.width,H);
  original_ffn_swiglu<<<std::min(1024,(r.rows*r.width+255)/256),
                         256,0,r.stream>>>(
      r.gate.gpu,r.up.gpu,r.mid.gpu,r.rows*r.width);
  original_ffn_bf16_down<<<dim3((H+warps-1)/warps,r.rows),
                            256,0,r.stream>>>(
      r.downw,r.mid.gpu,r.partial,r.rows,r.width);
  HC(hipMemcpyAsync(r.pinned_partial,r.partial,
                    static_cast<size_t>(r.rows)*H*4,
                    hipMemcpyDeviceToHost,r.stream));
  HC(hipGetLastError());
  HC(hipEventRecord(r.end,r.stream));
}
void compare(Check& item,int device) {
  HC(hipSetDevice(device));
  HC(hipMemcpy(item.returned.data(),item.gpu,
               item.returned.size()*2,hipMemcpyDeviceToHost));
  for(size_t i=0;i<item.reference.size();++i) {
    const double expected=hostbf(item.reference[i]);
    const double got=hostbf(item.returned[i]);
    const double error=std::abs(expected-got);
    item.max_abs=std::max(item.max_abs,error);
    if(item.returned[i]==item.reference[i])++item.exact;
    if(!std::isfinite(got) || error>.025+.03*std::abs(expected))
      throw std::runtime_error("VTP25 original BF16 FFN gate/up/SwiGLU GPU vs CPU mismatch");
  }
}
void check_rank(Rank& r) {
  compare(r.gate,r.id);compare(r.up,r.id);compare(r.mid,r.id);
  for(size_t i=0;i<r.partial_ref.size();++i) {
    const double expected=r.partial_ref[i];
    const double got=r.pinned_partial[i];
    const double error=std::abs(expected-got);
    r.max_partial_abs=std::max(r.max_partial_abs,error);
    if(!std::isfinite(got) || error>.15+.03*std::abs(expected))
      throw std::runtime_error("VTP25 original BF16 FFN down projection CPU mismatch");
  }
}
void destroy(Rank& r) {
  HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
  for(auto* p:{r.input,r.gatew,r.upw,r.downw,r.broadcast,
               r.gate.gpu,r.up.gpu,r.mid.gpu})
    HC(hipFree(p));
  HC(hipFree(r.partial));HC(hipHostFree(r.pinned_partial));
  HC(hipEventDestroy(r.begin));HC(hipEventDestroy(r.end));
  HC(hipStreamDestroy(r.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const int iterations=bounded(argc,argv,"--iterations",30);
    const std::string packet=arg(argc,argv,"--pack-dir");
    const std::string cpu=arg(argc,argv,"--cpu-dir");
    const auto residual=read_exact<uint16_t>(
        file(packet,"input.bf16"),static_cast<size_t>(rows)*H);
    const auto reduction_ref=read_exact<float>(
        file(cpu,"original-ffn-reduced.f32"),static_cast<size_t>(rows)*H);
    const auto final_ref=read_exact<uint16_t>(
        file(cpu,"original-layer0-post-ffn.bf16"),static_cast<size_t>(rows)*H);
    const auto ids=devices();
    std::array<Rank,3> ranks{};
    for(int i=0;i<3;++i) {
      ranks[i].id=ids[i];ranks[i].rows=rows;ranks[i].width=WIDTHS[i];
      setup(ranks[i],packet,cpu,i);
    }
    const size_t n=static_cast<size_t>(rows)*H;
    HC(hipSetDevice(ids[0]));
    uint16_t* pinned_final=nullptr;
    HC(hipHostMalloc(reinterpret_cast<void**>(&pinned_final),
                     n*2,hipHostMallocPortable));
    std::vector<double> walls;
    std::vector<float> reduction(n);
    for(int it=-2;it<iterations;++it) {
      const auto start=Clock::now();
      for(auto& r:ranks)launch(r);  // enqueue all 3 before any waiting
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));HC(hipEventSynchronize(r.end));
      }
      for(size_t i=0;i<n;++i) {
        float value=ranks[0].pinned_partial[i];
        value+=ranks[1].pinned_partial[i];
        value+=ranks[2].pinned_partial[i];
        reduction[i]=value;
        pinned_final[i]=hostround(hostbf(hostround(value))+hostbf(residual[i]));
      }
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        HC(hipMemcpyAsync(r.broadcast,pinned_final,n*2,
                          hipMemcpyHostToDevice,r.stream));
      }
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
      }
      if(it<0)continue;
      walls.push_back(elapsed(start));
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));
        float ms=0;HC(hipEventElapsedTime(&ms,r.begin,r.end));
        r.gpu_times.push_back(static_cast<double>(ms)*1000.0);
      }
    }
    for(auto& r:ranks)check_rank(r);
    double max_partial=0,max_gate=0,max_mid=0,max_reduction=0,max_final=0;
    for(auto& r:ranks) {
      max_partial=std::max(max_partial,r.max_partial_abs);
      max_gate=std::max({max_gate,r.gate.max_abs,r.up.max_abs});
      max_mid=std::max(max_mid,r.mid.max_abs);
      HC(hipSetDevice(r.id));
      HC(hipMemcpy(r.broadcast_cpu.data(),r.broadcast,n*2,hipMemcpyDeviceToHost));
      if(!std::equal(r.broadcast_cpu.begin(),r.broadcast_cpu.end(),pinned_final))
        throw std::runtime_error("VTP25 FFN next-layer GPU residual broadcast mismatch");
    }
    size_t exact=0;
    for(size_t i=0;i<n;++i) {
      const float got=reduction[i],expect=reduction_ref[i];
      const double e=std::abs(got-expect);
      max_reduction=std::max(max_reduction,e);
      if(!std::isfinite(got) || e>.2+.04*std::abs(expect))
        throw std::runtime_error("VTP25 FFN three-device F32 reduction differs from CPU oracle");
      if(pinned_final[i]==final_ref[i])++exact;
      const double bf=std::abs(hostbf(pinned_final[i])-hostbf(final_ref[i]));
      max_final=std::max(max_final,bf);
      if(bf>.2+.04*std::abs(hostbf(final_ref[i])))
        throw std::runtime_error("VTP25 original FFN next-layer BF16 residual outside tolerance");
    }
    printf("mode,rows,wall_us_p50,wall_us_p95,xtx0_gpu_us_p50,xtx1_gpu_us_p50,"
           "r9700_gpu_us_p50,max_gate_up_abs,max_swiglu_abs,max_down_abs,"
           "max_reduction_abs,max_post_ffn_bf16_abs,final_exact,total,oracle\n");
    printf("original_bf16_layer0_post_gdn_ffn_three_GPU,%d,"
           "%.2f,%.2f,%.2f,%.2f,%.2f,%.8g,%.8g,%.8g,%.8g,%.8g,%zu,%zu,PASS\n",
           rows,percentile(walls,.5),percentile(walls,.95),
           percentile(ranks[0].gpu_times,.5),percentile(ranks[1].gpu_times,.5),
           percentile(ranks[2].gpu_times,.5),
           max_gate,max_mid,max_partial,max_reduction,max_final,exact,n);
    printf("physical_ids=%d,%d,%d; original_layer0_FFN_weights=YES;"
           " GDN_residual_cpu_input=YES; post_attention_layernorm=YES;"
           " FFN_widths=5120/4864/7424; host_ordered_F32_sum=YES;"
           " next_layer_residual_broadcast=YES; synthetic_layer0_hidden=YES;"
           " no_P2P=YES; model64=NO; native_Radiance=NO\n",
           ids[0],ids[1],ids[2]);
    for(auto& r:ranks)destroy(r);
    HC(hipSetDevice(ids[0]));HC(hipHostFree(pinned_final));
    return 0;
  }catch(const std::exception& e) {
    fprintf(stderr,"VTP25 original BF16 FFN 3 GPU FAILED: %s\n",e.what());
    return 1;
  }
}
