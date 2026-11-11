// VTP13 opt-in model-shaped INT8 FFN -> physical TP3 host collective.
// Independent BF16-valued SYNTHETIC weight and activation oracle: NOT checkpoint
// payloads, NOT Radiance GEMM kernels, NOT quality or serving speed qualification.
// Weight values are exact BF16 multiples of 1/64; input is exact BF16 /16.
// An entire asymmetric 5120/4864/7424 SwiGLU layer is computed concurrently
// on two gfx1100 devices and one gfx1201 device, with a host F32 reduction.
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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define HC(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
  std::exit(2); } } while (0)

constexpr int H = 5120;
constexpr std::array<int,3> WIDTHS{5120,4864,7424};
constexpr float W_SCALE = 1.0f / 64.0f;
constexpr float X_SCALE = 1.0f / 16.0f;
constexpr int INTERMEDIATE = 17408;
struct Source { std::string path; uint64_t offset=0; };
struct CheckpointWeights { bool enabled=false; std::array<Source,3> tensors{}; };
static const char* argument(int argc,char** argv,const char* option) {
  for(int i=1;i<argc;++i) if(std::strcmp(argv[i],option)==0) {
    if(i+1>=argc) throw std::runtime_error(std::string("missing value: ")+option);
    return argv[i+1];
  }
  return nullptr;
}
static CheckpointWeights read_checkpoint_args(int argc,char** argv) {
  CheckpointWeights result;
  constexpr const char* paths[]={"--gate-file","--up-file","--down-file"};
  constexpr const char* offsets[]={"--gate-byte","--up-byte","--down-byte"};
  if (!argument(argc,argv,paths[0])) {
    for(int j=0;j<3;++j)
      if(argument(argc,argv,paths[j]) || argument(argc,argv,offsets[j]))
        throw std::runtime_error("partial BF16 source spec");
    return result;
  }
  result.enabled=true;
  for (int j=0;j<3;++j) {
    const char* path=argument(argc,argv,paths[j]);
    const char* offset=argument(argc,argv,offsets[j]);
    if(!path || !offset) throw std::runtime_error("partial BF16 source spec");
    char* end=nullptr;
    const unsigned long long pos=std::strtoull(offset,&end,10);
    if(!*offset || *offset=='-' || !end || *end)
      throw std::runtime_error("invalid tensor absolute byte offset");
    result.tensors[j]={path,static_cast<uint64_t>(pos)};
  }
  return result;
}
static float bf16_to_float(uint16_t v) {
  uint32_t bits=static_cast<uint32_t>(v)<<16;
  float f=0;std::memcpy(&f,&bits,sizeof(float));return f;
}
using Clock = std::chrono::steady_clock;
static double us(Clock::time_point t) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
static double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  // Nearest-rank percentile: with three iterations p95 must select the worst,
  // not the median (floor((n-1)*p) loses the upper tail).
  const auto rank=std::clamp(static_cast<size_t>(std::ceil(p*v.size())), size_t{1}, v.size());
  return v[rank-1];
}
static int positive_arg(int argc, char** argv, const char* name, int fallback) {
  for (int i=1; i<argc; ++i) if (std::strcmp(argv[i], name)==0) {
    if (i+1>=argc) throw std::runtime_error(std::string("missing value for ")+name);
    char* end=nullptr;
    const long v=std::strtol(argv[i+1], &end, 10);
    if (!end || *end || v<1 || v>100) throw std::runtime_error(std::string("invalid ")+name);
    return static_cast<int>(v);
  }
  return fallback;
}
static std::array<int,3> physical_devices() {
  int n=0; HC(hipGetDeviceCount(&n));
  std::vector<int> xtx, r;
  for (int i=0; i<n; ++i) {
    hipDeviceProp_t prop{}; HC(hipGetDeviceProperties(&prop, i));
    if (std::strncmp(prop.gcnArchName, "gfx1100", 7)==0) xtx.push_back(i);
    if (std::strncmp(prop.gcnArchName, "gfx1201", 7)==0) r.push_back(i);
  }
  if (xtx.size()!=2 || r.size()!=1)
    throw std::runtime_error("requires exactly two visible gfx1100 and one gfx1201");
  return {xtx[0],xtx[1],r[0]};
}
static int8_t bf16_source_weight(int op, int global_r, int global_k) {
  // A deterministic exact BF16-source value = integer / 64; not the checkpoint.
  uint32_t s = static_cast<uint32_t>(global_r) * 1664525u
             ^ static_cast<uint32_t>(global_k) * 1013904223u
             ^ static_cast<uint32_t>(op + 11) * 2246822519u;
  s ^= s >> 15;
  return static_cast<int8_t>(static_cast<int>(s % 9) - 4);
}
static float bf16_source_input(int row, int col) {
  return static_cast<float>(((col * 13 + row * 7) % 17) - 8) * X_SCALE;
}
static float swiglu(float gate, float up) {
  return (gate / (1.0f + std::exp(-gate))) * up;
}
__global__ void quant_x(const float* x, int8_t* q, int count) {
  for (int idx=blockIdx.x*blockDim.x+threadIdx.x; idx<count; idx+=gridDim.x*blockDim.x)
    q[idx] = static_cast<int8_t>(nearbyintf(x[idx] / X_SCALE));
}
// Each wave performs one true INT8 dot-product (integer multiply/accumulate).
// Deliberately plain baseline, not a claim of a competitive Radiance row kernel.
__global__ void dot_i8(const int8_t* w, const int8_t* x, const float* x_scales,
                       const float* w_scales, float* y, int n, int k) {
  const int lane=threadIdx.x%warpSize;
  const int output=blockIdx.x*(blockDim.x/warpSize)+threadIdx.x/warpSize;
  const int row=blockIdx.y;
  int32_t acc=0;
  if (output<n) {
    const int8_t* wr=w+static_cast<size_t>(output)*k;
    const int8_t* xr=x+static_cast<size_t>(row)*k;
    for (int col=lane; col<k; col+=warpSize)
      acc+=static_cast<int32_t>(wr[col])*static_cast<int32_t>(xr[col]);
  }
  for (int offset=warpSize/2; offset>0; offset/=2)
    acc+=__shfl_down(acc, offset);
  if (lane==0 && output<n) y[static_cast<size_t>(row)*n+output] =
    static_cast<float>(acc)*(w_scales[output]*x_scales[row]);
}
__global__ void silu_mul(const float* gate,const float* up,float* out,int n) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) {
    const float g=gate[i];
    out[i]=(g / (1.0f + expf(-g))) * up[i];
  }
}
__global__ void quant_mid(const float* mid,int8_t* q,float* scales,int n) {
  const int row=blockIdx.x;
  __shared__ float cache[256];
  float maxv=0.0f;
  for (int j=threadIdx.x; j<n; j+=blockDim.x)
    maxv=fmaxf(maxv,fabsf(mid[static_cast<size_t>(row)*n+j]));
  cache[threadIdx.x]=maxv;
  __syncthreads();
  for (int step=128; step>0; step/=2) {
    if (threadIdx.x<step) cache[threadIdx.x]=fmaxf(cache[threadIdx.x],cache[threadIdx.x+step]);
    __syncthreads();
  }
  const float scale=cache[0]>0.0f ? cache[0]/127.0f : 1.0f;
  if (threadIdx.x==0) scales[row]=scale;
  for (int j=threadIdx.x; j<n; j+=blockDim.x) {
    float v=nearbyintf(mid[static_cast<size_t>(row)*n+j]/scale);
    q[static_cast<size_t>(row)*n+j]=static_cast<int8_t>(fmaxf(-127.0f,fminf(127.0f,v)));
  }
}
struct Device {
  int id=-1, width=0, offset=0;
  hipStream_t stream=nullptr;
  hipEvent_t begin=nullptr, ready=nullptr;
  int8_t *wg=nullptr,*wu=nullptr,*wd=nullptr,*xq=nullptr,*midq=nullptr;
  float *x=nullptr,*xs=nullptr,*gate=nullptr,*up=nullptr,*mid=nullptr,
        *wg_scales=nullptr,*wu_scales=nullptr,*wd_scales=nullptr,
        *mid_scales=nullptr,*partial=nullptr,*result=nullptr;
  std::vector<int8_t> hgate,hup,hdown;
  std::vector<float> hgate_scales,hup_scales,hdown_scales;
  std::vector<uint16_t> bgate,bup,bdown;
  bool original_bf16=false;
  std::vector<double> gpu_us;
};
static void generate_weights(Device& d) {
  const int n=d.width, off=d.offset;
  d.hgate.resize(static_cast<size_t>(n)*H);
  d.hup.resize(static_cast<size_t>(n)*H);
  d.hdown.resize(static_cast<size_t>(H)*n);
  d.hgate_scales.assign(n,W_SCALE);
  d.hup_scales.assign(n,W_SCALE);
  d.hdown_scales.assign(H,W_SCALE);
  for (int j=0; j<n; ++j) for (int k=0;k<H;++k) {
    d.hgate[static_cast<size_t>(j)*H+k]=bf16_source_weight(0,off+j,k);
    d.hup[static_cast<size_t>(j)*H+k]=bf16_source_weight(1,off+j,k);
  }
  for (int j=0; j<H; ++j) for (int k=0;k<n;++k)
    d.hdown[static_cast<size_t>(j)*n+k]=bf16_source_weight(2,j,off+k);
}
// Original BF16 source ranges are supplied by the metadata-verifying Python
// wrapper. Per-output-channel scales are independent for every physical shard.
static void load_bf16_projection(const Source& src, int global_rows,int full_k,
                                 int local_rows,int local_k,int row_start,int col_start,
                                 std::vector<int8_t>& quant,std::vector<float>& scales,
                                 std::vector<uint16_t>& original) {
  std::ifstream file(src.path,std::ios::binary|std::ios::ate);
  if(!file) throw std::runtime_error("cannot open BF16 tensor shard");
  const uint64_t expected_end=src.offset+
      static_cast<uint64_t>(global_rows)*full_k*sizeof(uint16_t);
  if(file.tellg()<0 || static_cast<uint64_t>(file.tellg())<expected_end)
    throw std::runtime_error("truncated original BF16 tensor source");
  const size_t count=static_cast<size_t>(local_rows)*local_k;
  quant.resize(count); original.resize(count);scales.resize(local_rows);
  for(int row=0;row<local_rows;++row) {
    const uint64_t off=src.offset+
        (static_cast<uint64_t>(row+row_start)*full_k+col_start)*sizeof(uint16_t);
    file.seekg(static_cast<std::streamoff>(off),std::ios::beg);
    const size_t base=static_cast<size_t>(row)*local_k;
    file.read(reinterpret_cast<char*>(original.data()+base),
              static_cast<std::streamsize>(local_k*sizeof(uint16_t)));
    if(!file) throw std::runtime_error("truncated original BF16 row payload");
    float maxabs=0.0f;
    for(int k=0;k<local_k;++k) {
      const float v=bf16_to_float(original[base+k]);
      if(!std::isfinite(v)) throw std::runtime_error("nonfinite original BF16 tensor");
      maxabs=std::max(maxabs,std::abs(v));
    }
    const float scale=maxabs>0.0f ? maxabs/127.0f : 1.0f;
    scales[row]=scale;
    for(int k=0;k<local_k;++k) {
      const float value=nearbyintf(bf16_to_float(original[base+k])/scale);
      quant[base+k]=static_cast<int8_t>(
          std::max(-127.0f,std::min(127.0f,value)));
    }
  }
}
static void load_checkpoint_weights(Device& d,const CheckpointWeights& spec) {
  if(!spec.enabled) throw std::runtime_error("no BF16 checkpoint sources");
  const int n=d.width, off=d.offset;
  load_bf16_projection(spec.tensors[0],INTERMEDIATE,H,n,H,off,0,
                       d.hgate,d.hgate_scales,d.bgate);
  load_bf16_projection(spec.tensors[1],INTERMEDIATE,H,n,H,off,0,
                       d.hup,d.hup_scales,d.bup);
  load_bf16_projection(spec.tensors[2],H,INTERMEDIATE,H,n,0,off,
                       d.hdown,d.hdown_scales,d.bdown);
  d.original_bf16=true;
}
static void init(Device& d,int rows,const std::vector<float>& input) {
  HC(hipSetDevice(d.id));
  HC(hipStreamCreateWithFlags(&d.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&d.begin)); HC(hipEventCreate(&d.ready));
  const auto gw=static_cast<size_t>(d.width)*H;
  const auto dw=static_cast<size_t>(H)*d.width;
  const auto hbytes=static_cast<size_t>(rows)*H*sizeof(float);
  const auto ibytes=static_cast<size_t>(rows)*d.width;
  HC(hipMalloc(reinterpret_cast<void**>(&d.wg),gw));
  HC(hipMalloc(reinterpret_cast<void**>(&d.wu),gw));
  HC(hipMalloc(reinterpret_cast<void**>(&d.wd),dw));
  HC(hipMalloc(reinterpret_cast<void**>(&d.wg_scales),d.width*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.wu_scales),d.width*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.wd_scales),H*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.x),hbytes));
  HC(hipMalloc(reinterpret_cast<void**>(&d.xq),static_cast<size_t>(rows)*H));
  HC(hipMalloc(reinterpret_cast<void**>(&d.xs),rows*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.gate),ibytes*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.up),ibytes*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.mid),ibytes*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.midq),ibytes));
  HC(hipMalloc(reinterpret_cast<void**>(&d.mid_scales),rows*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&d.partial),hbytes));
  HC(hipMalloc(reinterpret_cast<void**>(&d.result),hbytes));
  HC(hipMemcpyAsync(d.wg,d.hgate.data(),gw,hipMemcpyHostToDevice,d.stream));
  HC(hipMemcpyAsync(d.wu,d.hup.data(),gw,hipMemcpyHostToDevice,d.stream));
  HC(hipMemcpyAsync(d.wd,d.hdown.data(),dw,hipMemcpyHostToDevice,d.stream));
  HC(hipMemcpyAsync(d.wg_scales,d.hgate_scales.data(),
                    d.width*sizeof(float),hipMemcpyHostToDevice,d.stream));
  HC(hipMemcpyAsync(d.wu_scales,d.hup_scales.data(),
                    d.width*sizeof(float),hipMemcpyHostToDevice,d.stream));
  HC(hipMemcpyAsync(d.wd_scales,d.hdown_scales.data(),
                    H*sizeof(float),hipMemcpyHostToDevice,d.stream));
  HC(hipMemcpyAsync(d.x,input.data(),hbytes,hipMemcpyHostToDevice,d.stream));
  std::vector<float> ones(rows,X_SCALE);
  HC(hipMemcpyAsync(d.xs,ones.data(),rows*sizeof(float),hipMemcpyHostToDevice,d.stream));
  HC(hipStreamSynchronize(d.stream));
}
static void launch(Device& d,int rows) {
  HC(hipSetDevice(d.id));
  const int n=d.width;
  const auto blocks=[&](int v){ return std::min(1024,(v+255)/256); };
  HC(hipEventRecord(d.begin,d.stream));
  quant_x<<<blocks(rows*H),256,0,d.stream>>>(d.x,d.xq,rows*H);
  const dim3 grid((n+7)/8,rows);
  dot_i8<<<grid,256,0,d.stream>>>(d.wg,d.xq,d.xs,d.wg_scales,d.gate,n,H);
  dot_i8<<<grid,256,0,d.stream>>>(d.wu,d.xq,d.xs,d.wu_scales,d.up,n,H);
  silu_mul<<<blocks(rows*n),256,0,d.stream>>>(d.gate,d.up,d.mid,rows*n);
  quant_mid<<<rows,256,0,d.stream>>>(d.mid,d.midq,d.mid_scales,n);
  dot_i8<<<dim3((H+7)/8,rows),256,0,d.stream>>>(
    d.wd,d.midq,d.mid_scales,d.wd_scales,d.partial,H,n);
  HC(hipGetLastError());
  HC(hipEventRecord(d.ready,d.stream));
}
static std::vector<float> cpu_reference(const std::array<Device,3>& d,
                                        const std::vector<float>& input, int rows,
                                        bool down_quant) {
  std::vector<float> out(static_cast<size_t>(rows)*H,0.0f);
  for (const auto& dev:d) {
    const int n=dev.width;
    std::vector<float> mid(n);
    std::vector<int8_t> midq(n);
    for (int row=0;row<rows;++row) {
      for (int j=0;j<n;++j) {
        int64_t ga=0,up=0;
        double bf_gate=0.0,bf_up=0.0;
        for (int k=0;k<H;++k) {
          const int xq=static_cast<int>(nearbyintf(input[static_cast<size_t>(row)*H+k]/X_SCALE));
          ga+=static_cast<int>(dev.hgate[static_cast<size_t>(j)*H+k])*xq;
          up+=static_cast<int>(dev.hup[static_cast<size_t>(j)*H+k])*xq;
          if(!down_quant && dev.original_bf16) {
            bf_gate+=bf16_to_float(dev.bgate[static_cast<size_t>(j)*H+k])*
                input[static_cast<size_t>(row)*H+k];
            bf_up+=bf16_to_float(dev.bup[static_cast<size_t>(j)*H+k])*
                input[static_cast<size_t>(row)*H+k];
          }
        }
        mid[j]=(!down_quant && dev.original_bf16)
            ? swiglu(static_cast<float>(bf_gate),static_cast<float>(bf_up))
            : swiglu(static_cast<float>(ga)*(dev.hgate_scales[j]*X_SCALE),
                     static_cast<float>(up)*(dev.hup_scales[j]*X_SCALE));
      }
      float scale=1.0f;
      if (down_quant) {
        float maxv=0.0f;
        for(float v:mid) maxv=std::max(maxv,std::abs(v));
        scale=maxv>0.0f ? maxv/127.0f : 1.0f;
        for (int j=0;j<n;++j) {
          const float v=nearbyintf(mid[j]/scale);
          midq[j]=static_cast<int8_t>(std::max(-127.0f,std::min(127.0f,v)));
        }
      }
      for (int j=0;j<H;++j) {
        const int8_t* w=dev.hdown.data()+static_cast<size_t>(j)*n;
        float val=0.0f;
        if (down_quant) {
          int64_t sum=0;
          for (int k=0;k<n;++k) sum+=static_cast<int>(w[k])*midq[k];
          val=static_cast<float>(sum)*(dev.hdown_scales[j]*scale);
        } else {
          // BF16-source weights and BF16-source input, accumulated in F32;
          // SiLU intermediate is unquantized FP32. Reference, not BF16-model quality.
          if(dev.original_bf16) {
            double reference=0.0;
            for(int k=0;k<n;++k)
              reference+=bf16_to_float(dev.bdown[static_cast<size_t>(j)*n+k])*mid[k];
            val=static_cast<float>(reference);
          } else {
            for (int k=0;k<n;++k) val+=static_cast<float>(w[k])*W_SCALE*mid[k];
          }
        }
        out[static_cast<size_t>(row)*H+j]+=val;
      }
    }
  }
  return out;
}
static void destroy(Device& d) {
  HC(hipSetDevice(d.id)); HC(hipStreamSynchronize(d.stream));
  HC(hipFree(d.wg)); HC(hipFree(d.wu)); HC(hipFree(d.wd));
  HC(hipFree(d.wg_scales));HC(hipFree(d.wu_scales));HC(hipFree(d.wd_scales));
  HC(hipFree(d.x)); HC(hipFree(d.xq)); HC(hipFree(d.xs));
  HC(hipFree(d.gate)); HC(hipFree(d.up)); HC(hipFree(d.mid));
  HC(hipFree(d.midq)); HC(hipFree(d.mid_scales));
  HC(hipFree(d.partial)); HC(hipFree(d.result));
  HC(hipEventDestroy(d.begin)); HC(hipEventDestroy(d.ready));
  HC(hipStreamDestroy(d.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=positive_arg(argc,argv,"--rows",1);
    const int iterations=positive_arg(argc,argv,"--iterations",5);
    if (rows>64 || iterations>30) throw std::runtime_error("bounded to <=64 rows and <=30 iterations");
    const CheckpointWeights checkpoint=read_checkpoint_args(argc,argv);
    bool reverse=false;
    for (int i=1;i<argc;++i) if (std::strcmp(argv[i],"--reverse")==0) reverse=true;
    const auto ids=physical_devices();
    std::array<Device,3> dev{};
    int offset=0;
    for (int i=0;i<3;++i) {
      dev[i].id=ids[i];dev[i].width=WIDTHS[i];dev[i].offset=offset;offset+=WIDTHS[i];
    }
    if (offset!=17408) throw std::runtime_error("illegal shard coverage");
    std::vector<float> input(static_cast<size_t>(rows)*H);
    for (int r=0;r<rows;++r) for(int k=0;k<H;++k)
      input[static_cast<size_t>(r)*H+k]=bf16_source_input(r,k);
    for (auto& d:dev) {
      if(checkpoint.enabled) load_checkpoint_weights(d,checkpoint);
      else generate_weights(d);
      init(d,rows,input);
    }
    // The oracle is computed separately from GPU outputs, before GPU timing.
    const auto expected=cpu_reference(dev,input,rows,true);
    const auto bf16ref=cpu_reference(dev,input,rows,false);
    const size_t count=static_cast<size_t>(rows)*H,bytes=count*sizeof(float);
    std::array<float*,4> host{};
    for (float*& ptr:host) HC(hipHostMalloc(reinterpret_cast<void**>(&ptr),bytes));
    std::vector<double> total_times,barriers,d2hs,sums,h2ds;
    const auto once=[&](){
      const auto start=Clock::now();
      for (int t=0;t<3;++t) launch(dev[reverse?2-t:t],rows);
      for (auto& d:dev) {HC(hipSetDevice(d.id));HC(hipEventSynchronize(d.ready));}
      const double barrier=us(start);
      for (auto& d:dev) {
        HC(hipSetDevice(d.id));
        float ms=0;HC(hipEventElapsedTime(&ms,d.begin,d.ready));
        d.gpu_us.push_back(static_cast<double>(ms)*1000.0);
      }
      const auto copy=Clock::now();
      for (int i=0;i<3;++i) {
        HC(hipSetDevice(dev[i].id));
        HC(hipMemcpyAsync(host[i],dev[i].partial,bytes,hipMemcpyDeviceToHost,dev[i].stream));
      }
      for (auto& d:dev) {HC(hipSetDevice(d.id));HC(hipStreamSynchronize(d.stream));}
      const double d2h=us(copy);
      const auto sum=Clock::now();
      for (size_t j=0;j<count;++j) host[3][j]=(host[0][j]+host[1][j])+host[2][j];
      const double cpu=us(sum);
      const auto back=Clock::now();
      for (auto& d:dev) {
        HC(hipSetDevice(d.id));
        HC(hipMemcpyAsync(d.result,host[3],bytes,hipMemcpyHostToDevice,d.stream));
      }
      for (auto& d:dev) {HC(hipSetDevice(d.id));HC(hipStreamSynchronize(d.stream));}
      return std::array<double,5>{us(start),barrier,d2h,cpu,us(back)};
    };
    for (int w=0;w<2;++w) (void)once();
    for (auto& d:dev) d.gpu_us.clear();
    for (int it=0;it<iterations;++it) {
      const auto t=once();total_times.push_back(t[0]);barriers.push_back(t[1]);
      d2hs.push_back(t[2]);sums.push_back(t[3]);h2ds.push_back(t[4]);
    }
    double maxabs=0.0,maxrel=0.0,maxbf=0.0,bfse=0.0;
    for(size_t j=0;j<count;++j) {
      const double diff=std::abs(static_cast<double>(host[3][j])-expected[j]);
      maxabs=std::max(maxabs,diff);
      maxrel=std::max(maxrel,diff/(1.0+std::abs(expected[j])));
      const double bf=std::abs(static_cast<double>(host[3][j])-bf16ref[j]);
      maxbf=std::max(maxbf,bf);bfse+=bf*bf;
    }
    // Numerical oracle after exact three-way CPU sum and H2D broadcast:
    // check every output element of every physical GPU independently.
    std::vector<float> check(count);
    for (auto& d:dev) {
      HC(hipSetDevice(d.id));
      HC(hipMemcpy(check.data(),d.result,bytes,hipMemcpyDeviceToHost));
      for (size_t j=0;j<count;++j) {
        const double tol=0.001 + 0.003 * std::abs(expected[j]);
        if (!std::isfinite(check[j]) || std::abs(check[j]-expected[j])>tol)
          throw std::runtime_error("BF16-derived independent INT8 FFN CPU oracle mismatch");
        if (check[j]!=host[3][j])
          throw std::runtime_error("rank-divergent physical TP3 broadcast");
      }
    }
    std::printf("mode,rows,widths,wall_us_p50,wall_us_p95,compute_barrier_us_p50,d2h_us_p50,host_f32_sum_us_p50,h2d_us_p50,gpu0_us_p50,gpu1_us_p50,gpu2_us_p50,oracle_max_abs,oracle_max_rel,bf16_source_max_abs,bf16_source_rmse,packets,oracle\n");
    std::printf("%s,%d,5120/4864/7424,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.7g,%.7g,%.7g,%.7g,3,PASS\n",
      checkpoint.enabled ? "checkpoint_bf16_weights_int8_tp3"
                         : "synthetic_bf16_w8a8_ffn_tp3",
      rows,percentile(total_times,.5),percentile(total_times,.95),
      percentile(barriers,.5),percentile(d2hs,.5),percentile(sums,.5),
      percentile(h2ds,.5),percentile(dev[0].gpu_us,.5),
      percentile(dev[1].gpu_us,.5),percentile(dev[2].gpu_us,.5),
      maxabs,maxrel,maxbf,std::sqrt(bfse/count));
    std::printf("physical_device_ids=%d,%d,%d; source=%s; actual_qwen_tensor_payloads=%s; radiance_native_gemm=NO; real_gemms=YES; collective=three_host_packets_f32\n",
      ids[0],ids[1],ids[2],checkpoint.enabled ? "original_bf16_weights" : "synthetic_bf16_exact",
      checkpoint.enabled ? "YES" : "NO");
    for (float* ptr:host) HC(hipHostFree(ptr));
    for (auto& d:dev) destroy(d);
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"VTP13 INT8 FFN collective FAILED: %s\n",e.what());
    return 1;
  }
}
