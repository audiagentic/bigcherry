// VTP13: physical 1/1/2-KV-head attention and 4/4/8 key,
// 12/12/24 value GDN head-state concurrency micro-oracles.
// Original Qwen3.8-27B head geometry. Synthetic exact-BF16 Q/K/V
// (no model projection weights); GDN recurrence is an explicit bounded
// simplified state update, NOT Qwen's full conv + gated delta implementation.
// One HIP context and stream per physical device, no P2P.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define HC(x) do { auto e_=(x); if(e_!=hipSuccess) { \
  std::fprintf(stderr,"HIP: %s at %s:%d\n",hipGetErrorString(e_),__FILE__,__LINE__);std::exit(2); \
} } while(0)
constexpr int ATTN_DIM=256, GDN_DIM=128;
constexpr std::array<int,3> QHEADS{6,6,12};
constexpr std::array<int,3> KVHEADS{1,1,2};
constexpr std::array<int,3> GDN_KEYS{4,4,8};
constexpr std::array<int,3> GDN_VALUES{12,12,24};
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double,std::micro>(Clock::now()-t).count();
}
int positive(int argc,char** argv,const char* opt,int def) {
  for(int i=1;i<argc;++i) if(std::strcmp(argv[i],opt)==0) {
    if(i+1>=argc) throw std::runtime_error(std::string("missing ")+opt);
    char* e=nullptr;long value=std::strtol(argv[i+1],&e,10);
    if(!e || *e || value<1 || value>4096) throw std::runtime_error(std::string("invalid ")+opt);
    return static_cast<int>(value);
  }
  return def;
}
bool flag(int argc,char** argv,const char* opt) {
  for(int i=1;i<argc;++i) if(std::strcmp(argv[i],opt)==0)return true;
  return false;
}
std::array<int,3> devices() {
  int n=0;HC(hipGetDeviceCount(&n));
  std::vector<int> xtx,r9700;
  for(int i=0;i<n;++i) {
    hipDeviceProp_t prop{};HC(hipGetDeviceProperties(&prop,i));
    if(std::strncmp(prop.gcnArchName,"gfx1100",7)==0)xtx.push_back(i);
    if(std::strncmp(prop.gcnArchName,"gfx1201",7)==0)r9700.push_back(i);
  }
  if(xtx.size()!=2 || r9700.size()!=1)
    throw std::runtime_error("requires exactly 2 gfx1100 and one gfx1201");
  return {xtx[0],xtx[1],r9700[0]};
}
static float q_value(int row,int head,int d,int mode) {
  int v=(row*7+head*13+d*9+mode*19)%37-18;
  return static_cast<float>(v)/64.0f;
}
static float k_value(int row,int head,int d,int mode) {
  int v=(row*11+head*17+d*5+mode*13)%41-20;
  return static_cast<float>(v)/64.0f;
}
static float v_value(int row,int head,int d,int mode) {
  int v=(row*5+head*7+d*11+mode*23)%43-21;
  return static_cast<float>(v)/64.0f;
}
// One thread processes one output dimension and all context tokens.
// One block per Q head per request. For R9700, its TWO KV heads and
// TWELVE Q heads are in the SAME launch (12 concurrent blocks per row).
__global__ void full_attention(const float* q,const float* k,const float* v,
                               float* out,int q_heads,int kv_heads,int rows,int context) {
  const int head=blockIdx.x,row=blockIdx.y,d=threadIdx.x;
  const int kv=head/6; // original 24Q:4KV GQA grouping
  if(kv>=kv_heads || d>=ATTN_DIM)return;
  __shared__ float reduction[ATTN_DIM];
  __shared__ float factors[2];
  float weighted=0.0f,denom=0.0f,max_score=-1.e30f;
  for(int t=0;t<context;++t) {
    const size_t qi=(static_cast<size_t>(row)*q_heads+head)*ATTN_DIM+d;
    const size_t ki=(static_cast<size_t>(kv)*context+t)*ATTN_DIM+d;
    reduction[d]=q[qi]*k[ki];
    __syncthreads();
    for(int step=ATTN_DIM/2;step>=1;step/=2) {
      if(d<step)reduction[d]+=reduction[d+step];
      __syncthreads();
    }
    if(d==0) {
      const float score=reduction[0]*(1.0f/16.0f);
      const float next=fmaxf(max_score,score);
      factors[0]=expf(max_score-next);
      factors[1]=expf(score-next);
      max_score=next;
    }
    __syncthreads();
    // All 256 dimension lanes must share the same running denominator.
    denom=denom*factors[0]+factors[1];
    weighted=weighted*factors[0]+factors[1]*v[ki];
    __syncthreads();
  }
  out[(static_cast<size_t>(row)*q_heads+head)*ATTN_DIM+d]=weighted/denom;
}
// One block per GDN VALUE head. Each owns a persistent 128x128 F32
// recurrent state. Three value heads share one key head. All 24 R9700
// value heads (8 distinct key heads) update concurrently in one launch.
// Qwen-style state dimension, but not full Qwen GDN: no conv/A_log/dt.
__global__ void gdn_update(const float* q,const float* k,const float* v,
                           float* state,float* out,int keys,int values,int rows) {
  const int h=blockIdx.x,dim=threadIdx.x,key=h/3;
  if(dim>=GDN_DIM || key>=keys || h>=values)return;
  for(int row=0;row<rows;++row) {
    const size_t ki=(static_cast<size_t>(row)*keys+key)*GDN_DIM;
    const size_t vi=(static_cast<size_t>(row)*values+h)*GDN_DIM+dim;
    float pred=0.0f;
    for(int i=0;i<GDN_DIM;++i)
      pred+=state[(static_cast<size_t>(h)*GDN_DIM+i)*GDN_DIM+dim]*k[ki+i];
    const float delta=0.5f*(v[vi]-pred);
    float projected=0.0f;
    for(int i=0;i<GDN_DIM;++i) {
      const size_t si=(static_cast<size_t>(h)*GDN_DIM+i)*GDN_DIM+dim;
      const float updated=0.99f*state[si]+k[ki+i]*delta;
      state[si]=updated;
      projected+=q[ki+i]*updated;
    }
    out[vi]=projected;
  }
}
struct Rank {
  int id=-1, q_heads=0,kv_heads=0,gdn_keys=0,gdn_values=0;
  hipStream_t stream=nullptr;
  hipEvent_t start=nullptr,finish=nullptr;
  float *dq=nullptr,*dk=nullptr,*dv=nullptr,*dout=nullptr,*state=nullptr;
  std::vector<float> hq,hk,hv,checked;
  size_t state_bytes=0,output_count=0;
  std::vector<double> timings;
};
void init(Rank& r,int rows,int context,bool gdn) {
  HC(hipSetDevice(r.id));
  HC(hipStreamCreateWithFlags(&r.stream,hipStreamNonBlocking));
  HC(hipEventCreate(&r.start));HC(hipEventCreate(&r.finish));
  if(gdn) {
    r.hq.resize(static_cast<size_t>(rows)*r.gdn_keys*GDN_DIM);
    r.hk.resize(r.hq.size());
    r.hv.resize(static_cast<size_t>(rows)*r.gdn_values*GDN_DIM);
    r.output_count=r.hv.size();
    r.state_bytes=static_cast<size_t>(r.gdn_values)*GDN_DIM*GDN_DIM*sizeof(float);
    for(int t=0;t<rows;++t) {
      for(int h=0;h<r.gdn_keys;++h)for(int d=0;d<GDN_DIM;++d) {
        const size_t i=(static_cast<size_t>(t)*r.gdn_keys+h)*GDN_DIM+d;
        r.hq[i]=q_value(t,h,d,1);r.hk[i]=k_value(t,h,d,1);
      }
      for(int h=0;h<r.gdn_values;++h)for(int d=0;d<GDN_DIM;++d)
        r.hv[(static_cast<size_t>(t)*r.gdn_values+h)*GDN_DIM+d]=v_value(t,h,d,1);
    }
  } else {
    r.hq.resize(static_cast<size_t>(rows)*r.q_heads*ATTN_DIM);
    r.hk.resize(static_cast<size_t>(context)*r.kv_heads*ATTN_DIM);
    r.hv.resize(r.hk.size());
    r.output_count=r.hq.size();
    for(int t=0;t<rows;++t)for(int h=0;h<r.q_heads;++h)for(int d=0;d<ATTN_DIM;++d)
      r.hq[(static_cast<size_t>(t)*r.q_heads+h)*ATTN_DIM+d]=q_value(t,h,d,0);
    for(int h=0;h<r.kv_heads;++h)for(int t=0;t<context;++t)for(int d=0;d<ATTN_DIM;++d) {
      const size_t i=(static_cast<size_t>(h)*context+t)*ATTN_DIM+d;
      r.hk[i]=k_value(t,h,d,0);r.hv[i]=v_value(t,h,d,0);
    }
  }
  r.checked.resize(r.output_count);
  HC(hipMalloc(reinterpret_cast<void**>(&r.dq),r.hq.size()*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.dk),r.hk.size()*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.dv),r.hv.size()*sizeof(float)));
  HC(hipMalloc(reinterpret_cast<void**>(&r.dout),r.output_count*sizeof(float)));
  if(gdn)HC(hipMalloc(reinterpret_cast<void**>(&r.state),r.state_bytes));
  HC(hipMemcpyAsync(r.dq,r.hq.data(),r.hq.size()*sizeof(float),
                    hipMemcpyHostToDevice,r.stream));
  HC(hipMemcpyAsync(r.dk,r.hk.data(),r.hk.size()*sizeof(float),
                    hipMemcpyHostToDevice,r.stream));
  HC(hipMemcpyAsync(r.dv,r.hv.data(),r.hv.size()*sizeof(float),
                    hipMemcpyHostToDevice,r.stream));
  HC(hipStreamSynchronize(r.stream));
}
void launch(Rank& r,int rows,int context,bool gdn) {
  HC(hipSetDevice(r.id));
  HC(hipEventRecord(r.start,r.stream));
  if(gdn) {
    HC(hipMemsetAsync(r.state,0,r.state_bytes,r.stream));
    gdn_update<<<r.gdn_values,GDN_DIM,0,r.stream>>>(
        r.dq,r.dk,r.dv,r.state,r.dout,r.gdn_keys,r.gdn_values,rows);
  } else {
    full_attention<<<dim3(r.q_heads,rows),ATTN_DIM,0,r.stream>>>(
        r.dq,r.dk,r.dv,r.dout,r.q_heads,r.kv_heads,rows,context);
  }
  HC(hipGetLastError());
  HC(hipEventRecord(r.finish,r.stream));
}
static double pctl(std::vector<double> v,double p) {
  std::sort(v.begin(),v.end());
  return v[std::clamp(static_cast<size_t>(std::ceil(p*v.size())),size_t{1},v.size())-1];
}
static std::vector<float> cpu_attention(const Rank& r,int rows,int context) {
  std::vector<float> result(r.output_count);
  for(int row=0;row<rows;++row)for(int h=0;h<r.q_heads;++h) {
    const int kv=h/6;
    const size_t qo=(static_cast<size_t>(row)*r.q_heads+h)*ATTN_DIM;
    std::vector<double> scores(context);
    double maxscore=-1e300;
    for(int t=0;t<context;++t) {
      const size_t ko=(static_cast<size_t>(kv)*context+t)*ATTN_DIM;
      double dot=0.0;
      for(int d=0;d<ATTN_DIM;++d)dot+=r.hq[qo+d]*r.hk[ko+d];
      scores[t]=dot/16.0;maxscore=std::max(maxscore,scores[t]);
    }
    double denom=0.0;
    for(int t=0;t<context;++t){scores[t]=std::exp(scores[t]-maxscore);denom+=scores[t];}
    for(int d=0;d<ATTN_DIM;++d) {
      double value=0.0;
      for(int t=0;t<context;++t)
        value+=scores[t]*r.hv[(static_cast<size_t>(kv)*context+t)*ATTN_DIM+d];
      result[qo+d]=static_cast<float>(value/denom);
    }
  }
  return result;
}
static std::vector<float> cpu_gdn(const Rank& r,int rows) {
  std::vector<float> result(r.output_count);
  std::vector<float> state(r.state_bytes/sizeof(float));
  for(int h=0;h<r.gdn_values;++h)for(int t=0;t<rows;++t) {
    const int key=h/3;
    const size_t ki=(static_cast<size_t>(t)*r.gdn_keys+key)*GDN_DIM;
    for(int d=0;d<GDN_DIM;++d) {
      const size_t vi=(static_cast<size_t>(t)*r.gdn_values+h)*GDN_DIM+d;
      float pred=0.0f;
      for(int i=0;i<GDN_DIM;++i)
        pred+=state[(static_cast<size_t>(h)*GDN_DIM+i)*GDN_DIM+d]*r.hk[ki+i];
      const float delta=0.5f*(r.hv[vi]-pred);
      float projected=0.0f;
      for(int i=0;i<GDN_DIM;++i) {
        const size_t si=(static_cast<size_t>(h)*GDN_DIM+i)*GDN_DIM+d;
        const float updated=0.99f*state[si]+r.hk[ki+i]*delta;
        state[si]=updated;projected+=r.hq[ki+i]*updated;
      }
      result[vi]=projected;
    }
  }
  return result;
}
static void destroy(Rank& r) {
  HC(hipSetDevice(r.id));HC(hipStreamSynchronize(r.stream));
  HC(hipFree(r.dq));HC(hipFree(r.dk));HC(hipFree(r.dv));HC(hipFree(r.dout));
  if(r.state)HC(hipFree(r.state));
  HC(hipEventDestroy(r.start));HC(hipEventDestroy(r.finish));
  HC(hipStreamDestroy(r.stream));
}
int main(int argc,char** argv) {
  try {
    const int rows=positive(argc,argv,"--rows",1);
    const int context=positive(argc,argv,"--context",256);
    const int iterations=positive(argc,argv,"--iterations",3);
    const bool gdn=flag(argc,argv,"--gdn");
    const bool reverse=flag(argc,argv,"--reverse");
    if(rows>64 || iterations>30 || context>2048)
      throw std::runtime_error("bounds: rows<=64, context<=2048, iterations<=30");
    const auto ids=devices();
    std::array<Rank,3> ranks{};
    for(int i=0;i<3;++i) {
      auto& r=ranks[i];r.id=ids[i];r.q_heads=QHEADS[i];r.kv_heads=KVHEADS[i];
      r.gdn_keys=GDN_KEYS[i];r.gdn_values=GDN_VALUES[i];
      init(r,rows,context,gdn);
    }
    std::vector<double> walls;
    for(int it=-2;it<iterations;++it) {
      const auto begin=Clock::now();
      for(int n=0;n<3;++n)launch(ranks[reverse?2-n:n],rows,context,gdn);
      for(auto& r:ranks) {HC(hipSetDevice(r.id));HC(hipEventSynchronize(r.finish));}
      if(it<0)continue;
      walls.push_back(elapsed(begin));
      for(auto& r:ranks) {
        HC(hipSetDevice(r.id));float ms=0;
        HC(hipEventElapsedTime(&ms,r.start,r.finish));
        r.timings.push_back(static_cast<double>(ms)*1000.0);
      }
    }
    double maxabs=0.0,maxrel=0.0;
    for(auto& r:ranks) {
      HC(hipSetDevice(r.id));
      HC(hipMemcpy(r.checked.data(),r.dout,r.output_count*sizeof(float),
                   hipMemcpyDeviceToHost));
      const auto expected=gdn?cpu_gdn(r,rows):cpu_attention(r,rows,context);
      for(size_t j=0;j<r.output_count;++j) {
        const double diff=std::abs(static_cast<double>(r.checked[j])-expected[j]);
        maxabs=std::max(maxabs,diff);
        maxrel=std::max(maxrel,diff/(1.0+std::abs(expected[j])));
        if(!std::isfinite(r.checked[j]) ||
           diff > 0.0005+0.005*std::abs(expected[j]))
          throw std::runtime_error("R9700 two-head attention/GDN independent CPU oracle mismatch");
      }
    }
    std::printf("mode,rows,context,physical_q_heads,physical_kv_heads,gdn_key_heads,gdn_value_heads,wall_us_p50,wall_us_p95,xtx0_us_p50,xtx1_us_p50,r9700_us_p50,r9700_us_p95,r9700_state_bytes,oracle_max_abs,oracle_max_rel,oracle\n");
    std::printf("%s,%d,%d,6/6/12,1/1/2,4/4/8,12/12/24,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%zu,%.8g,%.8g,PASS\n",
      gdn?"simplified_gdn_state_head_local":"full_attention_gqa_head_local",
      rows,context,pctl(walls,.5),pctl(walls,.95),
      pctl(ranks[0].timings,.5),pctl(ranks[1].timings,.5),
      pctl(ranks[2].timings,.5),pctl(ranks[2].timings,.95),
      ranks[2].state_bytes,maxabs,maxrel);
    std::printf("physical_ids=%d,%d,%d; model_weights=NO; real_attention_math=%s; real_recurrent_update=%s; full_qwen_gdn=NO; head_concurrency=yes; p2p=no\n",
      ids[0],ids[1],ids[2],gdn?"NO":"YES",gdn?"YES":"NO");
    for(auto& r:ranks)destroy(r);
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"VTP13 head-local experiment FAILED: %s\n",e.what());
    return 1;
  }
}
