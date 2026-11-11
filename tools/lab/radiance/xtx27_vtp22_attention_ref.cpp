// VTP22 original BF16 Qwen3.8-27B full-attention layer-3 CPU oracle.
//
// Reads byte-exact real checkpoint Q/K/V/O physical rank packages.
// Q rows are *interleaved per head*: query 256, output gate 256.
// Q and K use (1 + BF16 weight) RMSNorm with eps 1e-6.
// Text-only positions use the first 64/256 head dims for RoPE.
// Causal 24Q/4KV grouped attention: 6 Q per KV, split 6/6/12.
// Gate the attention result with sigmoid of the Q projection gate,
// BF16 round, apply original output column shards, ordered F32
// physical-rank host sum, BF16 rounding and original residual add.
//
// This is a deterministic reference with explicit BF16 boundaries,
// NOT a native HIP model, FlashAttention or full transformer decoder.
#include <algorithm>
#include <array>
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

constexpr int HIDDEN=5120,HEAD=256,ROPE=64;
constexpr std::array<int,3> Q{6,6,12};
constexpr std::array<int,3> KV{1,1,2};
constexpr std::array<const char*,3> ROLES{"xtx0","xtx1","r9700"};
constexpr double EPS=1e-6;
const char* arg(int argc,char** argv,const char* key) {
  for(int i=1;i<argc;++i) if(std::strcmp(argv[i],key)==0) {
    if(i+1>=argc)throw std::runtime_error(std::string("missing value ")+key);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing ")+key);
}
int bounded(int argc,char** argv,const char* key,int max) {
  const char* s=arg(argc,argv,key);
  char* e=nullptr;const long v=std::strtol(s,&e,10);
  if(!*s || !e || *e || v<1 || v>max)
    throw std::runtime_error(std::string("invalid bounded ")+key);
  return static_cast<int>(v);
}
static std::string path(const std::string& dir,const std::string& leaf) {
  return dir+"/"+leaf;
}
template<class T>
static std::vector<T> read_exact(const std::string& file,size_t n) {
  std::ifstream f(file,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=n*sizeof(T))
    throw std::runtime_error("incorrect checkpoint weight extent: "+file);
  f.seekg(0);std::vector<T> v(n);
  f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));
  if(!f)throw std::runtime_error("truncated source: "+file);
  return v;
}
template<class T>
static void write_once(const std::string& file,const std::vector<T>& v) {
  if(std::ifstream(file).good())throw std::runtime_error("refuse overwrite: "+file);
  std::ofstream f(file,std::ios::binary|std::ios::trunc);
  if(!f)throw std::runtime_error("cannot write: "+file);
  f.write(reinterpret_cast<const char*>(v.data()),
          static_cast<std::streamsize>(v.size()*sizeof(T)));
  if(!f)throw std::runtime_error("short written file: "+file);
}
static float fbf16(uint16_t v) {
  const uint32_t bits=static_cast<uint32_t>(v)<<16;
  float value=0;std::memcpy(&value,&bits,sizeof(value));return value;
}
static uint16_t bfrn(float x) {
  uint32_t bits;std::memcpy(&bits,&x,sizeof(bits));
  return static_cast<uint16_t>((bits+0x7fffu+((bits>>16)&1u))>>16);
}
static std::vector<uint16_t> gemv(const std::vector<uint16_t>& weight,
                                  const std::vector<uint16_t>& input,
                                  int rows,int out_width,int in_width) {
  if(weight.size()!=static_cast<size_t>(out_width)*in_width
      || input.size()!=static_cast<size_t>(rows)*in_width)
    throw std::runtime_error("BF16 GEMV rank shape mismatch");
  std::vector<uint16_t> projected(static_cast<size_t>(rows)*out_width);
  for(int row=0;row<rows;++row)
    for(int neuron=0;neuron<out_width;++neuron) {
      double accumulator=0.0;
      for(int k=0;k<in_width;++k)
        accumulator+=static_cast<double>(fbf16(
          weight[static_cast<size_t>(neuron)*in_width+k]))*
          static_cast<double>(fbf16(input[static_cast<size_t>(row)*in_width+k]));
      if(!std::isfinite(accumulator))throw std::runtime_error("nonfinite checkpoint GEMV");
      projected[static_cast<size_t>(row)*out_width+neuron]=bfrn(static_cast<float>(accumulator));
    }
  return projected;
}
static void normalize_rotate(std::vector<uint16_t>& values,
                             int rows,int heads,const std::vector<uint16_t>& norm,
                             int stride,int head_offset) {
  if(norm.size()!=HEAD)throw std::runtime_error("wrong Q/K norm extent");
  std::array<float,HEAD> weight{};
  for(int d=0;d<HEAD;++d)weight[d]=1.0f+fbf16(norm[d]);
  for(int t=0;t<rows;++t)for(int head=0;head<heads;++head) {
    const size_t base=static_cast<size_t>(t)*heads*stride+head*stride+head_offset;
    double square=0.0;
    for(int d=0;d<HEAD;++d) {
      const auto x=fbf16(values[base+d]);
      square+=static_cast<double>(x)*x;
    }
    const double scale=1.0/std::sqrt(square/HEAD+EPS);
    for(int d=0;d<HEAD;++d) {
      const auto x=fbf16(values[base+d]);
      values[base+d]=bfrn(static_cast<float>(x*scale*weight[d]));
    }
    // Standard text-only partial RoPE: all three mRoPE position grids
    // are the same 0..rows-1, so frequency recomposition is identity.
    std::array<float,ROPE> copied{};
    for(int d=0;d<ROPE;++d)copied[d]=fbf16(values[base+d]);
    for(int d=0;d<ROPE/2;++d) {
      const double inv=std::pow(10000000.0,-2.0*d/ROPE);
      const double angle=static_cast<double>(t)*inv;
      const auto cosine=std::cos(angle),sine=std::sin(angle);
      const double a=copied[d],b=copied[d+ROPE/2];
      values[base+d]=bfrn(static_cast<float>(a*cosine-b*sine));
      values[base+d+ROPE/2]=bfrn(static_cast<float>(b*cosine+a*sine));
    }
  }
}
static double sigmoid(double x) {
  return x>=0?1.0/(1.0+std::exp(-x)):std::exp(x)/(1.0+std::exp(x));
}
struct Rank {
  int qh=0,kh=0,rows=0,width=0;
  std::vector<uint16_t> q_gate,k,v,gated;
  std::vector<float> partial;
  double max_attn_score=0;
};
static Rank run_rank(const std::string& packet,const std::string& output,
                     int role,int rows,const std::vector<uint16_t>& input,
                     const std::vector<uint16_t>& qnorm,
                     const std::vector<uint16_t>& knorm) {
  const std::string name=ROLES[role];
  Rank r{};r.qh=Q[role];r.kh=KV[role];r.rows=rows;r.width=r.qh*HEAD;
  const auto qw=read_exact<uint16_t>(
    path(packet,"q-"+name+".bf16"),static_cast<size_t>(r.qh)*HEAD*2*HIDDEN);
  const auto kw=read_exact<uint16_t>(
    path(packet,"k-"+name+".bf16"),static_cast<size_t>(r.kh)*HEAD*HIDDEN);
  const auto vw=read_exact<uint16_t>(
    path(packet,"v-"+name+".bf16"),static_cast<size_t>(r.kh)*HEAD*HIDDEN);
  const auto ow=read_exact<uint16_t>(
    path(packet,"o-"+name+".bf16"),static_cast<size_t>(HIDDEN)*r.width);
  r.q_gate=gemv(qw,input,rows,r.qh*HEAD*2,HIDDEN);
  r.k=gemv(kw,input,rows,r.kh*HEAD,HIDDEN);
  r.v=gemv(vw,input,rows,r.kh*HEAD,HIDDEN);
  write_once(path(output,"q-gate-"+name+".bf16"),r.q_gate);
  write_once(path(output,"k-raw-"+name+".bf16"),r.k);
  write_once(path(output,"v-"+name+".bf16"),r.v);
  normalize_rotate(r.q_gate,rows,r.qh,qnorm,HEAD*2,0);
  normalize_rotate(r.k,rows,r.kh,knorm,HEAD,0);
  write_once(path(output,"q-rope-"+name+".bf16"),r.q_gate);
  write_once(path(output,"k-rope-"+name+".bf16"),r.k);
  r.gated.resize(static_cast<size_t>(rows)*r.width);
  // Attention is causal and grouped by 6 Q per KV. R9700 owns TWO
  // KV groups concurrently; all heads remain independent.
  for(int t=0;t<rows;++t)for(int h=0;h<r.qh;++h) {
    const int kv=h/6;
    const size_t qbase=static_cast<size_t>(t)*r.qh*HEAD*2+h*HEAD*2;
    std::vector<double> scores(t+1);
    double max_score=-std::numeric_limits<double>::infinity();
    for(int prev=0;prev<=t;++prev) {
      const size_t kbase=static_cast<size_t>(prev)*r.kh*HEAD+kv*HEAD;
      double dot=0;
      for(int d=0;d<HEAD;++d)
        dot+=static_cast<double>(fbf16(r.q_gate[qbase+d]))*
             static_cast<double>(fbf16(r.k[kbase+d]));
      scores[prev]=dot/16.0;
      max_score=std::max(max_score,scores[prev]);
      r.max_attn_score=std::max(r.max_attn_score,std::abs(scores[prev]));
    }
    double sum=0;
    for(double& score:scores) {
      score=std::exp(score-max_score);
      sum+=score;
    }
    if(!std::isfinite(sum) || sum<=0)throw std::runtime_error("nonfinite causal softmax");
    for(int d=0;d<HEAD;++d) {
      double attended=0;
      for(int prev=0;prev<=t;++prev) {
        const size_t index=static_cast<size_t>(prev)*r.kh*HEAD+kv*HEAD+d;
        attended+=scores[prev]/sum*fbf16(r.v[index]);
      }
      const float rounded=fbf16(bfrn(static_cast<float>(attended)));
      const float gate=fbf16(r.q_gate[qbase+HEAD+d]);
      const float value=static_cast<float>(rounded*sigmoid(gate));
      r.gated[(static_cast<size_t>(t)*r.qh+h)*HEAD+d]=bfrn(value);
    }
  }
  write_once(path(output,"attended-gated-"+name+".bf16"),r.gated);
  r.partial.resize(static_cast<size_t>(rows)*HIDDEN);
  for(int t=0;t<rows;++t)for(int neuron=0;neuron<HIDDEN;++neuron) {
    double accumulator=0;
    const size_t wbase=static_cast<size_t>(neuron)*r.width;
    const size_t xbase=static_cast<size_t>(t)*r.width;
    for(int k=0;k<r.width;++k)
      accumulator+=static_cast<double>(fbf16(ow[wbase+k]))*
                   static_cast<double>(fbf16(r.gated[xbase+k]));
    if(!std::isfinite(accumulator))throw std::runtime_error("nonfinite output projection");
    r.partial[static_cast<size_t>(t)*HIDDEN+neuron]=static_cast<float>(accumulator);
  }
  write_once(path(output,"partial-"+name+".f32"),r.partial);
  return r;
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const std::string pack=arg(argc,argv,"--pack-dir");
    const std::string output=arg(argc,argv,"--out-dir");
    const auto input=read_exact<uint16_t>(path(pack,"normed_input.bf16"),
                                          static_cast<size_t>(rows)*HIDDEN);
    const auto residual=read_exact<uint16_t>(path(pack,"hidden_input.bf16"),
                                             static_cast<size_t>(rows)*HIDDEN);
    const auto qnorm=read_exact<uint16_t>(path(pack,"q_norm.bf16"),HEAD);
    const auto knorm=read_exact<uint16_t>(path(pack,"k_norm.bf16"),HEAD);
    std::array<Rank,3> ranks{};
    for(int role=0;role<3;++role)
      ranks[role]=run_rank(pack,output,role,rows,input,qnorm,knorm);
    std::vector<float> combined(static_cast<size_t>(rows)*HIDDEN);
    std::vector<uint16_t> final(combined.size());
    size_t exact_residual=0;
    double max_abs=0, max_logits=0;
    for(const auto& r:ranks)max_logits=std::max(max_logits,r.max_attn_score);
    for(size_t i=0;i<combined.size();++i) {
      float sum=ranks[0].partial[i];
      sum+=ranks[1].partial[i];
      sum+=ranks[2].partial[i];
      combined[i]=sum;
      final[i]=bfrn(fbf16(bfrn(sum))+fbf16(residual[i]));
      if(!std::isfinite(sum))throw std::runtime_error("nonfinite TP3 attention output");
      if(final[i]==residual[i])++exact_residual;
      max_abs=std::max(max_abs,static_cast<double>(std::abs(sum)));
    }
    write_once(path(output,"original-attention-reduced.f32"),combined);
    write_once(path(output,"original-attention-post-residual.bf16"),final);
    std::printf("status=ORIGINAL_BF16_FULL_ATTENTION_CPU_LAYER3_REFERENCE_PASS;"
      " rows=%d; q_heads=6/6/12; kv_heads=1/1/2; o_cols=1536/1536/3072;"
      " RoPE_partial_dim=64; mRoPE_text_only=YES; causal_softmax=YES;"
      " q_gate_sigmoid=YES; original_BF16_weights=YES;"
      " original_layer_input_norm=YES; max_abs_output_f32=%.9g;"
      " max_abs_softmax_logit=%.9g; unchanged_residual_bf16=%zu/%zu;"
      " full_model=NO; GPU=NO\n",
      rows,max_abs,max_logits,exact_residual,final.size());
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"VTP22 original BF16 full attention CPU FAILED: %s\n",e.what());
    return 1;
  }
}
