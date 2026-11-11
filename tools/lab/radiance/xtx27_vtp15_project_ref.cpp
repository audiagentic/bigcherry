// VTP15: independent double-accumulation oracle for original BF16
// Qwen3.8 GDN projections. Card-free. A single original checkpoint
// rank-owned [N, K] BF16 matrix times [M,K] normalized BF16 input.
// Emits BF16-rounded output and pre-round FP32, both row-major.
// The tiny-shape mode supports hermetic correctness tests.
#include <algorithm>
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

static const char* arg(int argc,char** argv,const char* key) {
  for(int i=1;i<argc;++i)if(strcmp(key,argv[i])==0) {
    if(i+1==argc)throw std::runtime_error(std::string("missing ")+key);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("required ")+key);
}
static int bound(int argc,char** argv,const char* key,int hi) {
  const char* s=arg(argc,argv,key);char* end=nullptr;
  const long n=strtol(s,&end,10);
  if(!s[0] || !end || *end || n<=0 || n>hi)
    throw std::runtime_error(std::string("invalid ")+key);
  return static_cast<int>(n);
}
template<typename T> static std::vector<T> read_exact(const char* path,size_t count) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=sizeof(T)*count)
    throw std::runtime_error(std::string("wrong exact file size ")+path);
  f.seekg(0);std::vector<T> out(count);
  f.read(reinterpret_cast<char*>(out.data()),static_cast<std::streamsize>(out.size()*sizeof(T)));
  if(!f)throw std::runtime_error("truncated binary");
  return out;
}
template<typename T>static void write_exact(const char* path,const std::vector<T>& v) {
  std::ifstream prior(path);if(prior.good())throw std::runtime_error("output already exists");
  std::ofstream f(path,std::ios::binary|std::ios::trunc);
  if(!f)throw std::runtime_error(std::string("cannot write ")+path);
  f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));
  if(!f)throw std::runtime_error("write failed");
}
static inline float decode(uint16_t u) {
  uint32_t b=static_cast<uint32_t>(u)<<16;float out;
  memcpy(&out,&b,4);return out;
}
static inline uint16_t bf16rne(float f) {
  uint32_t u;memcpy(&u,&f,4);
  return static_cast<uint16_t>((u+0x7fffu+((u>>16)&1u))>>16);
}
int main(int argc,char** argv) {
  try {
    const int m=bound(argc,argv,"--m",8),n=bound(argc,argv,"--n",17408);
    const int k=bound(argc,argv,"--k",5120);
    const auto input=read_exact<uint16_t>(arg(argc,argv,"--input"),static_cast<size_t>(m)*k);
    const auto weights=read_exact<uint16_t>(arg(argc,argv,"--weights"),static_cast<size_t>(n)*k);
    std::vector<uint16_t> outputs(static_cast<size_t>(m)*n);
    std::vector<float> fp32(outputs.size());
    double maxabs=0.0;
    for(int row=0;row<m;++row) {
      for(int out=0;out<n;++out) {
        double acc=0.0;
        const auto* w=weights.data()+static_cast<size_t>(out)*k;
        const auto* x=input.data()+static_cast<size_t>(row)*k;
        for(int col=0;col<k;++col)
          acc+=static_cast<double>(decode(w[col]))*static_cast<double>(decode(x[col]));
        if(!std::isfinite(acc) || fabs(acc)>std::numeric_limits<float>::max())
          throw std::runtime_error("nonfinite projected activation");
        const float f=static_cast<float>(acc);
        fp32[static_cast<size_t>(row)*n+out]=f;
        outputs[static_cast<size_t>(row)*n+out]=bf16rne(f);
        maxabs=std::max(maxabs,std::abs(static_cast<double>(f)));
      }
    }
    write_exact(arg(argc,argv,"--out-bf16"),outputs);
    write_exact(arg(argc,argv,"--out-f32"),fp32);
    printf("VTP15 original BF16 projection CPU PASS m=%d n=%d k=%d "
           "max_abs_f32=%.8g output_elems=%zu\n",m,n,k,maxabs,outputs.size());
    return 0;
  } catch(const std::exception& e) {
    fprintf(stderr,"VTP15 BF16 GEMV reference FAILED: %s\n",e.what());return 1;
  }
}
