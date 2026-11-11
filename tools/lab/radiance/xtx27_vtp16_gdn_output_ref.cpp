// VTP16: CPU, original BF16 Qwen3.8 GDN out projection across three
// physical rank column shards, ordered F32 host reduction and BF16
// residual add. Independently computes the unsharded full BF16 GEMV.
// NO GPU, no engine, no output tokens, not a performance benchmark.
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

static const char* arg(int argc,char** argv,const char* key) {
  for(int i=1;i<argc;++i)if(std::strcmp(argv[i],key)==0) {
    if(i+1==argc)throw std::runtime_error(std::string("missing value ")+key);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing argument ")+key);
}
static int limited(int argc,char** argv,const char* key,int limit) {
  const char* val=arg(argc,argv,key);
  char* end=nullptr;const long n=std::strtol(val,&end,10);
  if(!*val || !end || *end || n<1 || n>limit)
    throw std::runtime_error(std::string("invalid positive value ")+key);
  return static_cast<int>(n);
}
template<class T> static std::vector<T> read_file(const std::string& path,size_t count) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=count*sizeof(T))
    throw std::runtime_error("unexpected byte extent for "+path);
  f.seekg(0);std::vector<T> data(count);
  f.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(count*sizeof(T)));
  if(!f)throw std::runtime_error("short input "+path);
  return data;
}
template<class T>static void write_file(const std::string& path,const std::vector<T>& data) {
  std::ifstream existing(path);
  if(existing.good())throw std::runtime_error("refusing to overwrite "+path);
  std::ofstream f(path,std::ios::binary|std::ios::trunc);
  if(!f)throw std::runtime_error("unable to create "+path);
  f.write(reinterpret_cast<const char*>(data.data()),
          static_cast<std::streamsize>(data.size()*sizeof(T)));
  if(!f)throw std::runtime_error("failed writing "+path);
}
static float f32(uint16_t v) {
  const uint32_t bits=static_cast<uint32_t>(v)<<16;
  float f=0.0f;std::memcpy(&f,&bits,sizeof(f));return f;
}
static uint16_t rn(float f) {
  uint32_t bits=0;std::memcpy(&bits,&f,sizeof(bits));
  return static_cast<uint16_t>((bits+0x7FFFu+((bits>>16)&1u))>>16);
}
static std::string concat(const std::string& path,const std::string& tail) {
  return path+"/"+tail;
}
int main(int argc,char** argv) {
  try {
    const int M=limited(argc,argv,"--m",8);
    const int N=limited(argc,argv,"--n",5120);
    const int K=limited(argc,argv,"--k",6144);
    const int A=limited(argc,argv,"--k0",6144),B=limited(argc,argv,"--k1",6144);
    const int C=limited(argc,argv,"--k2",6144);
    if(A+B+C!=K)throw std::runtime_error("column rank widths not complete");
    const std::array<int,3> widths{A,B,C};
    const std::array<const char*,3> roles{"xtx0","xtx1","r9700"};
    const std::string dir=arg(argc,argv,"--package-dir");
    const std::string output_dir=arg(argc,argv,"--output-dir");
    const auto full_weights=read_file<uint16_t>(concat(dir,"out-original.bf16"),
                                               static_cast<size_t>(N)*K);
    const auto input_full=read_file<uint16_t>(concat(dir,"gdn-original.bf16"),
                                             static_cast<size_t>(M)*K);
    const auto residual=read_file<uint16_t>(arg(argc,argv,"--residual"),
                                            static_cast<size_t>(M)*N);
    std::array<std::vector<uint16_t>,3> wparts{};
    std::array<std::vector<uint16_t>,3> inputs{};
    for(int r=0;r<3;++r) {
      wparts[r]=read_file<uint16_t>(concat(dir,std::string("out-")+roles[r]+".bf16"),
                                    static_cast<size_t>(N)*widths[r]);
      inputs[r]=read_file<uint16_t>(concat(dir,std::string("gdn-")+roles[r]+".bf16"),
                                    static_cast<size_t>(M)*widths[r]);
    }
    // Fail closed if either model-owned weight partition or the BF16
    // GDN activation input is inconsistent with the original tensors.
    int offset=0;
    for(int r=0;r<3;++r) {
      const int kr=widths[r];
      for(int n=0;n<N;++n)
        for(int k=0;k<kr;++k)
          if(full_weights[static_cast<size_t>(n)*K+offset+k] !=
              wparts[r][static_cast<size_t>(n)*kr+k])
            throw std::runtime_error("rank output column shards diverge from original weights");
      for(int m=0;m<M;++m)
        for(int k=0;k<kr;++k)
          if(input_full[static_cast<size_t>(m)*K+offset+k] !=
              inputs[r][static_cast<size_t>(m)*kr+k])
            throw std::runtime_error("rank GDN output input differs from global BF16 view");
      offset+=kr;
    }
    const size_t count=static_cast<size_t>(M)*N;
    std::vector<float> full_output(count),tp3_output(count);
    std::array<std::vector<float>,3> partial{};
    for(auto& v:partial)v.resize(count);
    std::vector<uint16_t> final_full(count),final_tp3(count);
    double max_projection_abs=0.0,max_relative=0.0,max_post_residual_abs=0.0;
    size_t exact_bf16=0,exact_residual_bf16=0;
    for(int m=0;m<M;++m) {
      for(int n=0;n<N;++n) {
        double unsharded=0.0;
        for(int k=0;k<K;++k)
          unsharded+=static_cast<double>(f32(
            full_weights[static_cast<size_t>(n)*K+k]))*
            static_cast<double>(f32(input_full[static_cast<size_t>(m)*K+k]));
        if(!std::isfinite(unsharded) || std::abs(unsharded)>std::numeric_limits<float>::max())
          throw std::runtime_error("nonfinite original unsharded output projection");
        full_output[static_cast<size_t>(m)*N+n]=static_cast<float>(unsharded);
        for(int r=0;r<3;++r) {
          const int kr=widths[r];
          double d=0.0;
          for(int k=0;k<kr;++k)
            d+=static_cast<double>(f32(wparts[r][static_cast<size_t>(n)*kr+k]))*
               static_cast<double>(f32(inputs[r][static_cast<size_t>(m)*kr+k]));
          partial[r][static_cast<size_t>(m)*N+n]=static_cast<float>(d);
        }
        const size_t pos=static_cast<size_t>(m)*N+n;
        // A defined and reproducible three-physical partial order.
        float reduced=partial[0][pos];
        reduced+=partial[1][pos];
        reduced+=partial[2][pos];
        tp3_output[pos]=reduced;
        const double err=std::abs(static_cast<double>(reduced)-full_output[pos]);
        max_projection_abs=std::max(max_projection_abs,err);
        max_relative=std::max(max_relative,err/(1.0+std::abs(full_output[pos])));
        if(err>0.0005 + 0.0001*std::abs(full_output[pos]))
          throw std::runtime_error("rank F32 reduced projection diverged from original BF16");
        const auto global_bf16=rn(full_output[pos]);
        const auto physical_bf16=rn(reduced);
        if(global_bf16==physical_bf16)++exact_bf16;
        const float res=f32(residual[pos]);
        // Engine has a BF16 w.h buffer after output-GEMM/all-reduce.
        // Compare the BF16-rounded collective then BF16 residual add.
        final_full[pos]=rn(f32(global_bf16)+res);
        final_tp3[pos]=rn(f32(physical_bf16)+res);
        if(final_full[pos]==final_tp3[pos])++exact_residual_bf16;
        max_post_residual_abs=std::max(max_post_residual_abs,
                                     static_cast<double>(std::abs(
                                       f32(final_full[pos])-f32(final_tp3[pos]))));
      }
    }
    write_file(concat(output_dir,"unsharded-original.f32"),full_output);
    write_file(concat(output_dir,"ordered-three-rank.f32"),tp3_output);
    write_file(concat(output_dir,"unsharded-residual.bf16"),final_full);
    write_file(concat(output_dir,"ordered-three-rank-residual.bf16"),final_tp3);
    for(int r=0;r<3;++r)
      write_file(concat(output_dir,std::string("partial-")+roles[r]+".f32"),partial[r]);
    std::printf("status=PASS; m=%d; n=%d; k=%d; splits=%d/%d/%d;"
                " max_projection_abs=%.9g; max_relative=%.9g;"
                " bf16_proj_exact=%zu/%zu; bf16_residual_exact=%zu/%zu;"
                " max_post_residual_abs=%.9g\n",
                M,N,K,A,B,C,max_projection_abs,max_relative,exact_bf16,count,
                exact_residual_bf16,count,max_post_residual_abs);
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"VTP16 original GDN output CPU oracle FAILED: %s\n",e.what());
    return 1;
  }
}
