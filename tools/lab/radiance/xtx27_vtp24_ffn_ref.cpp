// VTP24: standalone original-Qwen-BF16 FFN model-layer0 CPU oracle.
// Consumes the REAL original checkpoint post-attention RMSNorm,
// BF16 gate/up/down 5120/4864/7424 physical shards, and original
// GDN-after-residual hidden input. Calculates a complete layer-0
// post-FFN residual for the NEXT layer's input.
// Reference rounding: gate/up FP32(dot BF16*BF16) -> BF16, SiLU*
// BF16 up -> BF16, down F64 BF16*BF16 -> F32, ordered F32
// reduce, round to BF16 then add incoming BF16 residual -> BF16.
// Not yet framework bit-exact or any GPU/native-model inference.
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

constexpr int H=5120;
constexpr std::array<int,3> WIDTHS{5120,4864,7424};
constexpr std::array<const char*,3> ROLES{"xtx0","xtx1","r9700"};

const char* arg(int argc,char** argv,const char* name) {
  for(int i=1;i<argc;++i) if(std::strcmp(argv[i],name)==0) {
    if(i+1>=argc)throw std::runtime_error(std::string("missing ")+name);
    return argv[i+1];
  }
  throw std::runtime_error(std::string("missing ")+name);
}
int bounded(int argc,char** argv,const char* name,int maxval) {
  const char* x=arg(argc,argv,name);char* end=nullptr;
  const long v=std::strtol(x,&end,10);
  if(!*x || !end || *end || v<1 || v>maxval)
    throw std::runtime_error(std::string("invalid ")+name);
  return static_cast<int>(v);
}
std::string path(const std::string& directory,const std::string& filename) {
  return directory+"/"+filename;
}
template<class T>
std::vector<T> read_exact(const std::string& filename,size_t count) {
  std::ifstream f(filename,std::ios::binary|std::ios::ate);
  if(!f || f.tellg()<0 || static_cast<size_t>(f.tellg())!=count*sizeof(T))
    throw std::runtime_error("incorrect original BF16 weight/input extent: "+filename);
  f.seekg(0);std::vector<T> result(count);
  f.read(reinterpret_cast<char*>(result.data()),
         static_cast<std::streamsize>(result.size()*sizeof(T)));
  if(!f)throw std::runtime_error("truncated original BF16 source: "+filename);
  return result;
}
template<class T>
void write_once(const std::string& filename,const std::vector<T>& vals) {
  if(std::ifstream(filename).good())
    throw std::runtime_error("original BF16 reference overwrite refused: "+filename);
  std::ofstream file(filename,std::ios::binary|std::ios::trunc);
  if(!file)throw std::runtime_error("could not open CPU result: "+filename);
  file.write(reinterpret_cast<const char*>(vals.data()),
             static_cast<std::streamsize>(vals.size()*sizeof(T)));
  if(!file)throw std::runtime_error("short CPU output: "+filename);
}
float fb16(uint16_t raw) {
  uint32_t bits=static_cast<uint32_t>(raw)<<16;
  float value=0;
  std::memcpy(&value,&bits,sizeof(value));
  return value;
}
uint16_t bfround(float value) {
  uint32_t bits=0;std::memcpy(&bits,&value,sizeof(bits));
  if(!std::isfinite(value)) throw std::runtime_error("nonfinite original BF16 FFN result");
  return static_cast<uint16_t>((bits+0x7fffu+((bits>>16)&1u))>>16);
}
std::vector<uint16_t> project(const std::vector<uint16_t>& weights,
                              const std::vector<uint16_t>& input,
                              int rows,int width,int k) {
  if(weights.size()!=static_cast<size_t>(width)*k ||
     input.size()!=static_cast<size_t>(rows)*k)
    throw std::runtime_error("original FFN projection dimensions invalid");
  std::vector<uint16_t> result(static_cast<size_t>(rows)*width);
  for(int row=0;row<rows;++row)
    for(int neuron=0;neuron<width;++neuron) {
      const size_t wb=static_cast<size_t>(neuron)*k;
      const size_t xb=static_cast<size_t>(row)*k;
      double accumulator=0.0;
      for(int dim=0;dim<k;++dim)
        accumulator+=static_cast<double>(fb16(weights[wb+dim]))*
                     static_cast<double>(fb16(input[xb+dim]));
      result[static_cast<size_t>(row)*width+neuron]=
          bfround(static_cast<float>(accumulator));
    }
  return result;
}
std::vector<uint16_t> swiglu(const std::vector<uint16_t>& gate,
                             const std::vector<uint16_t>& up) {
  if(gate.size()!=up.size())throw std::runtime_error("FFN gate/up mismatch");
  std::vector<uint16_t> mid(gate.size());
  for(size_t i=0;i<gate.size();++i) {
    const double g=fb16(gate[i]),u=fb16(up[i]);
    const double sigmoid=g>=0?1.0/(1.0+std::exp(-g)):std::exp(g)/(1.0+std::exp(g));
    mid[i]=bfround(static_cast<float>((g*sigmoid)*u));
  }
  return mid;
}
std::vector<float> down_project(const std::vector<uint16_t>& weight,
                                const std::vector<uint16_t>& input,
                                int rows,int width) {
  if(weight.size()!=static_cast<size_t>(H)*width ||
     input.size()!=static_cast<size_t>(rows)*width)
    throw std::runtime_error("original down projection shape mismatch");
  std::vector<float> out(static_cast<size_t>(rows)*H);
  for(int row=0;row<rows;++row)
    for(int neuron=0;neuron<H;++neuron) {
      const auto wb=static_cast<size_t>(neuron)*width;
      const auto xb=static_cast<size_t>(row)*width;
      double accumulator=0.0;
      for(int d=0;d<width;++d)
        accumulator+=static_cast<double>(fb16(weight[wb+d]))*
                     static_cast<double>(fb16(input[xb+d]));
      if(!std::isfinite(accumulator))
        throw std::runtime_error("nonfinite BF16 FFN down source");
      out[static_cast<size_t>(row)*H+neuron]=static_cast<float>(accumulator);
    }
  return out;
}
int main(int argc,char** argv) {
  try {
    const int rows=bounded(argc,argv,"--rows",8);
    const std::string packet=arg(argc,argv,"--packet-dir");
    const std::string dest=arg(argc,argv,"--out-dir");
    const auto normed=read_exact<uint16_t>(
        path(packet,"normed.bf16"),static_cast<size_t>(rows)*H);
    const auto residual=read_exact<uint16_t>(
        path(packet,"input.bf16"),static_cast<size_t>(rows)*H);
    std::array<std::vector<float>,3> partials;
    for(int role=0;role<3;++role) {
      const std::string tag=ROLES[role];
      const int width=WIDTHS[role];
      const auto wgate=read_exact<uint16_t>(
          path(packet,"gate-"+tag+".bf16"),static_cast<size_t>(width)*H);
      const auto wup=read_exact<uint16_t>(
          path(packet,"up-"+tag+".bf16"),static_cast<size_t>(width)*H);
      const auto wdown=read_exact<uint16_t>(
          path(packet,"down-"+tag+".bf16"),static_cast<size_t>(width)*H);
      const auto gate=project(wgate,normed,rows,width,H);
      const auto up=project(wup,normed,rows,width,H);
      const auto mid=swiglu(gate,up);
      const auto output=down_project(wdown,mid,rows,width);
      write_once(path(dest,"gate-"+tag+".bf16"),gate);
      write_once(path(dest,"up-"+tag+".bf16"),up);
      write_once(path(dest,"mid-"+tag+".bf16"),mid);
      write_once(path(dest,"partial-"+tag+".f32"),output);
      partials[role]=output;
    }
    const size_t n=static_cast<size_t>(rows)*H;
    std::vector<float> reduced(n);
    std::vector<uint16_t> post(n);
    double maxabs=0;
    for(size_t i=0;i<n;++i) {
      float value=partials[0][i];
      value+=partials[1][i];
      value+=partials[2][i];
      reduced[i]=value;
      const float b=fb16(bfround(value));
      post[i]=bfround(b+fb16(residual[i]));
      maxabs=std::max(maxabs,static_cast<double>(std::abs(value)));
    }
    write_once(path(dest,"original-ffn-reduced.f32"),reduced);
    write_once(path(dest,"original-layer0-post-ffn.bf16"),post);
    std::printf(
        "status=ORIGINAL_BF16_QWEN3_8_LAYER0_FULL_FFN_CPU_ORACLE_PASS;"
        " M=%d; BF16_gate_up_down=YES; physical_FFN_5120_4864_7424=YES;"
        " original_post_attention_layernorm=YES;"
        " original_GDN_output_residual_input=YES;"
        " ordered_F32_reduce=YES; BF16_residual=YES;"
        " max_abs_ffn_output=%.9g; next_layer_input_produced=YES;"
        " GPU=NO; model64=NO\n",rows,maxabs);
    return 0;
  }catch(const std::exception& e) {
    std::fprintf(stderr,"VTP24 original BF16 FFN CPU oracle FAILED: %s\n",e.what());
    return 1;
  }
}
