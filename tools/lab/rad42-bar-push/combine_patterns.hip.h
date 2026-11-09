// RAD50 GPU word oracle and real CPU FP16 combine input profiles.
// Included only by bar_dmabuf_ab.hip within its anonymous namespace.
// Device expectations are independent of the production combine function.
__host__ __device__ uint16_t expected_combine_word(uint32_t profile,uint32_t col) {
    switch(profile) {
      case 0:return 0x4200u; // bf16(1)+bf16(2) -> f16(3)
      case 1:return 0x4000u; // helper mask chooses bf16(2)
      case 2:return col==0?0x7e01u:0x7c00u; // overflow fallback sentinel
      default:return 0x7e00u; // canonical NaN (not fallback)
    }
}
__global__ void verify_combine_rows(const uint16_t* output,
                                    uint32_t count,uint32_t generation,
                                    unsigned* errors) {
    const uint32_t id=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t step=gridDim.x*blockDim.x;
    unsigned bad=0;
    for(uint32_t i=id;i<count;i+=step) {
        const uint32_t profile=(i/COMBINE_WIDTH+generation)&3u;
        bad+=output[i]!=expected_combine_word(profile,i%COMBINE_WIDTH);
    }
    if(bad)atomicAdd(errors,bad);
}
struct CombineInput {
    std::array<std::array<std::vector<uint16_t>,2>,4> planes;
    CombineInput(){
        constexpr uint16_t lhs[]={0x3f80u,0x3f80u,0x4780u,0x7fc0u};
        constexpr uint16_t rhs[]={0x4000u,0x4000u,0x3f80u,0x0000u};
        for(unsigned k=0;k<4;++k) {
            planes[k][0].assign(COMBINE_WIDTH,lhs[k]);
            planes[k][1].assign(COMBINE_WIDTH,rhs[k]);
        }
    }
    bool into(uint16_t* destination,uint32_t rows,uint32_t generation)const{
        for(uint32_t row=0;row<rows;++row) {
            const uint32_t profile=(row+generation)&3u;
            const uint16_t* src[]={planes[profile][0].data(),planes[profile][1].data()};
            uint16_t* out=destination+row*COMBINE_WIDTH;
            const bool over=combine_row_f16(src,2,profile==1?uint8_t{2}:uint8_t{3},
                                            out,COMBINE_WIDTH);
            if(over!=(profile==2))return false;
            if(over)out[0]=r3::moe_shm::kCombineFallback;
        }
        return true;
    }
};
