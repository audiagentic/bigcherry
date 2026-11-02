#pragma once
// Test only: writes a small, valid .rad container of routed W4NL64 experts (the format every helper card type runs),
// directory, planes and encoding real and payload zero, so the expert ownership planner can be tested without a model.
#include "rad_encoding.h"
#include "rad_format.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace bc::test {

inline void write_rad(const std::filesystem::path& path, uint32_t layers, uint32_t experts, uint32_t hidden,
                      uint32_t intermediate) {
    RadEncoding enc;
    rad_enc_clear(&enc);
    rad_enc_copy_str(enc.scheme, "w4nl");
    rad_enc_add_plane(&enc, "codes", RAD_U4, 1, 1);
    rad_enc_add_table(&enc, "table", RAD_F16, 1, 16);
    rad_enc_add_plane(&enc, "scale", RAD_F8E4M3, 1, 64);
    if (!rad_enc_valid(&enc)) throw std::logic_error("test .rad: invalid encoding");

    std::vector<RadFileEntry> dir;
    std::vector<RadFilePlane> planes;
    const uint64_t header = sizeof(RadFileHeader);
    const uint64_t n_entries = uint64_t(layers) * experts * 2;
    const uint64_t dir_off = header, plane_off = dir_off + n_entries * sizeof(RadFileEntry);
    const uint64_t enc_off = plane_off + n_entries * 3 * sizeof(RadFilePlane);
    uint64_t data = (enc_off + sizeof(RadEncoding) + RAD_ALIGN_UNIT - 1) / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
    const uint64_t data_off = data;
    for (uint32_t l = 0; l < layers; ++l)
        for (uint32_t e = 0; e < experts; ++e)
            for (uint32_t slot = 0; slot < 2; ++slot) {
                // gate+up [2 x intermediate, hidden]; down [hidden, intermediate]
                const int64_t rows = slot == 0 ? 2 * int64_t(intermediate) : hidden;
                const int64_t cols = slot == 0 ? int64_t(hidden) : intermediate;
                RadFileEntry d{};
                d.enc = 0;
                d.rank = 2;
                d.shape[0] = rows;
                d.shape[1] = cols;
                d.layer = int32_t(l);
                d.expert = int32_t(e);
                d.slot = int32_t(slot);
                d.n_planes = 3;
                d.plane_first = planes.size();
                d.offset = data;
                const uint64_t sizes[3] = {uint64_t(rows * cols / 2), 32, uint64_t(rows * cols / 64)};
                for (uint64_t b : sizes) {
                    planes.push_back({data, b});
                    data += (b + RAD_ALIGN_SUB - 1) / RAD_ALIGN_SUB * RAD_ALIGN_SUB;
                }
                d.bytes = data - d.offset;
                data = (data + RAD_ALIGN_UNIT - 1) / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
                dir.push_back(d);
            }
    RadFileHeader h{};
    h.magic = RAD_MAGIC;
    h.version = RAD_FORMAT_VER;
    h.file_bytes = data;
    h.dir_off = dir_off;
    h.dir_count = dir.size();
    h.plane_off = plane_off;
    h.plane_count = planes.size();
    h.enc_off = enc_off;
    h.enc_count = 1;
    h.data_off = data_off;
    h.data_bytes = data - data_off;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    auto put = [&](uint64_t at, const void* p, size_t n) {
        out.seekp(std::streamoff(at));
        out.write(static_cast<const char*>(p), std::streamsize(n));
    };
    put(0, &h, sizeof h);
    put(dir_off, dir.data(), dir.size() * sizeof(RadFileEntry));
    put(plane_off, planes.data(), planes.size() * sizeof(RadFilePlane));
    put(enc_off, &enc, sizeof enc);
    const char zero = 0;
    put(data - 1, &zero, 1);
    if (!out) throw std::runtime_error("test .rad: cannot write " + path.string());
}

}  // namespace bc::test
