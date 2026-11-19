#include "rank_demand.h"

#include <stdexcept>

namespace bc::serving {
namespace {

constexpr int64_t kMaxMeasuredInteger = (int64_t{1} << 53) - 1;

void exact_fields(const nlohmann::json& value, std::initializer_list<const char*> fields,
                  const std::string& where) {
    if (!value.is_object() || value.size() != fields.size())
        throw std::runtime_error(where + ": missing or unknown members");
    for (const char* field : fields)
        if (!value.contains(field)) throw std::runtime_error(where + ": missing " + field);
}

int64_t measured_integer(const nlohmann::json& value, const std::string& where, bool positive) {
    if (!value.is_number_integer()) throw std::runtime_error(where + ": exact integer required");
    const int64_t parsed = value.get<int64_t>();
    if (parsed < 0 || parsed > kMaxMeasuredInteger || (positive && parsed == 0))
        throw std::runtime_error(where + ": out of measured wire range");
    return parsed;
}

}  // namespace

EngineRankDemand decode_engine_rank_demand(const nlohmann::json& record,
                                           const std::vector<std::string>& ordered_bdfs,
                                           const std::string& where) {
    exact_fields(record, {"schema_version", "kind", "gpu", "resident_other_bytes"}, where);
    if (!record.at("schema_version").is_number_integer() || record.at("schema_version") != 1 ||
        !record.at("kind").is_string() || record.at("kind") != "radiance_p1_rank_demand" ||
        !record.at("gpu").is_array() || record.at("gpu").size() != ordered_bdfs.size())
        throw std::runtime_error(where + ": invalid source identity or rank count");

    EngineRankDemand out;
    out.resident_other_bytes = measured_integer(record.at("resident_other_bytes"),
                                                where + ".resident_other_bytes", false);
    out.gpu.reserve(ordered_bdfs.size());
    for (size_t i = 0; i < ordered_bdfs.size(); ++i) {
        const auto& row = record.at("gpu").at(i);
        const std::string row_where = where + ".gpu[" + std::to_string(i) + "]";
        exact_fields(row, {"bdf", "required_bytes", "fixed_bytes", "claimable_bytes", "evidence"}, row_where);
        if (!row.at("bdf").is_string() || row.at("bdf").get<std::string>().empty() ||
            row.at("bdf").get<std::string>() != ordered_bdfs[i])
            throw std::runtime_error(row_where + ".bdf: differs from selected ordered engine devices");
        if (!row.at("evidence").is_string() || row.at("evidence").get<std::string>().empty() ||
            row.at("evidence").get<std::string>().find("rad0011") == std::string::npos)
            throw std::runtime_error(row_where + ".evidence: native rad0011 source required");
        EngineRankGpuDemand gpu;
        gpu.bdf = ordered_bdfs[i];
        gpu.required_bytes = measured_integer(row.at("required_bytes"), row_where + ".required_bytes", true);
        gpu.fixed_bytes = measured_integer(row.at("fixed_bytes"), row_where + ".fixed_bytes", true);
        gpu.claimable_bytes = measured_integer(row.at("claimable_bytes"), row_where + ".claimable_bytes", true);
        gpu.evidence = row.at("evidence").get<std::string>();
        if (gpu.fixed_bytes > gpu.required_bytes || gpu.required_bytes > gpu.claimable_bytes)
            throw std::runtime_error(row_where + ": fixed/required demand exceeds measured claimable room");
        out.gpu.push_back(std::move(gpu));
    }
    return out;
}

}  // namespace bc::serving
