#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace bc::serving {

/* The calibration store (RSC01 "calibration cache", RSC03): parameters of the resource plan's derivation rules, each
 * fitted to named measurements. Seeds are compiled in from calibration.toml; a cache file RHP12 writes is layered on
 * top. Lookup picks the most specific matching entry (each of model, layout, objective it names counts one), the cache
 * winning at equal specificity; two entries of one source at the same specificity are an error. A missing entry is an
 * error: there is no built-in default. */
struct CalibrationMatch {
    std::string model, layout, objective;
};

struct Calibrated {
    nlohmann::json value;
    std::string source;     // "seed" or "cache"
    std::string seeded;     // the measurement's date
    std::string evidence;
    nlohmann::json match;
    // The provenance source explain shows: "calibrated (seed: <date>) <key>: <evidence>".
    std::string describe(const std::string& key) const;
};

class Calibration {
public:
    // `seed_toml` is calibration.toml's text; `cache` an optional refit file in the same format.
    static Calibration load(const std::string& seed_toml, const std::filesystem::path& cache = {});
    // The compiled-in seeds plus `cache`.
    static Calibration standard(const std::filesystem::path& cache = {});
    Calibrated get(const std::string& key, const CalibrationMatch& at) const;
    double number(const std::string& key, const CalibrationMatch& at) const;

private:
    struct Entry {
        std::string key, seeded, evidence, source;
        nlohmann::json match, value;
    };
    std::vector<Entry> entries_;
    static void parse(const std::string& text, const std::string& where, const std::string& source,
                      std::vector<Entry>& into);
};

}  // namespace bc::serving
