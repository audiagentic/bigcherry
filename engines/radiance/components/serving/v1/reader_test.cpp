// bc_v1_test: the shared v1 reader against the committed examples, every required error test of errors.md §6 and the
// RSC08 slice 2 gates, and the typed admission check against the shared admission vectors.
#include "admission.h"
#include "jcs.h"
#include "manifest.h"
#include "serving_codec.h"
#include "strict_json.h"

#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>

#ifndef BC_CONTRACT_DIR
#error "BC_CONTRACT_DIR must name docs/design/serving-manifest-v1"
#endif

namespace {
using namespace bc::serving::v1;
using Json = nlohmann::json;
using C = ConfigErrorCode;
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::string read(const std::string& name) {
    std::ifstream in(std::string(BC_CONTRACT_DIR) + "/" + name, std::ios::binary);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

Json rehashed(Json m) {
    std::string d, why;
    m["metadata"].erase("content_digest");
    bc::serving::manifest_content_digest(m, &d, &why);
    m["metadata"]["content_digest"] = d;
    return m;
}

/* The code (and so the status), a needle in the message, and when given the path's start: checked separately, so a
 * message that merely mentions the expected path cannot stand in for the path. */
template <class T>
void expect(const Result<T>& r, C code, const std::string& needle, const std::string& what, const std::string& path = "") {
    if (r) {
        check(false, what + ": loaded");
        return;
    }
    const ConfigError& e = r.error();
    check(e.code == code && e.status == status_of(code) && e.message.find(needle) != std::string::npos &&
              e.path.rfind(path, 0) == 0,
          what + ": " + std::string(machine_name(e.code)) + " at '" + e.path + "': " + e.message);
}

Result<Manifest> from(const Json& j) { return read_manifest(j.dump(), "test.json"); }
Result<Manifest> from_text(const std::string& t) { return read_manifest(t, "test.json"); }

std::string replace_once(std::string s, const std::string& from, const std::string& to, bool* ok) {
    const size_t at = s.find(from);
    if (at == std::string::npos) {
        *ok = false;
        return s;
    }
    return s.replace(at, from.size(), to);
}
}  // namespace

int main() {
    const std::string prod_text = read("example.flashnext-4gpu.json"), lab_text = read("example.lab-experiment.json");
    const Json prod = Json::parse(prod_text), lab = Json::parse(lab_text);

    // ---- the examples load, and the views are what the documents say ----
    Result<Manifest> p = from_text(prod_text), l = from_text(lab_text);
    check(p.ok(), "production example loads: " + (p ? std::string() : render_human("bc_v1_test", p.error())));
    check(l.ok(), "lab example loads: " + (l ? std::string() : render_human("bc_v1_test", l.error())));
    if (failures) return 1;
    const EngineView ev = engine_view(p.value());
    check(ev.kv_backing_floor_ppm == 600000, "engine view: the selected backing floor");
    check(ev.engine.scheduler.tp == prod["spec"]["processes"]["engine"]["scheduler"]["tp"].get<int64_t>(), "engine view: tp");
    const auto& phelpers = prod["spec"]["processes"]["helpers"];
    for (size_t i = 0; i < phelpers.size(); ++i) {
        Result<HelperView> hv = helper_view(p.value(), "helper" + std::to_string(i));
        check(hv.ok(), "helper view " + std::to_string(i));
        if (!hv) continue;
        // every effective group leaf equals the role patch where present, else helper_common
        const Json base = prod["spec"]["processes"]["helper_common"];
        const Json patch = phelpers[i].value("settings", Json::object());
        const int64_t h2 = patch.contains("gemv") && patch["gemv"].contains("h2") ? patch["gemv"]["h2"].get<int64_t>()
                                                                                 : base["gemv"]["h2"].get<int64_t>();
        check(hv.value().settings.gemv.h2 == h2, "helper" + std::to_string(i) + " effective gemv.h2");
    }
    {   // lab: experiment common gemv.h2 overrides every helper; the helper2 patch sets its mode
        const Json shape = lab["spec"]["experiment"]["shape"];
        for (size_t i = 0; i < lab["spec"]["processes"]["helpers"].size(); ++i) {
            Result<HelperView> hv = helper_view(l.value(), "helper" + std::to_string(i));
            if (!hv) { check(false, "lab helper view"); continue; }
            if (shape.contains("helper_common") && shape["helper_common"].contains("gemv") &&
                shape["helper_common"]["gemv"].contains("h2"))
                check(hv.value().settings.gemv.h2 == shape["helper_common"]["gemv"]["h2"].get<int64_t>(),
                      "lab helper" + std::to_string(i) + ": experiment common h2");
            for (const Json& rp : shape.value("helpers", Json::array()))
                if (rp["role"] == "helper" + std::to_string(i) && rp["settings"].contains("gemv") &&
                    rp["settings"]["gemv"].contains("mode"))
                    check(hv.value().settings.gemv.mode == rp["settings"]["gemv"]["mode"].get<int64_t>(),
                          "lab helper" + std::to_string(i) + ": experiment role mode");
        }
    }
    expect(helper_view(p.value(), "helper9"), C::unresolved_reference, "helper9", "unknown helper role view");
    check(!check_engine_plan(p.value().spec.plan.engine_memory.expected, p.value().spec.plan.engine_memory.expected, 600000),
          "the production plan admits itself");

    // ---- errors.md §6 ----
    {
        Json j = prod;
        j["spec"]["processes"]["engine"]["execution"]["bogus"] = 1;
        expect(from(rehashed(j)), C::unknown_field, "unknown member", "unknown engine.execution key",
               "spec.processes.engine.execution.bogus");
        check(exit_code_of(from(rehashed(j)).error()) == 2 && status_of(C::unknown_field) == -9, "unknown field: -9, exit 2");
    }
    {
        Json j = prod;
        j["spec"]["experiment"] = Json::object();
        expect(from(rehashed(j)), C::experiment_forbidden,
               "spec.experiment is forbidden when spec.intent.service.profile=production", "production plus experiment {}");
        Json k = lab;
        k["spec"]["intent"]["service"]["profile"] = "production";
        expect(from(rehashed(k)), C::experiment_forbidden, "forbidden", "production plus a full experiment");
    }
    {
        Json j = prod;
        j["spec"]["intent"]["model"] = "/models/other.rad";
        expect(from(j), C::digest_mismatch, "content digest mismatch", "bad hash");
        check(status_of(C::digest_mismatch) == -14 && exit_code_of(from(j).error()) == 2, "bad hash: -14, exit 2");
        j["spec"]["plan"]["owner_map"]["helpers"][0]["count"] = 999;   // stale and semantically wrong: digest first
        expect(from(j), C::digest_mismatch, "content digest mismatch", "stale manifest with bad semantics");
    }
    {
        Json j = prod;
        auto& experts = j["spec"]["plan"]["owner_map"]["helpers"][0]["experts"];
        experts[0][1] = experts[0][1].get<int64_t>() + 1000;
        expect(from(rehashed(j)), C::ownership_mismatch, "owned_sha256", "rehashed invalid owner map");
        Json k = prod;
        k["spec"]["plan"]["owner_map"]["helpers"][0]["count"] = 0;
        expect(from(rehashed(k)), C::ownership_mismatch, "count", "rehashed owner count");
    }
    {
        Json j = lab;
        j["spec"]["experiment"]["shape"]["helpers"][0]["role"] = "helper9";
        expect(from(rehashed(j)), C::unresolved_reference, "helper9", "a lab patch for a missing helper");
        check(status_of(C::unresolved_reference) == -17, "unresolved reference: -17");
        Json k = prod;
        k["spec"]["processes"]["helpers"][0]["gpu"] = "0000:ff:00.0";
        expect(from(rehashed(k)), C::unresolved_reference, "0000:ff:00.0", "a nonexistent GPU BDF");
    }
    {
        Result<Manifest> r = load_manifest("/nonexistent/manifest.json");
        expect(r, C::io_failure, "cannot read", "missing manifest file");
        check(!r && exit_code_of(r.error()) == 3 && r.error().status == -13 && r.error().source_file, "missing file: -13, exit 3");
    }

    // ---- RSC08 slice 2 gates: duplicates at parse time, digest form, semantic codes after a rehash, size, UTF-8 ----
    const std::string compact = prod.dump();
    bool ok = true;
    expect(from_text(replace_once(compact, "\"kind\":\"ServingManifest\"", "\"kind\":\"ServingManifest\",\"kind\":\"ServingManifest\"", &ok)),
           C::duplicate_field, "\"kind\"", "duplicate at the top level");
    expect(from_text(replace_once(compact, "\"intent\":{", "\"intent\":{\"layout\":\"x\",", &ok)), C::duplicate_field,
           "spec.intent", "duplicate in a nested object");
    expect(from_text(replace_once(compact, "\"calibration\":[{", "\"calibration\":[{\"key\":\"checkpoint_share\",", &ok)),
           C::duplicate_field, "spec.calibration[0]", "duplicate inside an array element");
    expect(from_text(replace_once(compact, "\"apiVersion\":\"bigcherry/v1\"",
                                  "\"apiVersion\":\"bigcherry/v1\",\"\\u0061piVersion\":\"bigcherry/v1\"", &ok)),
           C::duplicate_field, "apiVersion", "escape-equivalent duplicate names");
    check(ok, "every duplicate fixture applied");
    {
        Json j = prod;
        j["metadata"]["content_digest"] = 5;
        expect(from(j), C::digest_mismatch, "must be a string", "content_digest of the wrong type");
        j["metadata"]["content_digest"] = "sha256:" + std::string(64, 'A');
        expect(from(j), C::digest_mismatch, "64 lowercase hex", "content_digest of the wrong format");
        j["metadata"].erase("content_digest");
        expect(from(j), C::digest_mismatch, "absent", "content_digest absent");
    }
    {
        Json j = prod;
        j["spec"]["plan"]["engine_memory"]["expected"]["host_pool_mib"] = 1;
        expect(from(rehashed(j)), C::invalid_value, "host_pool_mib", "rehashed receipt-echo mismatch is semantic");
        Json k = prod;
        for (Json& c : k["spec"]["calibration"])
            if (c["key"] == "kv_backing_floor.fraction_ppm") c["value"] = "lots";
        expect(from(rehashed(k)), C::calibration_invalid, "expected an integer", "a calibration value of the wrong type",
               "spec.calibration[");
        Json q = prod;
        q["provenance"]["/spec/nothing_here"] = q["provenance"][""];
        expect(from(rehashed(q)), C::provenance_invalid, "names no value", "provenance for a value that does not exist");
        Json v = prod;
        v["apiVersion"] = "bigcherry/v2";
        expect(from(v), C::unsupported_version, "apiVersion", "another apiVersion");
        check(status_of(C::unsupported_version) == -18, "unsupported version: -18");
    }
    expect(from_text(std::string(kMaxDocumentBytes + 1, ' ')), C::invalid_format, "over the", "oversize input");
    expect(from_text(replace_once(compact, "\"kind\"", std::string("\"k\xc3\"", 4), &ok)), C::invalid_format, "UTF-8",
           "invalid UTF-8 in a member name");
    expect(from_text(replace_once(compact, "\"tp\":", "\"tp\":1.5,\"x\":", &ok)), C::invalid_format, "non-integer",
           "a non-integer number");
    expect(from_text(replace_once(compact, "\"tp\":", "\"tp\":9007199254740992,\"x\":", &ok)), C::invalid_format, "I-JSON",
           "an integer past 2^53 - 1");
    {
        Result<Manifest> r = from_text(replace_once(compact, "{\"apiVersion\"", "{\n\"apiVersion\" ::", &ok));
        check(!r && r.error().code == C::invalid_format && r.error().line == 2u, "malformed JSON reports its line");
    }

    // ---- spec §11 amendment 6: identity leaves need no record of their own, any other /metadata leaf does ----
    {
        Json j = prod;
        for (const char* p : {"/apiVersion", "/kind", "/metadata", "/metadata/producer", "/metadata/instance_id",
                              "/metadata/producer/build_sha256"})
            j["provenance"].erase(p);
        check(prod["provenance"].contains("") && !check_identity_coverage(j),
              "exempt identity paths pass with only the root record");
        check(from(rehashed(j)).ok(), "the manifest with no identity records loads");
        Json k = j;
        k["metadata"]["build_host"] = "x";   // a /metadata leaf that is not exempt (a future field; v1 would refuse it)
        Status s = check_identity_coverage(k);
        check(s && s->code == C::provenance_invalid && s->message.find("/metadata/build_host") != std::string::npos,
              "a non-exempt /metadata leaf without a record is refused: " + (s ? s->message : std::string("passed")));
        k["provenance"]["/metadata/build_host"] = prod["provenance"][""];
        check(!check_identity_coverage(k), "the same leaf with its own record passes");
    }

    // ---- #536 review: the host inventory, cache identity, plan and owner invariants, duplicates, views ----
    {
        const EnginePlan& plan = p.value().spec.plan.engine_memory.expected;
        auto refused_plan = [&](EnginePlan live, C code, const std::string& needle, const std::string& what) {
            const Status st = check_engine_plan(plan, live, 600000);
            check(st && st->code == code && st->message.find(needle) != std::string::npos,
                  what + ": " + (st ? std::string(machine_name(st->code)) + " " + st->message : std::string("admitted")));
        };
        check(plan.host.size() >= 3, "the production plan has three host consumers");
        EnginePlan live = plan;
        live.host[2] = live.host[1];   // same length, one consumer swapped for a repeat of another
        refused_plan(live, C::plan_mismatch, "repeated in the live inventory", "a same-length host substitution");
        live = plan;
        live.host[0].bytes -= 1;
        refused_plan(live, C::plan_mismatch, "differ from the planned", "fewer live host-consumer bytes");
        live = plan;
        live.host.pop_back();
        refused_plan(live, C::plan_mismatch, "missing from the live plan", "a missing expected consumer");
        live = plan;
        live.host[0].residence = Residence::kPageable;
        refused_plan(live, C::plan_mismatch, "residence", "a changed residence");
        live = plan;
        live.checkpoints.want_slots += 1;
        refused_plan(live, C::plan_mismatch, "checkpoint want", "a changed want_slots");
        live = plan;
        live.host_pool_mib += 1;
        refused_plan(live, C::plan_mismatch, "host pool", "a changed host_pool_mib");
        live = plan;
        live.host[0].bytes = -1;
        refused_plan(live, C::invalid_value, "byte counts", "a negative live byte count is invalid input, not drift");
        live = plan;
        live.host[0].residence = static_cast<Residence>(7);
        refused_plan(live, C::invalid_value, "unknown residence", "an out-of-range residence");
    }
    {
        Json j = lab;   // a lab shape change keeps the old measured plan's cache_key
        j["spec"]["experiment"]["shape"]["helper_common"]["gemv"]["h2"] = 2;
        expect(from(rehashed(j)), C::plan_mismatch, "spec §5.1", "a stale cache_key after a shape change",
               "spec.plan.engine_memory.cache_key");
        check(plan_cache_key(p.value()) == prod["spec"]["plan"]["engine_memory"]["cache_key"].get<std::string>(),
              "the production example's cache_key is §5.1's");
        Json k = prod;
        k["spec"]["plan"]["engine_memory"]["expected"]["checkpoints"]["floor_slots"] =
            k["spec"]["plan"]["engine_memory"]["expected"]["checkpoints"]["want_slots"].get<int64_t>() + 1;
        expect(from(rehashed(k)), C::invalid_value, "exceed want_slots", "floor_slots above want_slots");
        Json o = prod;
        o["spec"]["plan"]["owner_map"]["helpers"][0]["bytes"] = 0;
        expect(from(rehashed(o)), C::ownership_mismatch, "device bytes", "a nonempty owned set of zero bytes");
        Json d = prod;
        d["spec"]["processes"]["helpers"][1]["role"] = "helper0";
        expect(from(rehashed(d)), C::duplicate_field, "repeated", "a duplicate helper role");
        Json g = prod;
        g["spec"]["facts"]["gpus"].push_back(g["spec"]["facts"]["gpus"][0]);
        expect(from(rehashed(g)), C::duplicate_field, "detected twice", "a duplicate detected GPU");
        // §2.2: rows_kernel (capability) or lanes.rows (policy) makes the rows kernel active; the typed manifest is
        // edited directly (a rehashed JSON edit would also move the cache key, which covers effective settings)
        Manifest rm = p.value();
        rm.spec.processes.helpers[0].rows_kernel = true;
        HelperLanesPatch lanes;
        lanes.rows = false;
        HelperPatch off;
        off.lanes = lanes;
        rm.spec.processes.helpers[0].settings = off;
        Result<HelperView> hv = helper_view(rm, "helper0");
        check(hv && !hv.value().settings.lanes.rows && hv.value().rows_active, "rows_kernel makes the rows kernel active");
    }
    {   // the decoder: sibling members of one name are fine; absent, null and a value stay distinct; patches apply 0/false
        check(parse_strict_json("{\"a\":{\"x\":1},\"b\":{\"x\":1}}", "t").ok(), "the same name in sibling objects");
        LaneDiagnostics ld;
        check(!decode(Json::parse("{\"trace_level\":1}"), "lanes", &ld) && !ld.capture_file, "absent capture_file");
        check(!decode(Json::parse("{\"capture_file\":null}"), "lanes", &ld) && ld.capture_file && !*ld.capture_file,
              "null capture_file");
        check(!decode(Json::parse("{\"capture_file\":\"f\"}"), "lanes", &ld) && ld.capture_file && **ld.capture_file == "f",
              "capture_file value");
        check(decode(Json::parse("{}"), "lanes", &ld).has_value(), "an empty diagnostics group is refused");
        HelperCommon base = p.value().spec.processes.helper_common;
        base.combine.enabled = true;
        base.kernels.moe_long_rows = 9;
        HelperPatch patch;
        if (!decode(Json::parse("{\"combine\":{\"enabled\":false},\"kernels\":{\"moe_long_rows\":0}}"), "patch", &patch)) {
            apply(base, patch);
            check(!base.combine.enabled && base.kernels.moe_long_rows == 0, "a patch applies false and 0 as values");
        } else {
            check(false, "the patch decodes");
        }
        check(decode(Json::parse("{\"cpu\":{}}"), "patch", &patch).has_value(), "an empty override group is refused");
        Json c = prod;
        c["spec"]["calibration"][0]["seeded"] = "2026-02-30";
        expect(from(rehashed(c)), C::calibration_invalid, "YYYY-MM-DD", "a malformed date");
    }
    // spec §11 amendment 7: pattern strings are bounded before the regex sees them
    {
        std::string deep;
        for (int i = 0; i < 100000; ++i) deep += "/a";
        Json j = prod;
        j["provenance"][deep] = prod["provenance"][""];
        expect(from(rehashed(j)), C::provenance_invalid, "longer than 4096", "a 100k-segment pointer, refused cleanly");
        std::string ptr;
        for (int i = 0; i < 2048; ++i) ptr += "/a";   // 4096 characters, 2048 levels deep
        ProvenanceRecord rec;
        const Json base_rec = prod["provenance"][""];
        Json at_limit = base_rec;
        at_limit["inputs"] = Json::array({ptr});
        check(ptr.size() == 4096 && !decode(at_limit, "provenance[\"\"]", &rec), "a 4096-character pointer passes");
        Json over = base_rec;
        over["inputs"] = Json::array({ptr + "b"});
        const Status s4097 = decode(over, "provenance[\"\"]", &rec);
        check(s4097 && s4097->code == C::provenance_invalid && s4097->message.find("longer than 4096") != std::string::npos,
              "a 4097-character pointer is refused");
        Metadata md;
        Json meta = prod["metadata"];
        check(meta["content_digest"].get<std::string>().size() == 71 && !decode(meta, "metadata", &md), "a 71-character digest");
        meta["content_digest"] = meta["content_digest"].get<std::string>() + "0";
        const Status s72 = decode(meta, "metadata", &md);
        check(s72 && s72->message.find("longer than 71") != std::string::npos, "a 72-character digest is refused");
    }

    {   // amendment 8: nesting is bounded by the schema's own depth (1,000,000 levels took quadratic time and memory)
        const std::string deep = "{\"x\":" + std::string(1000000, '[') + std::string(1000000, ']') + "}";
        expect(from_text(deep), C::invalid_format, "nested deeper than", "a million nested arrays, refused at once");
        const std::string inner = std::string(kMaxDepth - 1, '[') + std::string(kMaxDepth - 1, ']');
        check(parse_strict_json("{\"x\":" + inner + "}", "t").ok(), "kMaxDepth levels parse");
        check(!parse_strict_json("{\"x\":[" + inner + "]}", "t").ok(), "kMaxDepth + 1 levels are refused");
        check(kMaxDepth >= 9 && kMaxDepth < 20, "kMaxDepth fits the examples (9 deep): " + std::to_string(kMaxDepth));
    }

    // ---- re-review of #536: round trips, reused destinations, cheap uniqueness, character counts, inventories ----
    {
        check(encode(p.value()) == prod, "the writer reproduces the production example exactly");
        check(encode(l.value()) == lab, "the writer reproduces the lab example exactly");
        LaneDiagnostics ld;
        decode(Json{{"capture_file", "trace.bin"}}, "lanes", &ld);
        decode(Json{{"trace_level", 1}}, "lanes", &ld);
        check(!ld.capture_file && ld.trace_level == 1, "a reused destination keeps nothing stale");
        Json j = prod;
        Json many = Json::array();
        for (int i = 0; i < 50000; ++i) many.push_back("invalid_" + std::to_string(i));
        j["spec"]["processes"]["helpers"][0]["formats"] = many;
        expect(from(rehashed(j)), C::invalid_value, "is not one of", "50,000 invalid formats fail on the first item");
        Json dup = prod;
        dup["spec"]["processes"]["helpers"][0]["formats"] = Json::array({"w4nl64", "w4nl64"});
        expect(from(rehashed(dup)), C::duplicate_field, "repeats item 0", "a repeated format");
        std::string ptr = "/";
        for (int i = 0; i < 3000; ++i) ptr += "\xc3\xa9";   // 3,001 characters, 6,001 bytes
        ProvenanceRecord rec;
        Json r1 = prod["provenance"][""];
        r1["inputs"] = Json::array({ptr});
        check(!decode(r1, "provenance[\"\"]", &rec), "maxLength counts characters, not bytes");
        const EnginePlan& plan = p.value().spec.plan.engine_memory.expected;
        EnginePlan e2 = plan;
        e2.host.push_back(e2.host[0]);
        const Status st = check_engine_plan(e2, plan, 600000);
        check(st && st->code == C::plan_mismatch && st->message.find("repeated in the planned inventory") != std::string::npos,
              "a repeated name in the expected inventory");
        EnginePlan extra = plan;
        PlanHostConsumer other = extra.host[0];
        other.name = "surprise";
        extra.host.push_back(other);
        const Status sx = check_engine_plan(plan, extra, 600000);
        check(sx && sx->code == C::plan_mismatch && sx->message.find("not in the planned inventory") != std::string::npos,
              "an extra live consumer");
    }

    // ---- renderers (errors.md §3, §4) ----
    {
        ConfigError e = make_error(C::experiment_forbidden, "spec.experiment",
                                   "spec.experiment is forbidden when spec.intent.service.profile=production",
                                   "select profile=lab explicitly and regenerate the manifest");
        e.source_file = "site.toml";
        e.line = 27;
        e.column = 1;
        check(render_json(e) == "{\"code\":\"BC_CFG_EXPERIMENT_FORBIDDEN\",\"status\":-9,\"path\":\"spec.experiment\","
                                "\"message\":\"spec.experiment is forbidden when spec.intent.service.profile=production\","
                                "\"hint\":\"select profile=lab explicitly and regenerate the manifest\",\"source_file\":"
                                "\"site.toml\",\"line\":27,\"column\":1}",
              "JSON rendering: " + render_json(e));
        check(render_human("bc-serve", e) ==
                  "bc-serve: config: BC_CFG_EXPERIMENT_FORBIDDEN (RAD_E_SCHEMA, -9)\n"
                  "  spec.experiment: spec.experiment is forbidden when spec.intent.service.profile=production\n"
                  "  source: site.toml:27:1\n  hint: select profile=lab explicitly and regenerate the manifest\n",
              "human rendering: " + render_human("bc-serve", e));
        ConfigError bare = make_error(C::io_failure, "", "x");
        check(render_json(bare).find("\"source_file\":null,\"line\":null,\"column\":null") != std::string::npos,
              "nullable locations are explicit nulls");
        const std::pair<C, int> table[] = {{C::unknown_field, -9}, {C::missing_field, -9}, {C::duplicate_field, -11},
                                           {C::invalid_type, -12}, {C::invalid_value, -12}, {C::unsupported_version, -18},
                                           {C::invalid_format, -14}, {C::digest_mismatch, -14}, {C::experiment_forbidden, -9},
                                           {C::unresolved_reference, -17}, {C::provenance_invalid, -9},
                                           {C::calibration_invalid, -9}, {C::ownership_mismatch, -9}, {C::plan_mismatch, -9},
                                           {C::admission_refused, -6}, {C::io_failure, -13}, {C::abi_mismatch, -10},
                                           {C::invalid_state, -16}};
        for (const auto& [code, status] : table)
            check(status_of(code) == status, std::string(machine_name(code)) + " maps to " + std::to_string(status));
        check(exit_code_of(make_error(C::admission_refused, "", "")) == 4 && exit_code_of(make_error(C::invalid_state, "", "")) == 5,
              "exit classes 4 and 5");
    }

    // ---- the typed admission check against the shared vectors ----
    {
        const Json doc = Json::parse(read("admission-vectors.json"));
        int run = 0, decode_refusals = 0;
        for (const Json& v : doc.at("vectors")) {
            const std::string name = v.at("name").get<std::string>();
            const Json& verdict = v.at("verdict");
            EnginePlan expected, live;
            Status de = decode(v.at("expected"), "expected", &expected);
            Status dl = de ? de : decode(v.at("live"), "live", &live);
            ++run;
            if (de || dl) {   // a plan the contract cannot even decode is invalid input, with the decoder's exact code
                const ConfigError& e = de ? *de : *dl;
                const C want = name == "kv-resolved-missing" ? C::missing_field : C::invalid_value;
                check(!verdict.at("admit").get<bool>() && verdict.at("class") == "input" && e.code == want,
                      "vector " + name + " does not decode: " + std::string(machine_name(e.code)) + " " + e.message);
                ++decode_refusals;
                continue;
            }
            Status s = check_engine_plan(expected, live, v.at("ppm").get<int64_t>());
            if (verdict.at("admit").get<bool>()) {
                check(!s, "vector " + name + " must admit: " + (s ? s->message : std::string()));
                continue;
            }
            const std::string cls = verdict.at("class").get<std::string>();
            const C want = cls == "admission" ? C::admission_refused : cls == "plan" ? C::plan_mismatch : C::invalid_value;
            check(s && s->code == want && s->message.find(verdict.at("needle").get<std::string>()) != std::string::npos,
                  "vector " + name + ": want " + cls + " '" + verdict.at("needle").get<std::string>() + "', got " +
                      (s ? std::string(machine_name(s->code)) + " " + s->message : std::string("admit")));
            if (s && s->code == C::admission_refused) check(exit_code_of(*s) == 4, "vector " + name + ": exit 4");
        }
        check(run >= 46, "all admission vectors ran");
        check(decode_refusals == 2, "exactly the two out-of-contract vectors fail decoding: " + std::to_string(decode_refusals));
        // errors.md §4's memory admission example is this vector's rendering, byte for byte
        for (const Json& v : doc.at("vectors"))
            if (v.at("name") == "prod-kv-truncation-below") {
                EnginePlan e, l;
                decode(v.at("expected"), "", &e);
                decode(v.at("live"), "", &l);
                const Status st = check_engine_plan(e, l, v.at("ppm").get<int64_t>());
                const std::string doc_md = read("errors.md");
                check(st && doc_md.find(render_human("radiance", *st)) != std::string::npos,
                      "errors.md §4 shows the admission refusal as render_human writes it");
            }
    }

    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
