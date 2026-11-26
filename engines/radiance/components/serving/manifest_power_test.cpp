// RSC02 step 3: real v2 power admission and GFXOFF recording, injected
// hardware and sensors only. No sysfs writes or GPU access in this test.
#include "jcs.h"
#include "manifest_power.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <unistd.h>

#ifndef BC_CONTRACT_V2_DOCS_DIR
#error "BC_CONTRACT_V2_DOCS_DIR must reference the committed v2 contract fixtures"
#endif

namespace fs = std::filesystem;
namespace ps = bc::serving::power;
namespace ss = bc::serving::session;
namespace v2 = bc::serving::v2;
using Json = nlohmann::json;

namespace {
void expect(bool truth, const char* reason) {
    if (!truth)
        throw std::runtime_error(reason);
}
struct Temporary {
    fs::path root;
    Temporary() {
        std::string name = (fs::temp_directory_path() / "bc-manifest-power-XXXXXX").string();
        std::vector<char> buffer(name.begin(), name.end());
        buffer.push_back('\0');
        const auto* created = ::mkdtemp(buffer.data());
        if (!created)
            throw std::runtime_error("mkdtemp");
        root = created;
    }
    ~Temporary() {
        std::error_code unused;
        fs::remove_all(root, unused);
    }
};
class Cards final : public ps::Hardware {
  public:
    int reads = 0, writes = 0;
    int64_t board_cap = 100000000;

  private:
    v2::Result<ps::CardSnapshot> do_inspect(const ps::PciAddress& pci) override {
        ++reads;
        ps::CardSnapshot card{};
        card.pci = pci;
        card.amdgpu = true;
        card.hwmon = "hwmon0";
        for (auto& value : card.controls) {
            value.current = 100;
            value.minimum = 0;
            value.maximum = 1000;
            value.writable = true;
        }
        // The injected cap-only controller has an absolute shader ceiling;
        // it must not advertise the mutually exclusive signed offset.
        card.controls[static_cast<size_t>(ps::ControlId::shader_offset)] = {};
        auto& cap = card.controls[static_cast<size_t>(ps::ControlId::board_cap)];
        cap.current = board_cap;
        cap.minimum = 50000000;
        cap.maximum = 400000000;
        return card;
    }
    v2::Status do_execute(const ps::PowerCommand& command) override {
        ++writes;
        if (const auto* cap = std::get_if<ps::BoardCap>(&command.value))
            board_cap = cap->value.value;
        return std::nullopt;
    }
};
class Telemetry final : public ps::Sensor {
  public:
    int floor = 0, idle = 0;

  private:
    v2::Result<ps::Window> do_measure(const ps::PciAddress& pci, ps::WindowPhase phase, size_t count) override {
        ps::Window window{};
        window.pci = pci;
        window.phase = phase;
        window.identity = "gfx1201";
        window.sensor_identity = "hwmon0";
        window.acquisition = ps::CaptureState::measured;
        for (size_t index = 0; index < count; ++index) {
            const auto offset = phase == ps::WindowPhase::engine_loaded_idle ? 10 : 0;
            window.samples.push_back(
                {ps::Microwatts{20000000 + offset}, ps::Millivolts{800}, 1U,
                 std::chrono::system_clock::now() +
                     std::chrono::seconds(phase == ps::WindowPhase::engine_loaded_idle ? 2 : 0)});
        }
        if (phase == ps::WindowPhase::no_process)
            ++floor;
        else
            ++idle;
        return window;
    }
};
// The S10 output as the owner has it: strictly read once, then handed to power as a typed value.
v2::Manifest typed(const fs::path& path) {
    const auto loaded = v2::load_manifest(path, v2::PlanUse::serving);
    if (!loaded)
        throw std::runtime_error("fixture manifest refused: " + v2::render_human("fixture", loaded.error()));
    return loaded.value();
}
} // namespace

int main() {
    try {
        Temporary t;
        Cards hardware;
        Telemetry sensor;
        ps::DiskRecords records(t.root / "journals", t.root / "receipts");
        ss::ManifestPower bridge(hardware, records, sensor, [] { return false; }, false);
        const fs::path original = fs::path(BC_CONTRACT_V2_DOCS_DIR) / "example.flashnext-4gpu.json";
        const auto admission = bridge.begin(typed(original), v2::PlanUse::serving);
        expect(!admission, "valid strict-v2 manifest with stock observe must admit without a GPU lease");
        expect(hardware.reads == 4 && hardware.writes == 0,
               "observe inspects all four ordered engine/helper BDFs without hardware writes");
        expect(!bridge.verify_serving(original), "same authenticated manifest passes final policy continuity");
        expect(!bridge.capture(ps::WindowPhase::no_process), "floor phase persists advisory evidence");
        expect(!bridge.capture(ps::WindowPhase::engine_loaded_idle), "loaded idle phase persists typed GFXOFF");
        expect(sensor.floor == 1 && sensor.idle == 1, "captures exactly paired floor and idle per engine BDF");
        expect(!bridge.finish(), "observe finish has no restoration writes");
        expect(hardware.writes == 0, "no write path can be reached by an observe manifest");
        // RED P0-2: a second owner execution of the same original bytes
        // must not overwrite the first execution's durable power receipt.
        ss::ManifestPower replay(hardware, records, sensor, [] { return false; }, false);
        expect(!replay.begin(typed(original), v2::PlanUse::serving),
               "second owner must admit identical immutable bytes");
        expect(!replay.verify_serving(original), "second owner has the same authenticated power policy");
        expect(!replay.finish(), "second owner finish");
        size_t receipts = 0;
        for (const auto& entry : fs::directory_iterator(t.root / "receipts"))
            if (entry.is_regular_file())
                ++receipts;
        expect(receipts == 2U, "two observe owner executions require separate durable receipts");
        expect(bridge.begin(typed(original), v2::PlanUse::serving).has_value(),
               "power admission must never be repeated for a run");
        // P0-2: the same strict authenticated apply manifest must admit
        // successive owner executions with distinct durable journal/receipt
        // IDs. A crashed execution cannot be reused: its unresolved journal
        // must block admission until the real recovery API clears it.
        Json apply_manifest = Json::parse(std::ifstream(original));
        apply_manifest["spec"]["plan"]["power"]["mode"] = "apply";
        apply_manifest["spec"]["intent"]["power"]["profile"] = "rsc02-injected-apply";
        apply_manifest["spec"]["plan"]["power"]["profile"] = "rsc02-injected-apply";
        apply_manifest["spec"]["plan"]["power"]["devices"][0]["settings"]["power_cap_w"] = 200;
        apply_manifest["spec"]["intent"]["power"]["role_overrides"]["engine"] = {{"power_cap_w", 200}};
        apply_manifest["provenance"]["/spec/plan/power/mode"]["source"] =
            "test source apply-mode owner admission";
        apply_manifest["metadata"].erase("content_digest");
        std::string apply_digest, digest_failure;
        expect(bc::serving::manifest_content_digest(apply_manifest, &apply_digest, &digest_failure),
               "JCS recomputes modified strict-v2 apply manifest digest");
        apply_manifest["metadata"]["content_digest"] = apply_digest;
        const fs::path apply_path = t.root / "apply-v2-manifest.json";
        std::ofstream(apply_path) << apply_manifest.dump(2) << '\n';
        const auto parsed_apply = v2::load_manifest(apply_path, v2::PlanUse::serving);
        if (!parsed_apply)
            throw std::runtime_error("strict apply fixture refused: " +
                                     v2::render_human("apply", parsed_apply.error()));
        std::string first_apply_id;
        {
            ss::ManifestPower apply(hardware, records, sensor, [] { return true; }, true);
            const auto status = apply.begin(typed(apply_path), v2::PlanUse::serving);
            if (status)
                throw std::runtime_error("injected real apply refused: " + v2::render_human("apply", *status));
            const auto& applied = apply.execution();
            expect(hardware.writes > 0 && applied.has_value(),
                   "real apply must durably admit and issue typed hardware commands");
            if (!applied)
                throw std::runtime_error("durable apply did not issue an execution");
            first_apply_id = applied->run_id;
            expect(applied->manifest_origin_instance_id == parsed_apply.value().metadata.instance_id,
                   "fresh execution keeps the exact immutable origin as reference");
            // Destroy the owner WITHOUT finish: simulate owner crash after
            // durable apply, not a fake journal shape.
        }
        const auto unresolved = records.unresolved_journals();
        expect(unresolved && unresolved.value().size() == 1,
               "real apply crash must leave exactly one recoverable journal");
        ss::ManifestPower blocked(hardware, records, sensor, [] { return true; }, true);
        expect(blocked.begin(typed(apply_path), v2::PlanUse::serving).has_value(),
               "fresh owner must refuse while earlier apply journal is unresolved");
        expect(!ps::recover(
                   unresolved.value().at(0), hardware, records, [] { return true; }, [] { return true; }),
               "real typed crash recovery must restore the prior controls");
        expect(hardware.board_cap == 100000000, "crashed apply's original board cap must be restored");
        ss::ManifestPower apply_replay(hardware, records, sensor, [] { return true; }, true);
        const auto again = apply_replay.begin(typed(apply_path), v2::PlanUse::serving);
        if (again)
            throw std::runtime_error("second apply after recovery refused: " +
                                     v2::render_human("apply-replay", *again));
        const auto& second_execution = apply_replay.execution();
        const auto& observed_execution = replay.execution();
        if (!second_execution || !observed_execution)
            throw std::runtime_error("replay admission did not produce execution identities");
        expect(second_execution->run_id != first_apply_id &&
                   second_execution->run_id != observed_execution->run_id,
               "every apply and observe replay must use a new execution run identity");
        apply_replay.children_verified();
        expect(!apply_replay.finish(), "second apply must restore after verified process exit");
        expect(hardware.board_cap == 100000000, "second apply must restore original hardware control values");

        std::ifstream reader(original);
        Json altered = Json::parse(reader);
        altered["spec"]["processes"]["engine"]["device"]["gpus"][0] = "0000:45:00.0";
        const fs::path changed = t.root / "altered.json";
        std::ofstream(changed) << altered.dump(2);
        expect(bridge.verify_serving(changed).has_value(),
               "tampering with a final BDF without a matching content digest is refused");
        expect(bridge.capture(ps::WindowPhase::engine_loaded_idle).has_value() == false,
               "capturing a repeated advisory window does not write any hardware");
        // The real process is the fake lease's permitted ancestor. All lease
        // files live in this temporary test tree; no hardware lease is acquired.
        const fs::path lease_dir = t.root / "fake-leases";
        fs::create_directories(lease_dir);
        expect(::setenv("AGENT_INSTANCE", "rsc02-step3-fixture", 1) == 0, "set test instance");
        Json fake_lease = {{"instance", "rsc02-step3-fixture"},
                           {"expires", "2099-01-01T00:00:00Z"},
                           {"pids", Json::array({::getpid()})}};
        const auto write_lease = [&](const Json& value) {
            std::ofstream(lease_dir / "gpu.json", std::ios::trunc) << value.dump();
        };
        write_lease(fake_lease);
        expect(ss::owner_has_gpu_lease(lease_dir),
               "current live ancestor, instance and UTC expiry are required");
        fake_lease["expires"] = "2099-01-01T00:00:00junkZ";
        write_lease(fake_lease);
        expect(!ss::owner_has_gpu_lease(lease_dir),
               "lease cannot accept garbage inside a future UTC timestamp");
        fake_lease["expires"] = "2099-01-01T00:00:00.125Z";
        write_lease(fake_lease);
        expect(ss::owner_has_gpu_lease(lease_dir), "fractional RFC3339 UTC timestamp is valid");
        fake_lease["instance"] = "another-instance";
        write_lease(fake_lease);
        expect(!ss::owner_has_gpu_lease(lease_dir), "another lease holder cannot authorize power");
        // One identity key only: a matching lease under the old alias alone, or with no key at all, is denied.
        fake_lease["instance"] = "rsc02-step3-fixture";
        write_lease(fake_lease);
        expect(::unsetenv("AGENT_INSTANCE") == 0, "unset canonical instance");
        expect(!ss::owner_has_gpu_lease(lease_dir), "no instance key cannot authorize power");
        expect(::setenv("CLAUDE_RC_NAME", "rsc02-step3-fixture", 1) == 0, "set old alias");
        expect(!ss::owner_has_gpu_lease(lease_dir), "the old CLAUDE_RC_NAME alias alone cannot authorize power");
        expect(::unsetenv("CLAUDE_RC_NAME") == 0, "unset old alias");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSC02 step 3 test failed: " << error.what() << '\n';
        return 1;
    }
}
