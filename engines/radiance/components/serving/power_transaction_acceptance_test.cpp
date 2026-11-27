#include "power_io.h"
// Card-free RSC02 S12 acceptance: domain assertions and checked-in Python
// behavioural golden fixtures. No Python imports and no physical GPU access.
#include "power_transaction.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <nlohmann/json.hpp>
#include "power_qualification.h"

#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using bc::serving::power::BoardCap;
using bc::serving::power::Capability;
using bc::serving::power::CaptureState;
using bc::serving::power::CardSnapshot;
using bc::serving::power::CommandValue;
using bc::serving::power::ControlId;
using bc::serving::power::DevicePolicy;
using bc::serving::power::Digest;
using bc::serving::power::DiskRecords;
using bc::serving::power::EventKind;
using bc::serving::power::GfxoffReason;
using bc::serving::power::GfxoffState;
template <class T>
concept RawCallableVisible = requires(T& value) { value.function_; };
static_assert(!std::is_invocable_v<bc::serving::power::SessionPower::Predicate>);
static_assert(!std::is_invocable_v<bc::serving::power::LinuxSensor::Interval>);
static_assert(!RawCallableVisible<bc::serving::power::SessionPower::Predicate>);
static_assert(!RawCallableVisible<bc::serving::power::LinuxSensor::Interval>);

using bc::serving::power::Hardware;
using bc::serving::power::HostIssue;
using bc::serving::power::Journal;
using bc::serving::power::LinuxHardware;
using bc::serving::power::LinuxSensor;
using bc::serving::power::Megahertz;
using bc::serving::power::MemoryClock;
using bc::serving::power::Millivolts;
using bc::serving::power::Mode;
using bc::serving::power::OverdriveCommit;
using bc::serving::power::PciAddress;
using bc::serving::power::PowerCommand;
using bc::serving::power::PowerPlan;
using bc::serving::power::Receipt;
using bc::serving::power::Records;
using bc::serving::power::Role;
using bc::serving::power::Sample;
using bc::serving::power::SelectedGpu;
using bc::serving::power::SessionPower;
using bc::serving::power::ShaderClock;
using bc::serving::power::ShaderOffset;
using bc::serving::power::Stage;
using bc::serving::power::SysfsWrite;
using bc::serving::power::Tuning;
using bc::serving::power::Verdict;
using bc::serving::power::VoltageOffset;
using bc::serving::power::Watts;
using bc::serving::power::Window;
using bc::serving::power::WindowPhase;
using bc::serving::v2::ConfigError;
using bc::serving::v2::ConfigErrorCode;
using bc::serving::v2::Result;
using bc::serving::v2::Status;
using Json = nlohmann::json;

void expect(bool truth, const std::string& reason) {
    if (!truth)
        throw std::runtime_error("ASSERT: " + reason);
}
PciAddress bdf(std::string_view value) {
    auto parsed = PciAddress::parse(value);
    if (!parsed)
        throw std::runtime_error(parsed.error().message);
    return parsed.value();
}
Digest digest(std::string_view value) {
    auto parsed = Digest::parse(value);
    if (!parsed)
        throw std::runtime_error(parsed.error().message);
    return parsed.value();
}
Role role(std::string_view name) {
    auto parsed = Role::parse(name);
    if (!parsed)
        throw std::runtime_error(parsed.error().message);
    return parsed.value();
}
void assert_error(Status error, ConfigErrorCode code, std::string_view reason) {
    if (!error)
        throw std::runtime_error("expected typed error " + std::string(reason));
    expect(error->code == code, "expected typed error for " + std::string(reason) + "; got: " + error->message);
    expect(error->message.find(reason) != std::string::npos, "unexpected typed reason: " + error->message);
}
Json load_json(const fs::path& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open test fixture " + path.string());
    return Json::parse(file);
}
void write_text(const fs::path& path, std::string_view value) {
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    file << value;
    expect(bool(file), "fixture file write");
}
struct Temporary {
    fs::path root;
    Temporary() {
        auto path = (fs::temp_directory_path() / "rsc02-cpp-XXXXXX").string();
        std::vector<char> name(path.begin(), path.end());
        name.push_back('\0');
        auto* result = ::mkdtemp(name.data());
        if (!result)
            throw std::runtime_error("mkdtemp");
        root = result;
    }
    ~Temporary() {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
};

struct SimulatedCard {
    PciAddress pci;
    bool amdgpu = true;
    bool writable = true;
    std::optional<std::string> hwmon = "hwmon2";
    int64_t voltage = -100, shader = 2600, memory = 1300, cap = 300000000;
    int64_t max_shader = 3200, max_memory = 1500;
    std::string identity = "kernel=k1;amdgpu=v1;board=744c:1002:0e3b";
    std::optional<int64_t> staged_shader, staged_memory, staged_voltage;
    bool signed_shader = false;
};
class SimulatedHardware final : public Hardware {
  public:
    std::vector<SimulatedCard> cards;
    std::vector<PowerCommand> issued;
    size_t fail_on = std::numeric_limits<size_t>::max();
    bool ignore_cap = false;
    bool fail_restore = false;
    size_t inspect_calls = 0, execute_calls = 0;
    size_t throw_inspect_on = std::numeric_limits<size_t>::max();
    size_t throw_execute_on = std::numeric_limits<size_t>::max();
    size_t throw_unknown_inspect_on = std::numeric_limits<size_t>::max();
    size_t throw_unknown_execute_on = std::numeric_limits<size_t>::max();
    SimulatedCard& card(const PciAddress& address) {
        auto it = std::find_if(cards.begin(), cards.end(),
                               [&](const SimulatedCard& card) { return card.pci == address; });
        if (it == cards.end())
            throw std::runtime_error("missing simulated device");
        return *it;
    }

  private:
    Result<CardSnapshot> do_inspect(const PciAddress& address) override {
        const auto call_index = inspect_calls++;
        if (call_index == throw_unknown_inspect_on)
            throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
        if (call_index == throw_inspect_on)
            throw fs::filesystem_error("injected inspect exception", std::make_error_code(std::errc::io_error));
        const auto& simulated = card(address);
        CardSnapshot snapshot;
        snapshot.pci = address;
        snapshot.amdgpu = simulated.amdgpu;
        snapshot.hwmon = simulated.hwmon;
        snapshot.identity = simulated.identity;
        auto fill = [&](ControlId id, int64_t current, int64_t min, int64_t max) {
            snapshot.controls[static_cast<size_t>(id)] = Capability{current, min, max, simulated.writable};
        };
        fill(ControlId::voltage, simulated.voltage, -200, 0);
        if (simulated.signed_shader)
            fill(ControlId::shader_offset, simulated.shader, -500, 1000);
        else
            fill(ControlId::shader, simulated.shader, 2000, simulated.max_shader);
        fill(ControlId::memory, simulated.memory, 900, simulated.max_memory);
        fill(ControlId::board_cap, simulated.cap, 200000000, 350000000);
        return snapshot;
    }
    Status do_execute(const PowerCommand& command) override {
        const auto call_index = execute_calls++;
        if (call_index == throw_unknown_execute_on)
            throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
        if (call_index == throw_execute_on)
            throw std::runtime_error("injected execute exception");
        if (issued.size() == fail_on) {
            fail_on = std::numeric_limits<size_t>::max();
            return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power.fake",
                                               "injected power write failure");
        }
        auto& simulated = card(command.pci);
        if (fail_restore && std::holds_alternative<ShaderClock>(command.value) &&
            std::get<ShaderClock>(command.value).value.value == 2600)
            return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power.fake",
                                               "injected restore write failure");
        issued.push_back(command);
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, ShaderClock> || std::is_same_v<T, ShaderOffset>) {
                    simulated.staged_shader = value.value.value;
                } else if constexpr (std::is_same_v<T, MemoryClock>) {
                    simulated.staged_memory = value.value.value;
                } else if constexpr (std::is_same_v<T, VoltageOffset>) {
                    simulated.staged_voltage = value.value.value;
                } else if constexpr (std::is_same_v<T, BoardCap>) {
                    if (!ignore_cap)
                        simulated.cap = value.value.value;
                } else if constexpr (std::is_same_v<T, OverdriveCommit>) {
                    if (simulated.staged_shader)
                        simulated.shader = *simulated.staged_shader;
                    if (simulated.staged_memory)
                        simulated.memory = *simulated.staged_memory;
                    if (simulated.staged_shader || simulated.staged_memory)
                        simulated.voltage = 0;
                    if (simulated.staged_voltage)
                        simulated.voltage = *simulated.staged_voltage;
                    simulated.staged_shader.reset();
                    simulated.staged_memory.reset();
                    simulated.staged_voltage.reset();
                }
            },
            command.value);
        return std::nullopt;
    }

  public:
    std::vector<SysfsWrite> operations() const {
        std::vector<SysfsWrite> result;
        for (const auto& command : issued) {
            auto line = bc::serving::power::render(command);
            if (!line)
                throw std::runtime_error(line.error().message);
            result.push_back(line.value());
        }
        return result;
    }
};
struct SimulatedRecords final : Records {
    std::vector<Journal> journals;
    std::vector<Receipt> receipts;
    size_t calls = 0, fail_on = std::numeric_limits<size_t>::max();
    std::set<size_t> fail_indices;
    bool fail_reconcile = false;
    enum class Operation : uint8_t { none, save, unresolved, contains, load, reconcile, journal_path };
    mutable Operation throw_operation = Operation::none;
    bool throw_nonstandard = false;
    size_t throw_save_on = std::numeric_limits<size_t>::max();
    void maybe_throw(Operation operation) const {
        if (throw_operation == operation) {
            throw_operation = Operation::none;
            if (throw_nonstandard)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("injected records virtual exception");
        }
    }
    bool inject_failure() {
        maybe_throw(Operation::save);
        const size_t index = calls++;
        if (index == throw_save_on) {
            throw_save_on = std::numeric_limits<size_t>::max();
            if (throw_nonstandard)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("injected records persistence exception");
        }
        return index == fail_on || fail_indices.contains(index);
    }

  private:
    Status do_save(const Journal& journal) override {
        if (inject_failure())
            return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power.records",
                                               "injected journal persistence failure");
        auto found = std::find_if(journals.begin(), journals.end(),
                                  [&](const Journal& j) { return j.run_id == journal.run_id; });
        if (found == journals.end())
            journals.push_back(journal);
        else
            *found = journal;
        return std::nullopt;
    }
    Status do_save(const Receipt& receipt) override {
        if (inject_failure())
            return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power.records",
                                               "injected receipt persistence failure");
        auto found = std::find_if(receipts.begin(), receipts.end(),
                                  [&](const Receipt& r) { return r.run_id == receipt.run_id; });
        if (found == receipts.end())
            receipts.push_back(receipt);
        else
            *found = receipt;
        return std::nullopt;
    }
    Result<std::vector<fs::path>> do_unresolved_journals() override {
        maybe_throw(Operation::unresolved);
        std::vector<fs::path> pending;
        for (const auto& record : journals)
            if (bc::serving::power::unresolved(record.stage))
                pending.push_back(journal_path(record.run_id));
        return pending;
    }
    Result<bool> do_contains(std::string_view id) override {
        maybe_throw(Operation::contains);
        return std::any_of(journals.begin(), journals.end(), [&](const Journal& j) { return j.run_id == id; });
    }
    Result<Journal> do_load(const fs::path& path) override {
        maybe_throw(Operation::load);
        auto found = std::find_if(journals.begin(), journals.end(),
                                  [&](const Journal& j) { return journal_path(j.run_id) == path; });
        if (found == journals.end())
            return bc::serving::v2::make_error(ConfigErrorCode::io_failure, path.string(),
                                               "missing simulated journal");
        return *found;
    }
    Status do_reconcile_recovery(const Journal& journal, const bc::serving::power::Observation& stop) override {
        maybe_throw(Operation::reconcile);
        if (fail_reconcile)
            return bc::serving::v2::make_error(ConfigErrorCode::io_failure, "power.receipt",
                                               "injected recovery receipt persistence failure");
        auto found = std::find_if(receipts.begin(), receipts.end(),
                                  [&](const Receipt& r) { return r.run_id == journal.run_id; });
        if (found == receipts.end())
            return bc::serving::v2::make_error(ConfigErrorCode::invalid_format, "power.receipt",
                                               "missing manifest-bound recovery receipt");
        if (found->plan.manifest_digest.value != journal.manifest_digest.value ||
            found->plan.profile_digest.value != journal.profile_digest.value)
            return bc::serving::v2::make_error(ConfigErrorCode::invalid_format, "power.receipt",
                                               "recovery manifest binding mismatch");
        for (const auto& saved : journal.devices) {
            const bool present = std::any_of(
                found->plan.devices.begin(), found->plan.devices.end(), [&](const DevicePolicy& policy) {
                    return policy.device.pci == saved.policy.device.pci &&
                           policy.tuning.voltage_offset == saved.policy.tuning.voltage_offset &&
                           policy.tuning.shader_ceiling == saved.policy.tuning.shader_ceiling &&
                           policy.tuning.memory_ceiling == saved.policy.tuning.memory_ceiling &&
                           policy.tuning.board_cap == saved.policy.tuning.board_cap;
                });
            if (!present)
                return bc::serving::v2::make_error(ConfigErrorCode::invalid_format, "power.receipt",
                                                   "recovery receipt selection mismatch");
        }
        found->stage = journal.stage;
        found->events = journal.events;
        found->observations.push_back(stop);
        return std::nullopt;
    }
    fs::path do_journal_path(std::string_view id) const override {
        maybe_throw(Operation::journal_path);
        return fs::path("/simulated") / (std::string(id) + ".json");
    }
};

PowerPlan make_plan(const Json& fixture) {
    PowerPlan result;
    result.mode = fixture.at("mode") == "apply" ? Mode::apply : Mode::observe;
    result.profile = fixture.at("profile").get<std::string>();
    result.profile_digest = digest(fixture.at("profile_digest").get<std::string>());
    result.manifest_digest = digest(fixture.at("manifest_digest").get<std::string>());
    for (const auto& device : fixture.at("devices")) {
        Tuning tuning;
        const auto& s = device.at("settings");
        if (s.contains("vddgfx_offset_mv"))
            tuning.voltage_offset = Millivolts{s.at("vddgfx_offset_mv").get<int64_t>()};
        if (s.contains("sclk_max_mhz"))
            tuning.shader_ceiling = Megahertz{s.at("sclk_max_mhz").get<int64_t>()};
        if (s.contains("mclk_max_mhz"))
            tuning.memory_ceiling = Megahertz{s.at("mclk_max_mhz").get<int64_t>()};
        if (s.contains("power_cap_w"))
            tuning.board_cap = Watts{s.at("power_cap_w").get<int64_t>()};
        result.devices.push_back(
            {SelectedGpu{role(device.at("role").get<std::string>()), bdf(device.at("bdf").get<std::string>())},
             tuning});
    }
    return result;
}
SimulatedHardware make_hardware(const Json& fixture) {
    SimulatedHardware fake;
    for (const auto& device : fixture.at("devices")) {
        const auto& state = device.at("before");
        SimulatedCard card;
        card.pci = bdf(device.at("bdf").get<std::string>());
        card.hwmon = device.at("hwmon").get<std::string>();
        card.voltage = state.at("vddgfx_offset_mv").get<int64_t>();
        card.shader = state.at("sclk_max_mhz").get<int64_t>();
        card.memory = state.at("mclk_max_mhz").get<int64_t>();
        card.cap = state.at("power_cap_uw").get<int64_t>();
        fake.cards.push_back(card);
    }
    return fake;
}
std::vector<SysfsWrite> golden_ops(const Json& golden, const char* name) {
    std::vector<SysfsWrite> out;
    for (const auto& op : golden.at(name)) {
        out.push_back({op.at("path").get<std::string>(), op.at("value").get<std::string>()});
    }
    return out;
}
const Json& matching(const Json& golden, const std::string& name) {
    for (const auto& sample : golden.at("cases"))
        if (sample.at("name") == name)
            return sample;
    throw std::runtime_error("missing golden case " + name);
}
void golden_parity() {
    const auto data = fs::path(__FILE__).parent_path() / "testdata";
    const auto inputs = load_json(data / "power_qualified_inputs.json");
    const auto golden = load_json(data / "power_qualified_golden.json");
    const auto transactions = load_json(data / "power_transaction_oracle_golden.json");
    expect(inputs.at("cases").size() == golden.at("cases").size(), "golden fixture count");
    for (const auto& fixture : inputs.at("cases")) {
        const std::string name = fixture.at("name").get<std::string>();
        const auto& oracle = matching(golden, name);
        auto hardware = make_hardware(fixture);
        auto plan = make_plan(fixture);
        Temporary t;
        DiskRecords records(t.root / "journals", t.root / "receipts");
        Receipt initial;
        initial.run_id = "golden";
        initial.plan = plan;
        expect(!records.save(initial), name + ": initial receipt serialization");
        expect(load_json(t.root / "receipts/golden.power.json") == oracle.at("receipt"),
               name + ": entire Python golden initial receipt, not merely a projection");
        SessionPower session(plan, "golden", hardware, records, [] { return true; }, [] { return true; });
        expect(!session.begin(), name + ": begin");
        expect(hardware.operations() == golden_ops(oracle, "apply"),
               name + ": Python golden apply command sequence");
        if (plan.mode == Mode::apply) {
            const auto& transaction = matching(transactions, name);
            expect(transaction.at("prepared_state") == "prepared" &&
                       transaction.at("applied_state") == bc::serving::power::stage_name(session.stage()),
                   name + ": real Python transaction prepare/apply stages");
            const auto journal = load_json(records.journal_path("golden"));
            expect(journal.at("touched") == transaction.at("touched"),
                   name + ": Python transaction durable touched inventory and BDF order");
            Json event_kinds = Json::array();
            for (const auto& event : journal.at("events"))
                if (event.at("kind") == "applied")
                    event_kinds.push_back(event.at("kind"));
            expect(event_kinds == transaction.at("applied_events"),
                   name + ": real Python transaction applied event sequence");
            for (const auto& card : hardware.cards) {
                const auto& actual = transaction.at("applied_values").at(card.pci.value);
                expect(actual.at("sclk_max_mhz") == card.shader && actual.at("mclk_max_mhz") == card.memory &&
                           actual.at("vddgfx_offset_mv") == card.voltage &&
                           actual.at("power_cap_w") == card.cap,
                       name + ": real Python transaction applied readback controls");
            }
        }
        auto pre_stop = load_json(t.root / "receipts/golden.power.json");
        expect(pre_stop.at("observations").at("start").at("cards").size() == plan.devices.size(),
               name + ": typed measured start observation covers selected devices");
        expect(!session.report_gfxoff({GfxoffState::active, GfxoffReason::matching_floor}),
               name + ": advisory GFXOFF receipt update");
        expect(!session.finish(), name + ": verified stop");
        const auto full = hardware.operations();
        auto expected = golden_ops(oracle, "apply");
        const auto restore = golden_ops(oracle, "restore");
        expected.insert(expected.end(), restore.begin(), restore.end());
        expect(full == expected, name + ": Python golden restoration sequence");
        const auto actual_receipt = load_json(t.root / "receipts/golden.power.json");
        const auto reference = oracle.at("receipt");
        expect(actual_receipt.at("observations").at("gfxoff").at("status") == "active",
               name + ": typed GFXOFF verdict is persisted as an observation");
        expect(actual_receipt.at("schema_version") == reference.at("schema_version") &&
                   actual_receipt.at("run_id") == reference.at("run_id") &&
                   actual_receipt.at("manifest") == reference.at("manifest") &&
                   actual_receipt.at("request") == reference.at("request"),
               name + ": Python golden observable receipt/request projection");
        expect(actual_receipt.at("transaction").at("state") ==
                   (plan.mode == Mode::observe ? "observe" : "restored"),
               name + ": complete final receipt");
        if (plan.mode == Mode::apply) {
            const auto& transaction = matching(transactions, name);
            const auto journal = load_json(records.journal_path("golden"));
            expect(journal.at("state") == transaction.at("restored_state"),
                   name + ": real Python transaction durable restored journal");
            Json event_kinds = Json::array();
            for (const auto& event : journal.at("events"))
                if (event.at("kind") == "applied" || event.at("kind") == "restored")
                    event_kinds.push_back(event.at("kind"));
            expect(event_kinds == transaction.at("restored_events"),
                   name + ": real Python transaction apply/restore event sequence");
        }
    }
}

void signed_offset_transaction() {
    auto fixture = load_json(fs::path(__FILE__).parent_path() /
        "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto plan = make_plan(fixture);
    auto hardware = make_hardware(fixture);
    plan.devices.resize(1);
    auto& dev = plan.devices.front();
    auto& card = hardware.card(dev.device.pci);
    card.signed_shader = true;
    card.shader = -25;
    card.voltage = -10;
    dev.tuning = {};
    dev.tuning.shader_offset = Megahertz{-100};
    dev.tuning.voltage_offset = Millivolts{-50};
    Temporary t;
    DiskRecords records(t.root / "j", t.root / "r");
    SessionPower session(plan, "signed", hardware, records, [] { return true; }, [] { return true; });
    expect(!session.begin(), "signed apply");
    expect(card.shader == -100 && card.voltage == -50, "signed readback");
    auto ops = hardware.operations();
    expect(ops.size() == 4 && ops[0].value == "s -100" && ops[3].value == "c",
           "RDNA4 signed command grammar");
    const auto saved_journal = records.journal_path("signed");
    const auto good = load_json(saved_journal);
    auto missing = good;
    missing["devices"][0]["before"].erase("sclk_offset_mhz");
    write_text(saved_journal, missing.dump());
    const auto writes = hardware.issued.size();
    expect(bool(bc::serving::power::recover(saved_journal, hardware, records,
           [] { return true; }, [] { return true; })), "missing signed original refuses");
    expect(hardware.issued.size() == writes, "no writes on missing signed original");
    write_text(saved_journal, good.dump());
    const auto identity = card.identity;
    card.identity = "changed board";
    expect(bool(bc::serving::power::recover(saved_journal, hardware, records,
           [] { return true; }, [] { return true; })), "changed physical identity refuses");
    expect(hardware.issued.size() == writes, "no writes on changed signed identity");
    card.identity = identity;
    expect(!bc::serving::power::recover(records.journal_path("signed"), hardware, records,
           [] { return true; }, [] { return true; }), "signed journal recovery");
    expect(card.shader == -25 && card.voltage == -10, "exact original signed values");
    Temporary t2;
    DiskRecords records2(t2.root / "j", t2.root / "r");
    hardware.fail_on = hardware.issued.size() + 2;
    SessionPower fault(plan, "signed-fault", hardware, records2, [] { return true; }, [] { return true; });
    expect(bool(fault.begin()), "signed staged write failure");
    expect(card.shader == -25 && card.voltage == -10, "signed rollback after staged failure");
    Temporary t3;
    DiskRecords records3(t3.root / "j", t3.root / "r");
    dev.tuning.shader_offset = Megahertz{0};
    SessionPower zero(plan, "signed-zero", hardware, records3, [] { return true; }, [] { return true; });
    expect(!zero.begin() && card.shader == 0, "explicit zero offset is not absent");
    expect(!zero.finish() && card.shader == -25 && card.voltage == -10, "zero offset restores nonzero original");
    bc::serving::v2::PowerSettings requested;
    requested.sclk_offset_mhz = 0;
    requested.vddgfx_offset_mv = -50;
    expect(bc::serving::power::qualifying_receipt(t3.root / "r", card.pci, requested, card.identity).has_value(),
           "exact signed receipt qualifies controls");
    auto receipt = load_json(t3.root / "r/signed-zero.power.json");
    receipt["observations"]["stop"]["cards"][0].erase("sclk_offset_mhz");
    write_text(t3.root / "r/signed-zero.power.json", receipt.dump());
    expect(!bc::serving::power::qualifying_receipt(t3.root / "r", card.pci, requested, card.identity),
           "receipt missing signed restoration never qualifies");

}
void binding_and_preflight() {
    expect(!Role::parse("helper9999999999999999999999999999999"),
           "oversized helper index refused without integer wraparound");
    bc::serving::v2::Manifest manifest{};
    manifest.metadata.content_digest = "sha256:" + std::string(64U, 'b');
    manifest.spec.intent.power.profile = "qualified";
    auto& power = manifest.spec.plan.power;
    power.mode = bc::serving::v2::PowerMode::kApply;
    power.profile = "qualified";
    power.profile_digest = "sha256:" + std::string(64U, 'a');
    bc::serving::v2::PowerDevice gpu{};
    gpu.role = "engine";
    gpu.bdf = "0000:03:00.0";
    gpu.settings.power_cap_w = 280;
    power.devices.push_back(gpu);
    std::vector<SelectedGpu> selected{{role("engine"), bdf("0000:03:00.0")}};
    auto bound = bc::serving::power::bind(manifest, selected);
    expect(bool(bound) && bound.value().devices.size() == 1U,
           "typed strict-reader manifest binds selected role/BDF");
    selected[0].pci = bdf("0000:06:00.0");
    expect(!bc::serving::power::bind(manifest, selected),
           "power binder refuses different actual selected device");
    selected[0].pci = bdf("0000:03:00.0");
    power.devices.front().settings.sclk_offset_mhz = -100;
    auto signed_plan = bc::serving::power::bind(manifest, selected);
    expect(bool(signed_plan) && signed_plan.value().devices.front().tuning.shader_offset->value == -100,
           "signed offset binds typed plan");
    power.devices.front().settings.sclk_max_mhz = 2600;
    expect(!bc::serving::power::bind(manifest, selected), "incompatible shader controls refuse");
    Temporary t;
    auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto plan = make_plan(fixture);
    auto hardware = make_hardware(fixture);
    hardware.card(bdf("0000:06:00.0")).max_shader = 2750;
    DiskRecords records(t.root / "j", t.root / "r");
    SessionPower rejected(plan, "bad", hardware, records, [] { return true; }, [] { return true; });
    assert_error(rejected.begin(), ConfigErrorCode::admission_refused, "outside device range");
    expect(hardware.issued.empty(), "whole-device preflight failed without writes");
    hardware.card(bdf("0000:06:00.0")).max_shader = 3200;
    hardware.card(bdf("0000:06:00.0")).writable = false;
    SessionPower permissions(plan, "permissions", hardware, records, [] { return true; }, [] { return true; });
    assert_error(permissions.begin(), ConfigErrorCode::admission_refused, "not writable");
    expect(hardware.issued.empty(), "permissions refused without writes");
    SessionPower no_lease(plan, "unleased", hardware, records, [] { return false; }, [] { return true; });
    assert_error(no_lease.begin(), ConfigErrorCode::admission_refused, "lease");
    expect(hardware.issued.empty(), "unleased start made no writes");
}
bool originals_preserved(const SimulatedHardware& hardware, const Json& fixture) {
    for (const auto& card : fixture.at("devices")) {
        const auto pci = bdf(card.at("bdf").get<std::string>());
        const auto it = std::find_if(hardware.cards.begin(), hardware.cards.end(),
                                     [&](const SimulatedCard& c) { return c.pci == pci; });
        if (it == hardware.cards.end())
            return false;
        const auto& before = card.at("before");
        if (it->shader != before.at("sclk_max_mhz").get<int64_t>() ||
            it->memory != before.at("mclk_max_mhz").get<int64_t>() ||
            it->voltage != before.at("vddgfx_offset_mv").get<int64_t>() ||
            it->cap != before.at("power_cap_uw").get<int64_t>())
            return false;
    }
    return true;
}
void fault_matrix(const Json& fixture, const PowerPlan& plan) {
    const auto oracle =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_transaction_oracle_golden.json");
    const auto python = matching(oracle, "qualified_two_device");
    for (const auto& failure : python.at("faults")) {
        const size_t index = failure.at("index").get<size_t>();
        auto hardware = make_hardware(fixture);
        hardware.fail_on = index;
        SimulatedRecords records;
        SessionPower session(
            plan, "python-command-fault", hardware, records, [] { return true; }, [] { return true; });
        assert_error(session.begin(), ConfigErrorCode::io_failure, "injected");
        expect(originals_preserved(hardware, fixture), "Python transaction failure restores every original");
        expect(!records.journals.empty() && bc::serving::power::stage_name(records.journals.front().stage) ==
                                                failure.at("state").get<std::string>(),
               "Python transaction failure durable stage " + std::to_string(index));
        Json touched = Json::array();
        for (const auto& pci : records.journals.front().touched)
            touched.push_back(pci.value);
        expect(touched == failure.at("touched"),
               "Python transaction fault touched inventory " + std::to_string(index));
        Json events = Json::array();
        for (const auto& event : records.journals.front().events) {
            switch (event.kind) {
            case EventKind::applied:
                events.push_back("applied");
                break;
            case EventKind::restore_failed:
            case EventKind::rollback:
            case EventKind::touched:
            case EventKind::prepared:
            case EventKind::restore_refused:
                break;
            case EventKind::apply_failed:
                events.push_back("apply_failed");
                break;
            case EventKind::restored:
                events.push_back("restored");
                break;
            }
        }
        expect(events == failure.at("events"),
               "Python transaction fault observable event order " + std::to_string(index));
    }
    for (size_t index = 0; index < 10U; ++index) {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        records.fail_on = index;
        SessionPower session(plan, "persist", hardware, records, [] { return true; }, [] { return true; });
        const auto result = session.begin();
        expect(bool(result) && result->code == ConfigErrorCode::io_failure,
               "persistence index " + std::to_string(index) + " must fail typed");
        expect(originals_preserved(hardware, fixture),
               "persistence index " + std::to_string(index) + " must restore all controls");
        if (index >= 6U) {
            expect(session.stage() == Stage::rolled_back,
                   "post-write persistence index " + std::to_string(index) + " rolled back");
            expect(!records.journals.empty() && records.journals.back().stage == Stage::rolled_back,
                   "post-write persistence index " + std::to_string(index) + " journal reconciled");
        }
    }
    for (size_t fail_at : {size_t{6}, size_t{8}}) {
        for (size_t persist_after : {size_t{1}, size_t{2}}) {
            auto hardware = make_hardware(fixture);
            SimulatedRecords records;
            records.fail_on = fail_at;
            records.fail_indices.insert(fail_at + persist_after);
            SessionPower session(
                plan, "failed-rollback-record", hardware, records, [] { return true; }, [] { return true; });
            expect(bool(session.begin()), "rollback persistence failure must report error");
            expect(originals_preserved(hardware, fixture),
                   "rollback persist failure must still restore hardware");
            expect(!records.journals.empty() && bc::serving::power::unresolved(records.journals.back().stage),
                   "failed rollback persistence must leave unresolved durable journal");
        }
    }
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        records.fail_on = 6U;
        hardware.fail_restore = true;
        SessionPower session(
            plan, "rollback-hardware-fault", hardware, records, [] { return true; }, [] { return true; });
        expect(bool(session.begin()), "rollback hardware fault must refuse apply");
        expect(!records.journals.empty() && records.journals.back().stage == Stage::rollback_failed,
               "failed rollback retains unresolved journal");
    }
    auto baseline_hardware = make_hardware(fixture);
    SimulatedRecords baseline_records;
    SessionPower baseline(
        plan, "baseline", baseline_hardware, baseline_records, [] { return true; }, [] { return true; });
    expect(!baseline.begin(), "baseline for enumerating every hardware call");
    const size_t inspections = baseline_hardware.inspect_calls;
    const size_t commands = baseline_hardware.execute_calls;
    for (size_t index = 0; index < inspections; ++index) {
        auto hardware = make_hardware(fixture);
        hardware.throw_inspect_on = index;
        SimulatedRecords records;
        SessionPower session(plan, "thrown-probe", hardware, records, [] { return true; }, [] { return true; });
        const auto fault = session.begin();
        expect(bool(fault) && fault->code == ConfigErrorCode::io_failure,
               "inspection exception " + std::to_string(index) + " converts to typed error");
        expect(originals_preserved(hardware, fixture),
               "inspection exception " + std::to_string(index) + " restores all controls");
        if (!records.journals.empty() && !hardware.issued.empty())
            expect(records.journals.back().stage == Stage::rolled_back,
                   "inspection exception after writing rolls back durable journal");
    }
    for (size_t index = 0; index < commands; ++index) {
        auto hardware = make_hardware(fixture);
        hardware.throw_execute_on = index;
        SimulatedRecords records;
        SessionPower session(
            plan, "thrown-command", hardware, records, [] { return true; }, [] { return true; });
        const auto fault = session.begin();
        expect(bool(fault) && fault->code == ConfigErrorCode::io_failure,
               "execute exception " + std::to_string(index) + " converts to typed error");
        expect(originals_preserved(hardware, fixture),
               "execute exception " + std::to_string(index) + " restores all controls");
        expect(!records.journals.empty() && records.journals.back().stage == Stage::rolled_back,
               "execute exception " + std::to_string(index) + " rolls back durable journal");
    }
}

void injected_interface_exception_matrix() {
    const auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    const auto plan = make_plan(fixture);
    auto baseline_hardware = make_hardware(fixture);
    SimulatedRecords baseline_records;
    SessionPower baseline(
        plan, "count-callable-sites", baseline_hardware, baseline_records, [] { return true; },
        [] { return true; });
    expect(!baseline.begin(), "enumerate every injected hardware call");
    for (bool nonstandard : {false, true}) {
        {
            SimulatedRecords records;
            records.throw_operation = SimulatedRecords::Operation::journal_path;
            records.throw_nonstandard = nonstandard;
            bool typed = false;
            try {
                (void)records.journal_path("throw-journal-path");
            } catch (const bc::serving::callback::Failure& failure) {
                typed = failure.error().code == ConfigErrorCode::io_failure;
            }
            expect(typed, "journal-path virtual throws typed failure");
        }
        for (size_t index = 0; index < baseline_hardware.inspect_calls; ++index) {
            auto hardware = make_hardware(fixture);
            if (nonstandard)
                hardware.throw_unknown_inspect_on = index;
            else
                hardware.throw_inspect_on = index;
            SimulatedRecords records;
            SessionPower session(
                plan, "throw-hardware-inspect", hardware, records, [] { return true; }, [] { return true; });
            const auto failure = session.begin();
            expect(bool(failure) && failure->code == ConfigErrorCode::io_failure,
                   "every Hardware::inspect injection returns typed failure");
            expect(originals_preserved(hardware, fixture), "Hardware::inspect injection restores all controls");
            if (!hardware.issued.empty())
                expect(!records.journals.empty() && records.journals.back().stage == Stage::rolled_back,
                       "Hardware::inspect injection after writes reconciles rollback journal");
        }
        for (size_t index = 0; index < 10U; ++index) {
            auto hardware = make_hardware(fixture);
            SimulatedRecords records;
            records.throw_save_on = index;
            records.throw_nonstandard = nonstandard;
            SessionPower session(
                plan, "throw-record", hardware, records, [] { return true; }, [] { return true; });
            const auto failure = session.begin();
            expect(bool(failure) && failure->code == ConfigErrorCode::io_failure,
                   "throwing save at each persistence index must return typed error");
            expect(originals_preserved(hardware, fixture),
                   "throwing persistence must not leave modified hardware");
            if (index >= 6U) {
                expect(session.stage() == Stage::rolled_back && !records.journals.empty() &&
                           records.journals.back().stage == Stage::rolled_back,
                       "throwing post-write save must restore and reconcile journal");
            }
        }
        for (const auto operation :
             {SimulatedRecords::Operation::contains, SimulatedRecords::Operation::unresolved}) {
            auto hardware = make_hardware(fixture);
            SimulatedRecords records;
            records.throw_nonstandard = nonstandard;
            records.throw_operation = operation;
            SessionPower session(
                plan, "throw-preflight", hardware, records, [] { return true; }, [] { return true; });
            const auto failure = session.begin();
            expect(bool(failure) && failure->code == ConfigErrorCode::io_failure && hardware.issued.empty(),
                   "prewrite injected Records failure must not write");
        }
        for (const auto operation :
             {SimulatedRecords::Operation::load, SimulatedRecords::Operation::reconcile}) {
            auto hardware = make_hardware(fixture);
            SimulatedRecords records;
            records.throw_nonstandard = nonstandard;
            SessionPower session(
                plan, "throw-recovery-record", hardware, records, [] { return true; }, [] { return true; });
            expect(!session.begin(), "records recovery fault fixture");
            records.throw_operation = operation;
            const auto failure = bc::serving::power::recover(
                records.journal_path("throw-recovery-record"), hardware, records, [] { return true; },
                [] { return true; });
            expect(bool(failure) && failure->code == ConfigErrorCode::io_failure,
                   "recovery injected Records failure must return typed IO failure");
            expect(!records.journals.empty() && bc::serving::power::unresolved(records.journals.back().stage),
                   "recovery Records failure retains unresolved journal");
            expect(!bc::serving::power::recover(
                       records.journal_path("throw-recovery-record"), hardware, records, [] { return true; },
                       [] { return true; }),
                   "recovery after transient Records callback failure");
            expect(originals_preserved(hardware, fixture) && records.journals.back().stage == Stage::restored &&
                       records.receipts.back().stage == Stage::restored,
                   "retried recovery restores originals and both durable records");
        }
        for (size_t index = 0; index < baseline_hardware.execute_calls; ++index) {
            auto hardware = make_hardware(fixture);
            hardware.throw_unknown_execute_on = index;
            SimulatedRecords records;
            SessionPower session(
                plan, "throw-nonstandard-execute", hardware, records, [] { return true; }, [] { return true; });
            const auto failure = session.begin();
            expect(bool(failure) && failure->code == ConfigErrorCode::io_failure,
                   "injected nonstandard hardware command exception is typed");
            expect(originals_preserved(hardware, fixture) && !records.journals.empty() &&
                       records.journals.back().stage == Stage::rolled_back,
                   "hardware throw restores originals and journal");
        }
    }
}

void failures_and_recovery() {
    auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto plan = make_plan(fixture);
    fault_matrix(fixture, plan);
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        hardware.fail_on = 6U;
        SessionPower session(plan, "rollback", hardware, records, [] { return true; }, [] { return true; });
        assert_error(session.begin(), ConfigErrorCode::io_failure, "injected");
        expect(session.stage() == Stage::rolled_back, "partial failure rolled back, durable state recorded");
        expect(hardware.card(bdf("0000:03:00.0")).voltage == -100 &&
                   hardware.card(bdf("0000:03:00.0")).cap == 300000000 &&
                   hardware.card(bdf("0000:06:00.0")).shader == 2700,
               "rollback restored all changed devices");
    }
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        records.fail_on = 1U; // initial receipt persistence after prepared journal
        SessionPower session(plan, "io_failure", hardware, records, [] { return true; }, [] { return true; });
        assert_error(session.begin(), ConfigErrorCode::io_failure, "injected");
        expect(hardware.issued.empty(), "receipt persistence failure blocks any write");
    }
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        bool lease = true, children = false;
        {
            SessionPower terminated(
                plan, "killed", hardware, records, [&] { return lease; }, [&] { return children; });
            expect(!terminated.begin(), "killed owner began successfully");
            const auto count = hardware.issued.size();
            assert_error(terminated.finish(), ConfigErrorCode::admission_refused, "may still be alive");
            expect(hardware.issued.size() == count, "child potentially alive prevents any restore writes");
        }
        SessionPower restarted(
            plan, "next", hardware, records, [&] { return lease; }, [&] { return children; });
        assert_error(restarted.begin(), ConfigErrorCode::admission_refused, "unresolved power journal");
        const auto pending = records.journal_path("killed");
        assert_error(bc::serving::power::recover(
                         pending, hardware, records, [&] { return lease; }, [&] { return children; }),
                     ConfigErrorCode::admission_refused, "not verified exited");
        children = true;
        lease = false;
        assert_error(bc::serving::power::recover(
                         pending, hardware, records, [&] { return lease; }, [&] { return children; }),
                     ConfigErrorCode::admission_refused, "lease");
        lease = true;
        records.journals.front()
            .devices.front()
            .before.controls[static_cast<size_t>(ControlId::shader)]
            .current = 5000;
        const auto writes_before_refusal = hardware.issued.size();
        assert_error(bc::serving::power::recover(
                         pending, hardware, records, [&] { return lease; }, [&] { return children; }),
                     ConfigErrorCode::admission_refused, "original shader clock");
        expect(hardware.issued.size() == writes_before_refusal,
               "all-device recovery preflight refuses a forged original before writes");
        records.journals.front()
            .devices.front()
            .before.controls[static_cast<size_t>(ControlId::shader)]
            .current = 2600;
        hardware.fail_restore = true;
        assert_error(bc::serving::power::recover(
                         pending, hardware, records, [&] { return lease; }, [&] { return children; }),
                     ConfigErrorCode::io_failure, "injected");
        expect(records.journals.front().stage == Stage::restore_failed,
               "failed recovery retains an unresolved journal");
        hardware.fail_restore = false;
        records.fail_reconcile = true;
        assert_error(bc::serving::power::recover(
                         pending, hardware, records, [&] { return lease; }, [&] { return children; }),
                     ConfigErrorCode::io_failure, "injected recovery receipt persistence");
        expect(records.journals.front().stage == Stage::restore_failed,
               "recovery receipt failure must preserve unresolved journal");
        records.fail_reconcile = false;
        expect(!bc::serving::power::recover(
                   pending, hardware, records, [&] { return lease; }, [&] { return children; }),
               "explicit crash recovery succeeds after all children are exited");
        expect(records.journals.front().stage == Stage::restored, "recovered state persisted");
        expect(!records.receipts.empty() && records.receipts.front().stage == Stage::restored,
               "crash recovery must reconcile same-run receipt to restored state");
        expect(std::any_of(records.receipts.front().events.begin(), records.receipts.front().events.end(),
                           [](const auto& event) { return event.kind == EventKind::restored; }),
               "recovered receipt records restoration events");
        SessionPower new_owner(plan, "new", hardware, records, [&] { return lease; }, [&] { return children; });
        expect(!new_owner.begin(), "recovered run no longer blocks a new owner");
    }
}

void python_refusal_and_recovery_parity() {
    const auto data = fs::path(__FILE__).parent_path() / "testdata";
    const auto fixture = load_json(data / "power_qualified_inputs.json").at("cases").at(0);
    const auto oracle = load_json(data / "power_transaction_oracle_golden.json");
    const auto python = matching(oracle, "qualified_two_device");
    const auto plan = make_plan(fixture);
    for (const auto& refusal : python.at("refusals")) {
        const auto kind = refusal.at("kind").get<std::string>();
        std::string category = "unresolved";
        if (kind == "lease")
            category = "lease";
        else if (kind == "range")
            category = "outside";
        expect(refusal.at("writes") == 0 &&
                   refusal.at("reason").get<std::string>().find(category) != std::string::npos,
               "Python real-transaction refusal oracle category " + kind);
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        if (kind == "range")
            hardware.card(bdf("0000:03:00.0")).max_shader = 2600;
        if (kind == "stale") {
            Journal prior;
            prior.run_id = "another";
            prior.stage = Stage::active;
            records.journals.push_back(std::move(prior));
        }
        SessionPower session(
            plan, "typed-refusal", hardware, records, [kind] { return kind != "lease"; }, [] { return true; });
        assert_error(session.begin(), ConfigErrorCode::admission_refused, category);
        expect(hardware.issued.empty(), "same Python refusal category implies zero C++ writes");
    }

    auto hardware = make_hardware(fixture);
    SimulatedRecords records;
    bool children = false;
    SessionPower session(
        plan, "recover-oracle", hardware, records, [] { return true; }, [&] { return children; });
    expect(!session.begin(), "Python recovery-oracle baseline began");
    const auto count = hardware.issued.size();
    assert_error(session.finish(), ConfigErrorCode::admission_refused, "may still be alive");
    expect(hardware.issued.size() == count &&
               bc::serving::power::stage_name(records.journals.front().stage) ==
                   python.at("exit_and_recovery").at("refused_state").get<std::string>(),
           "Python restoration gate retains active journal without writes");
    Json observed = Json::array();
    for (const auto& event : records.journals.front().events) {
        if (event.kind == EventKind::applied)
            observed.push_back("applied");
        else if (event.kind == EventKind::restore_refused)
            observed.push_back("restore_refused");
    }
    expect(observed == python.at("exit_and_recovery").at("refused_events"),
           "Python child-exit refusal event projection");
    children = true;
    expect(!bc::serving::power::recover(
               records.journal_path("recover-oracle"), hardware, records, [] { return true; },
               [&] { return children; }),
           "Python recovery-oracle exact path succeeds");
    expect(originals_preserved(hardware, fixture) &&
               bc::serving::power::stage_name(records.journals.front().stage) ==
                   python.at("exit_and_recovery").at("recovered_state").get<std::string>() &&
               records.receipts.front().stage == Stage::restored,
           "Python recovery journal and manifest receipt reconcile");
    observed = Json::array();
    for (const auto& event : records.journals.front().events) {
        if (event.kind == EventKind::applied)
            observed.push_back("applied");
        else if (event.kind == EventKind::restore_refused)
            observed.push_back("restore_refused");
        else if (event.kind == EventKind::restored)
            observed.push_back("restored");
    }
    expect(observed == python.at("exit_and_recovery").at("recovered_events"),
           "Python successful crash recovery event projection");
}

void owner_lock_contention() {
    Temporary t;
    auto held = bc::serving::power::detail::lock_power_owner(t.root);
    bool refused = false;
    try {
        auto other = bc::serving::power::detail::lock_power_owner(t.root);
    } catch (const std::runtime_error&) {
        refused = true;
    }
    expect(refused, "second owner must not enter while first owns the fence");
    held = bc::serving::power::detail::FileDescriptor{};
    auto next = bc::serving::power::detail::lock_power_owner(t.root);
    expect(bool(next), "lock is released with descriptor");
}
void peer_platform_admission() {
    Temporary t;
    auto fixture = load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto hardware = make_hardware(fixture);
    DiskRecords records(t.root / "power/journals", t.root / "power/receipts");
    const auto peer = t.root / "flashnext/power/journals/platform-run.json";
    Json doc = {{"schema_version", 1}, {"run_id", "platform-run"}, {"state", "restore_failed"},
                {"devices", Json::array()}, {"events", Json::array()}, {"touched", Json::array()}};
    write_text(peer, doc.dump());
    SessionPower blocked(make_plan(fixture), "blocked", hardware, records, [] { return true; }, [] { return true; });
    assert_error(blocked.begin(), ConfigErrorCode::admission_refused, "unresolved power journal");
    expect(hardware.issued.empty(), "platform journal blocks native writes");
    write_text(peer, "invalid JSON");
    expect(!records.unresolved_journals(), "malformed platform journal refuses");
    doc["state"] = "restored";
    write_text(peer, doc.dump());
    SessionPower clean(make_plan(fixture), "clean", hardware, records, [] { return true; }, [] { return true; });
    expect(!clean.begin() && !clean.finish(), "resolved platform record permits native transaction");
}
void real_records_are_safe() {
    Temporary t;
    const auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto hardware = make_hardware(fixture);
    DiskRecords records(t.root / "journals", t.root / "receipts");
    SessionPower session(
        make_plan(fixture), "crash", hardware, records, [] { return true; }, [] { return false; });
    expect(!session.begin(), "persisted crash fixture");
    const auto path = records.journal_path("crash");
    auto journal = load_json(path);
    journal["devices"][0]["before"]["sclk_max_mhz"] = 5000;
    write_text(path, journal.dump(2));
    const auto written_before = hardware.issued.size();
    assert_error(bc::serving::power::recover(
                     path, hardware, records, [] { return true; }, [] { return true; }),
                 ConfigErrorCode::invalid_format, "identity mismatch");
    expect(hardware.issued.size() == written_before,
           "disk journal tamper cannot write out-of-range restored state");
    fs::rename(path, t.root / "journals/real.json");
    fs::create_symlink("real.json", path);
    assert_error(bc::serving::power::recover(
                     path, hardware, records, [] { return true; }, [] { return true; }),
                 ConfigErrorCode::io_failure, "symlinked");
    expect(hardware.issued.size() == written_before, "symlink recovery wrote nothing");
    fs::remove(path);
    fs::rename(t.root / "journals/real.json", path);
    write_text(path, "not valid JSON");
    auto pending = records.unresolved_journals();
    expect(!pending, "malformed unresolved journal blocks future writes");
}

// A receipt is the qualification: it matches only the same BDF, exact settings and board identity of an apply that
// read back exactly and restored exactly. The workload it names is evidence, never a match key.
void receipts_qualify() {
    using bc::serving::power::probe_of;
    using bc::serving::power::qualifying_receipt;
    const auto pci = bdf("0000:03:00.0");
    const auto settings = [](int64_t watts) {
        bc::serving::v2::PowerSettings value;
        value.power_cap_w = watts;
        return value;
    };
    const auto run = [&](const fs::path& root, const std::string& id, int64_t watts, size_t fail_on) {
        SimulatedHardware hardware;
        SimulatedCard card;
        card.pci = pci;
        hardware.cards.push_back(card);
        hardware.fail_on = fail_on;
        PowerPlan plan;
        plan.mode = Mode::apply;
        plan.profile = "p";
        plan.profile_digest = Digest::parse("sha256:" + std::string(64, 'a')).value();
        plan.manifest_digest = Digest::parse("sha256:" + std::string(64, 'b')).value();
        Tuning tuning;
        tuning.board_cap = Watts{watts};
        plan.devices.push_back({{Role{}, pci}, tuning});
        DiskRecords records(root / "journals", root / "receipts");
        SessionPower session(plan, id, hardware, records, [] { return true; }, [] { return true; });
        session.set_workload("some-config");
        const auto began = session.begin();
        if (!began)
            expect(!session.finish(), "restoration after a qualifying apply");
    };
    Temporary t;
    run(t.root, "good", 250, std::numeric_limits<size_t>::max());
    run(t.root, "failed", 260, 0); // the first power write fails: rolled back, never qualifies
    const auto receipts = t.root / "receipts";
    const std::string identity = "kernel=k1;amdgpu=v1;board=744c:1002:0e3b";
    const auto found = qualifying_receipt(receipts, pci, settings(250), identity);
    expect(found && *found == "good", "an exact applied, read back and restored receipt qualifies its setting");
    expect(!qualifying_receipt(receipts, pci, settings(240), identity), "a different setting is not qualified");
    expect(!qualifying_receipt(receipts, pci, settings(260), identity), "a failed apply does not qualify");
    expect(!qualifying_receipt(receipts, bdf("0000:06:00.0"), settings(250), identity), "another BDF is not qualified");
    expect(!qualifying_receipt(receipts, pci, settings(250), "kernel=k2;amdgpu=v1;board=744c:1002:0e3b"),
           "another kernel is not qualified");
    expect(!qualifying_receipt(receipts, pci, settings(250), ""), "an empty identity qualifies nothing");
    auto receipt = load_json(receipts / "good.power.json");
    expect(receipt.at("workload") == "some-config", "the receipt names the workload as evidence");
    const auto clean_receipt = receipt;
    for (const auto* kind : {"restore_failed", "recovery"}) {
        receipt = clean_receipt;
        receipt["transaction"]["events"].push_back({{"kind", kind}, {"bdf", "0000:06:00.0"}});
        write_text(receipts / "good.power.json", receipt.dump());
        expect(!qualifying_receipt(receipts, pci, settings(250), identity),
               "another card failure or recovery disqualifies the whole transaction");
    }
    receipt = clean_receipt;
    receipt["observations"]["stop"]["cards"][0]["identity"] = "replacement board";
    write_text(receipts / "good.power.json", receipt.dump());
    expect(!qualifying_receipt(receipts, pci, settings(250), identity), "stop identity must match");
    receipt = clean_receipt;
    receipt["observations"]["stop"]["cards"][0]["power_cap_uw"] = 1;
    write_text(receipts / "good.power.json", receipt.dump());
    expect(!qualifying_receipt(receipts, pci, settings(250), identity), "an inexact restore does not qualify");
    SimulatedHardware hardware;
    SimulatedCard card;
    card.pci = pci;
    hardware.cards.push_back(card);
    const auto probe = probe_of(hardware.inspect(pci).value());
    expect(probe.bdf == pci.value && probe.driver == "amdgpu" && probe.controls.at("power_cap_w").supported &&
               probe.controls.at("power_cap_w").minimum == 200000000 &&
               probe.controls.at("power_cap_w").original_readable,
           "the probe is the transaction's own capability read");
}

void recovery_binding_disk() {
    Temporary t;
    const auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto hardware = make_hardware(fixture);
    DiskRecords records(t.root / "journals", t.root / "receipts");
    SessionPower session(
        make_plan(fixture), "manifest-bound", hardware, records, [] { return true; }, [] { return false; });
    expect(!session.begin(), "create a real durable manifest-bound journal and receipt");
    const auto journal = records.journal_path("manifest-bound");
    const auto receipt_path = t.root / "receipts/manifest-bound.power.json";
    const auto original_receipt = load_json(receipt_path);
    auto replaced = original_receipt;
    replaced["manifest"]["content_digest"] = "sha256:" + std::string(64U, '3');
    write_text(receipt_path, replaced.dump(2));
    const auto count = hardware.issued.size();
    assert_error(bc::serving::power::recover(
                     journal, hardware, records, [] { return true; }, [] { return true; }),
                 ConfigErrorCode::invalid_format, "identity mismatch");
    expect(load_json(journal).at("state") == "applied",
           "forged receipt binding cannot mark recovery journal resolved");
    expect(hardware.issued.size() == count,
           "receipt tamper identified before restoration writes");
    write_text(receipt_path, original_receipt.dump(2));
    // Old absolute-clock journals had no identity; the original receipt,
    // not a fresh probe, must supply it for recovery.
    auto legacy = load_json(journal);
    for (auto& dev : legacy["devices"]) dev.erase("identity");
    write_text(journal, legacy.dump(2));
    auto loaded = records.load(journal);
    expect(bool(loaded) && !loaded.value().devices.front().before.identity.empty(),
           "legacy journal binds identity from original receipt");
    auto& card = hardware.card(loaded.value().devices.front().policy.device.pci);
    const auto identity = card.identity;
    card.identity = "replacement board";
    assert_error(bc::serving::power::recover(journal, hardware, records,
                 [] { return true; }, [] { return true; }),
                 ConfigErrorCode::admission_refused, "physical identity changed");
    expect(hardware.issued.size() == count, "replacement board gets no writes");
    card.identity = identity;
    expect(!bc::serving::power::recover(
               journal, hardware, records, [] { return true; }, [] { return true; }),
           "recovery with the original manifest receipt completes");
    const auto finished = load_json(receipt_path);
    expect(finished.at("transaction").at("state") == "restored" &&
               finished.at("observations").at("stop").at("status") == "restored",
           "disk recovery restores prior receipt under authenticated manifest identity");
}
void diagnostics() {
    const auto start = std::chrono::system_clock::time_point{};
    const auto floor = [&](uint8_t busy = 0) {
        Window w;
        w.pci = bdf("0000:03:00.0");
        w.phase = WindowPhase::no_process;
        w.identity = "driver-v1";
        w.sensor_identity = "sensor-v1";
        w.acquisition = CaptureState::measured;
        for (int i = 0; i < 3; ++i)
            w.samples.push_back({bc::serving::power::Microwatts{22000000}, Millivolts{750}, busy,
                                 start + std::chrono::seconds(i)});
        return w;
    };
    const auto idle = [&](int64_t power, int64_t voltage, uint8_t busy) {
        Window w = floor();
        w.phase = WindowPhase::engine_loaded_idle;
        w.samples.clear();
        for (int i = 0; i < 3; ++i)
            w.samples.push_back({bc::serving::power::Microwatts{power}, Millivolts{voltage}, busy,
                                 start + std::chrono::seconds(10 + i)});
        return w;
    };
    expect(bc::serving::power::evaluate(floor(), idle(22000000, 750, 0)).state == GfxoffState::active,
           "floor-matched GFXOFF state");
    expect(bc::serving::power::evaluate(floor(), idle(28000000, 800, 2)).state == GfxoffState::absent,
           "all three floor signals elevated");
    auto altered = idle(28000000, 800, 2);
    altered.identity = "changed-driver";
    const Verdict diff = bc::serving::power::evaluate(floor(), altered);
    expect(diff.state == GfxoffState::ambiguous && diff.reason == GfxoffReason::identity_mismatch,
           "changed identity ambiguous");
    expect(bc::serving::power::evaluate(std::nullopt, altered).state == GfxoffState::needs_idle_context,
           "missing no-process floor");
    expect(bc::serving::power::evaluate(floor(1), idle(28000000, 800, 2)).reason ==
               GfxoffReason::floor_not_idle,
           "busy floor cannot establish GFXOFF");
    Temporary t;
    const auto device = t.root / "sys/bus/pci/devices/0000:03:00.0";
    write_text(device / "ip_discovery/die/0/GC/0/major", "12\n");
    write_text(device / "device", "0x744c\n");
    write_text(t.root / "sys/module/amdgpu/srcversion", "fake-driver\n");
    write_text(device / "hwmon/hwmon2/name", "amdgpu\n");
    write_text(device / "hwmon/hwmon2/power1_average", "22000000\n");
    write_text(device / "hwmon/hwmon2/in0_input", "750\n");
    write_text(device / "gpu_busy_percent", "0\n");
    LinuxSensor sensor(t.root / "sys", [] {});
    auto sample = sensor.measure(bdf("0000:03:00.0"), WindowPhase::no_process, 3U);
    expect(bool(sample) && sample.value().acquisition == CaptureState::measured &&
               sample.value().samples.size() == 3U,
           "read-only sysfs gfx12 sampling");
    for (const bool foreign : {false, true}) {
        LinuxSensor broken(t.root / "sys", [foreign] {
            if (foreign)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("interval exception");
        });
        const auto measured = broken.measure(bdf("0000:03:00.0"), WindowPhase::no_process, 3U);
        expect(!measured && measured.error().code == ConfigErrorCode::io_failure,
               "throwing interval becomes typed failure");
    }
    write_text(device / "ip_discovery/die/0/GC/0/major", "11\n");
    sample = sensor.measure(bdf("0000:03:00.0"), WindowPhase::no_process, 3U);
    expect(bool(sample) && sample.value().acquisition == CaptureState::not_applicable,
           "GFXOFF not applicable on non gfx12 card");
    write_text(device / "pp_od_clk_voltage", "OD_SCLK:\n0: 500Mhz\n1: 2600Mhz\nOD_MCLK:\n0: 97Mhz\n1: 1300Mhz\n"
                                             "OD_VDDGFX_OFFSET:\n-100mV\nOD_RANGE:\nSCLK: 500Mhz 3200Mhz\n"
                                             "MCLK: 97Mhz 1500Mhz\nVDDGFX_OFFSET: -200mV 0mV\n");
    write_text(device / "hwmon/hwmon2/power1_cap", "300000000\n");
    write_text(device / "hwmon/hwmon2/power1_cap_min", "200000000\n");
    write_text(device / "hwmon/hwmon2/power1_cap_max", "350000000\n");
    fs::create_symlink("/drivers/amdgpu", device / "driver");
    LinuxHardware adapter(t.root);
    auto snapshot = adapter.inspect(bdf("0000:03:00.0"));
    expect(bool(snapshot) && snapshot.value().amdgpu &&
               snapshot.value().at(ControlId::shader).current == 2600 &&
               snapshot.value().at(ControlId::memory).maximum == 1500,
           "Linux OverDrive probe uses hardware-reported controls");
    write_text(device / "pp_od_clk_voltage",
               "OD_SCLK_OFFSET:\n-25Mhz\nOD_MCLK:\n0: 97Mhz\n1: 1259MHz\n"
               "OD_RANGE:\nSCLK_OFFSET: -500Mhz 1000Mhz\nMCLK: 97Mhz 1500Mhz\n"
               "VDDGFX_OFFSET: -200mv 0mv\nOD_VDDGFX_OFFSET:\n-10mV\n");
    snapshot = adapter.inspect(bdf("0000:03:00.0"));
    expect(bool(snapshot) && !snapshot.value().at(ControlId::shader).current &&
           snapshot.value().at(ControlId::shader_offset).current == -25 &&
           snapshot.value().at(ControlId::shader_offset).minimum == -500,
           "signed OverDrive table and range parsed independently of ceiling");
    fs::remove_all(device / "hwmon");
    fs::create_symlink("hwmon", device / "hwmon");
    auto invalid_monitor = adapter.inspect(bdf("0000:03:00.0"));
    expect(!invalid_monitor && invalid_monitor.error().code == ConfigErrorCode::io_failure,
           "Linux hwmon enumeration error must be returned as a typed IO failure");
    auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    auto fake = make_hardware(fixture);
    auto plan = make_plan(fixture);
    auto issues = bc::serving::power::inspect_host(plan, fake, std::span<const fs::path>{},
                                                   [](const fs::path&) { return true; });
    expect(issues.empty(), "healthy selected cards satisfy host check");
    for (const bool foreign : {false, true}) {
        const std::array<fs::path, 1> rapl{{"/fake/energy_uj"}};
        auto read_fake = make_hardware(fixture);
        issues = bc::serving::power::inspect_host(plan, read_fake, rapl, [foreign](const fs::path&) -> bool {
            if (foreign)
                throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw
            throw std::runtime_error("readability exception");
        });
        expect(!issues.empty() && issues.back().code == HostIssue::cpu_energy_unreadable,
               "throwing readability becomes typed host finding");
        auto probe_fake = make_hardware(fixture);
        if (foreign)
            probe_fake.throw_unknown_inspect_on = 0;
        else
            probe_fake.throw_inspect_on = 0;
        issues = bc::serving::power::inspect_host(plan, probe_fake, std::span<const fs::path>{},
                                                  [](const fs::path&) { return true; });
        expect(!issues.empty() && issues.front().code == HostIssue::probe_failed,
               "throwing Hardware::inspect becomes typed host finding");
    }
    fake.card(bdf("0000:06:00.0")).amdgpu = false;
    issues = bc::serving::power::inspect_host(plan, fake, std::span<const fs::path>{},
                                              [](const fs::path&) { return true; });
    expect(!issues.empty() && issues.front().code == HostIssue::missing_backend,
           "host-check gives typed missing-backend finding");
}

// One-shot throws are deliberate: a valid lease exists on the next check, so
// failure to roll back cannot be blamed on a permanently missing lease.
void predicate_exception_matrix() {
    const auto fixture =
        load_json(fs::path(__FILE__).parent_path() / "testdata/power_qualified_inputs.json").at("cases").at(0);
    const auto plan = make_plan(fixture);
    size_t apply_checks = 0;
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        SessionPower session(
            plan, "count-apply", hardware, records,
            [&] {
                ++apply_checks;
                return true;
            },
            [] { return true; });
        expect(!session.begin(), "count baseline apply lease gates");
    }
    expect(apply_checks > 9U, "exercise both preflight and per-command lease call sites");
    for (size_t index = 0; index < apply_checks; ++index) {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        size_t seen = 0;
        SessionPower session(
            plan, "throw-apply", hardware, records,
            [&]() -> bool {
                if (seen++ == index) {
                    if (index % 2U == 0U)
                        throw std::runtime_error("one-shot lease exception");
                    throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard throw //
                              // NOLINT(bugprone-std-exception-baseclass): exercise catch(...)
                }
                return true;
            },
            [] { return true; });
        const auto fault = session.begin();
        assert_error(fault, ConfigErrorCode::io_failure, "predicate threw");
        expect(originals_preserved(hardware, fixture),
               "apply lease check " + std::to_string(index) + " restores every original");
        if (hardware.execute_calls != 0U) {
            expect(session.stage() == Stage::rolled_back && !records.journals.empty() &&
                       records.journals.back().stage == Stage::rolled_back && !records.receipts.empty() &&
                       records.receipts.back().stage == Stage::rolled_back,
                   "after-write predicate exception must roll back and reconcile durable records");
        } else if (!records.journals.empty()) {
            expect(bc::serving::power::unresolved(records.journals.back().stage),
                   "before-write predicate failure leaves prepared record unresolved");
        }
    }

    size_t finish_checks = 0;
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        bool finishing = false;
        SessionPower session(
            plan, "count-finish", hardware, records,
            [&] {
                if (finishing)
                    ++finish_checks;
                return true;
            },
            [] { return true; });
        expect(!session.begin(), "count finish baseline start");
        finishing = true;
        expect(!session.finish(), "count finish baseline stop");
    }
    expect(finish_checks > 2U, "exercise both finish lease and restore command gates");
    for (size_t index = 0; index < finish_checks; ++index) {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        bool finishing = false;
        size_t seen = 0;
        SessionPower session(
            plan, "throw-finish", hardware, records,
            [&]() -> bool {
                if (finishing && seen++ == index)
                    throw std::runtime_error("one-shot finish lease exception");
                return true;
            },
            [] { return true; });
        expect(!session.begin(), "finish lease fixture start");
        finishing = true;
        assert_error(session.finish(), ConfigErrorCode::io_failure, "predicate threw");
        expect(!records.journals.empty() && bc::serving::power::unresolved(records.journals.back().stage),
               "finish lease exception must leave unresolved journal");
        expect(!session.finish(), "repeat finish after transient lease exception");
        expect(originals_preserved(hardware, fixture), "retry finish restores every original");
        expect(records.journals.back().stage == Stage::restored &&
                   records.receipts.back().stage == Stage::restored,
               "retry finish reconciles journal and receipt");
    }

    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        bool thrown = false;
        SessionPower session(
            plan, "throw-child", hardware, records, [] { return true; },
            [&]() -> bool {
                if (!thrown) {
                    thrown = true;
                    throw std::runtime_error("one-shot child-exit exception");
                }
                return true;
            });
        expect(!session.begin(), "child callback start");
        const size_t before = hardware.execute_calls;
        assert_error(session.finish(), ConfigErrorCode::io_failure, "predicate threw");
        expect(hardware.execute_calls == before && records.journals.back().stage == Stage::active,
               "child-exit exception refuses writes, retains active durable journal");
        expect(!session.finish() && originals_preserved(hardware, fixture),
               "child-exit retry restores hardware");
        expect(records.journals.back().stage == Stage::restored &&
                   records.receipts.back().stage == Stage::restored,
               "child-exit retry reconciles records");
    }

    // Enumerate lease checks reached specifically during failure rollback.
    // The fifth command faults; each later check belongs to restore_touched.
    constexpr size_t fault_command = 4U;
    size_t rollback_checks = 0;
    {
        auto hardware = make_hardware(fixture);
        hardware.fail_on = fault_command;
        SimulatedRecords records;
        SessionPower session(
            plan, "count-rollback", hardware, records,
            [&] {
                if (hardware.execute_calls > fault_command)
                    ++rollback_checks;
                return true;
            },
            [] { return true; });
        assert_error(session.begin(), ConfigErrorCode::io_failure, "injected");
        expect(originals_preserved(hardware, fixture), "baseline rollback restored originals");
    }
    expect(rollback_checks > 2U, "enumerate rollback precheck and command lease gates");
    for (size_t index = 0; index < rollback_checks; ++index) {
        auto hardware = make_hardware(fixture);
        hardware.fail_on = fault_command;
        SimulatedRecords records;
        size_t seen = 0;
        SessionPower session(
            plan, "throw-rollback", hardware, records,
            [&]() -> bool {
                if (hardware.execute_calls > fault_command && seen++ == index)
                    throw std::runtime_error("one-shot rollback lease exception");
                return true;
            },
            [] { return true; });
        assert_error(session.begin(), ConfigErrorCode::io_failure, "injected");
        expect(!records.journals.empty() && records.journals.back().stage == Stage::rollback_failed,
               "rollback lease failure retains unresolved durable journal");
        expect(!bc::serving::power::recover(
                   records.journal_path("throw-rollback"), hardware, records, [] { return true; },
                   [] { return true; }),
               "successful recovery after rollback predicate fault");
        expect(originals_preserved(hardware, fixture), "recovery after rollback fault restores originals");
        expect(records.journals.back().stage == Stage::restored &&
                   records.receipts.back().stage == Stage::restored,
               "recovery after rollback fault reconciles records");
    }

    size_t recovery_checks = 0;
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        SessionPower session(
            plan, "count-recovery", hardware, records, [] { return true; }, [] { return true; });
        expect(!session.begin(), "count recovery fixture");
        expect(!bc::serving::power::recover(
                   records.journal_path("count-recovery"), hardware, records,
                   [&] {
                       ++recovery_checks;
                       return true;
                   },
                   [] { return true; }),
               "count baseline recovery");
        expect(originals_preserved(hardware, fixture), "baseline recover returns originals");
    }
    expect(recovery_checks > 2U, "exercise recovery initial gate and per-command checks");
    for (size_t index = 0; index < recovery_checks; ++index) {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        SessionPower session(
            plan, "throw-recovery", hardware, records, [] { return true; }, [] { return true; });
        expect(!session.begin(), "recovery exception fixture start");
        size_t seen = 0;
        const auto fault = bc::serving::power::recover(
            records.journal_path("throw-recovery"), hardware, records,
            [&]() -> bool {
                if (seen++ == index)
                    throw std::runtime_error("one-shot recovery lease exception");
                return true;
            },
            [] { return true; });
        assert_error(fault, ConfigErrorCode::io_failure, "predicate threw");
        expect(bc::serving::power::unresolved(records.journals.back().stage),
               "recovery exception preserves unresolved durable journal");
        expect(!bc::serving::power::recover(
                   records.journal_path("throw-recovery"), hardware, records, [] { return true; },
                   [] { return true; }),
               "retry recovery after transient lease exception");
        expect(originals_preserved(hardware, fixture),
               "retry after recovery lease exception restores every original");
        expect(records.journals.back().stage == Stage::restored &&
                   records.receipts.back().stage == Stage::restored,
               "recovery retry reconciles durable records");
    }
    {
        auto hardware = make_hardware(fixture);
        SimulatedRecords records;
        SessionPower session(
            plan, "recover-child", hardware, records, [] { return true; }, [] { return true; });
        expect(!session.begin(), "recovery child fixture start");
        const size_t before = hardware.execute_calls;
        assert_error(bc::serving::power::recover(
                         records.journal_path("recover-child"), hardware, records, [] { return true; },
                         []() -> bool { throw std::runtime_error("recovery child-exit exception"); }),
                     ConfigErrorCode::io_failure, "predicate threw");
        expect(hardware.execute_calls == before && records.journals.back().stage == Stage::active,
               "recovery child exception blocks any restore and retains journal");
        expect(!bc::serving::power::recover(
                   records.journal_path("recover-child"), hardware, records, [] { return true; },
                   [] { return true; }),
               "recovery child-exit retry");
        expect(originals_preserved(hardware, fixture) && records.journals.back().stage == Stage::restored &&
                   records.receipts.back().stage == Stage::restored,
               "recovery child-exit retry restores originals and records");
    }
}

} // namespace
int main() {
    try {
        golden_parity();
        binding_and_preflight();
        signed_offset_transaction();
        failures_and_recovery();
        predicate_exception_matrix();
        injected_interface_exception_matrix();
        python_refusal_and_recovery_parity();
        real_records_are_safe();
        peer_platform_admission();
        owner_lock_contention();
        recovery_binding_disk();
        receipts_qualify();
        diagnostics();
        std::cout << "bc_power_transaction_test: 5 groups passed "
                     "(4 Python golden profiles, crash/recovery, hardware fakes)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bc_power_transaction_test: " << error.what() << "\n";
        return 1;
    }
}
