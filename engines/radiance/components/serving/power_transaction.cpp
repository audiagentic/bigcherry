#include "power_transaction.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <set>
#include <utility>

namespace bc::serving::power {
namespace {

using Code = v2::ConfigErrorCode;
using Status = v2::Status;

v2::ConfigError invalid(std::string path, std::string message) {
    return v2::make_error(Code::invalid_value, std::move(path), std::move(message));
}
v2::ConfigError unsafe(std::string path, std::string message) {
    return v2::make_error(Code::admission_refused, std::move(path), std::move(message));
}
v2::ConfigError state_error(std::string message) {
    return v2::make_error(Code::invalid_state, "power.transaction", std::move(message));
}
constexpr int64_t kIjson = 9007199254740991LL;
constexpr int64_t kMicrowattsPerWatt = 1000000LL;

size_t slot(ControlId control) {
    return static_cast<size_t>(control);
}

bool in_range(const Capability& cap, int64_t value) {
    return cap.writable && cap.current && cap.minimum && cap.maximum && *cap.minimum <= value &&
           value <= *cap.maximum;
}

Status original_can_restore(const SavedDevice& saved, const CardSnapshot& current) {
    const auto& before = saved.before;
    const auto& tuning = saved.policy.tuning;
    const auto bdf = saved.policy.device.pci.value;
    if (current.pci != saved.policy.device.pci || !current.amdgpu || current.hwmon != before.hwmon) {
        return unsafe("power.devices", bdf + ": controller identity changed");
    }
    const auto check = [&](ControlId id, std::string_view name) -> Status {
        const auto& old = before.at(id).current;
        if (!old || !in_range(current.at(id), *old))
            return unsafe("power.devices", bdf + ": original " + std::string(name) +
                                               " cannot be safely restored within device range");
        return std::nullopt;
    };
    const bool signed_shader = before.at(ControlId::shader_offset).current.has_value();
    const bool live_signed = current.at(ControlId::shader_offset).current.has_value();
    if (signed_shader && (before.identity.empty() || current.identity != before.identity))
        return unsafe("power.devices", bdf + ": physical identity changed");
    if (signed_shader != live_signed)
        return unsafe("power.devices", bdf + ": original shader offset missing or changed control kind");
    // An OverDrive commit is coupled: SCLK and the voltage offset must both
    // be restored even when a profile requests only one of them.
    if (tuning.changes_overdrive()) {
        if (auto error = check(ControlId::voltage, "voltage offset"))
            return error;
        const bool absolute = before.at(ControlId::shader).current.has_value();
        if (absolute == signed_shader || (signed_shader && tuning.shader_ceiling) ||
            (!signed_shader && tuning.shader_offset) ||
            current.at(ControlId::shader).current.has_value() != absolute)
            return unsafe("power.devices", bdf + ": incompatible or ambiguous shader control kind");
        if (auto error = check(signed_shader ? ControlId::shader_offset : ControlId::shader,
                               signed_shader ? "shader offset" : "shader clock"))
            return error;
    }
    if (tuning.memory_ceiling || (signed_shader && tuning.changes_overdrive())) {
        if (auto error = check(ControlId::memory, "memory clock"))
            return error;
    }
    if (tuning.board_cap) {
        if (auto error = check(ControlId::board_cap, "board cap"))
            return error;
    }
    return std::nullopt;
}

Status verify_readback(const SavedDevice& saved, const CardSnapshot& current, bool restoring) {
    const auto& tuning = saved.policy.tuning;
    const auto& original = saved.before;
    if (current.pci != original.pci || current.amdgpu != original.amdgpu || current.hwmon != original.hwmon)
        return unsafe("power.readback", original.pci.value + ": controller identity changed");
    // Signed-shader cards save and compare the complete represented state,
    // even controls not requested. Do not confuse missing readback with zero.
    if (original.at(ControlId::shader_offset).current) {
        if (original.identity.empty() || current.identity != original.identity)
            return unsafe("power.readback", original.pci.value + ": physical identity changed");
        for (size_t index = 0; index < control_count; ++index) {
            const auto id = static_cast<ControlId>(index);
            auto desired = original.at(id).current;
            if (!restoring) {
                if (id == ControlId::voltage && tuning.voltage_offset)
                    desired = tuning.voltage_offset->value;
                if (id == ControlId::shader_offset && tuning.shader_offset)
                    desired = tuning.shader_offset->value;
                if (id == ControlId::memory && tuning.memory_ceiling)
                    desired = tuning.memory_ceiling->value;
                if (id == ControlId::board_cap && tuning.board_cap)
                    desired = tuning.board_cap->value * kMicrowattsPerWatt;
            }
            if (current.at(id).current != desired)
                return unsafe("power.readback", original.pci.value + ": signed-card control readback mismatch");
        }
        return std::nullopt;
    }
    const auto check = [&](ControlId id, std::optional<int64_t> desired, std::string_view label) -> Status {
        if (desired && current.at(id).current != desired)
            return unsafe("power.readback",
                          original.pci.value + ": " + std::string(label) + " readback mismatch");
        return std::nullopt;
    };
    if (tuning.changes_overdrive()) {
        std::optional<int64_t> offset = original.at(ControlId::voltage).current;
        if (!restoring && tuning.voltage_offset)
            offset = tuning.voltage_offset->value;
        if (auto error = check(ControlId::voltage, offset, "voltage offset"))
            return error;
    }
    if (current.at(ControlId::shader_offset).current)
        return unsafe("power.readback", original.pci.value + ": shader offset readback mismatch");
    if (tuning.shader_ceiling || tuning.voltage_offset) {
        auto shader = original.at(ControlId::shader).current;
        if (!restoring && tuning.shader_ceiling)
            shader = tuning.shader_ceiling->value;
        if (auto error = check(ControlId::shader, shader, "shader clock"))
            return error;
    }
    if (tuning.memory_ceiling) {
        auto memory = original.at(ControlId::memory).current;
        if (!restoring && tuning.memory_ceiling)
            memory = tuning.memory_ceiling->value;
        if (auto error = check(ControlId::memory, memory, "memory clock"))
            return error;
    }
    if (tuning.board_cap) {
        auto board = original.at(ControlId::board_cap).current;
        if (!restoring && tuning.board_cap)
            board = tuning.board_cap->value * kMicrowattsPerWatt;
        if (auto error = check(ControlId::board_cap, board, "board cap"))
            return error;
    }
    return std::nullopt;
}

v2::ConfigError backend_exception(std::string_view operation, std::string_view message) {
    return v2::make_error(Code::io_failure, "power.hardware",
                          std::string(operation) + " threw: " + std::string(message));
}

v2::Result<CardSnapshot> probe(Hardware& hardware, const PciAddress& pci) {
    return hardware.inspect(pci);
}

Status issue(Hardware& hardware, const PowerCommand& command) {
    return hardware.execute(command);
}

v2::Result<Journal> validate_recovery(Journal record, Hardware& hardware) {
    if (!unresolved(record.stage))
        return state_error("power record is not unresolved");
    if (record.stage == Stage::active && record.touched.size() != record.devices.size())
        return unsafe("power.recovery", "active journal has incomplete touched inventory");
    std::set<std::string> available, touched;
    for (const auto& saved : record.devices) {
        if (!available.insert(saved.policy.device.pci.value).second)
            return invalid("power.recovery", "duplicate journal device");
        if (!saved.policy.tuning.changes_hardware())
            return invalid("power.recovery", "journal contains device with no writes");
    }
    for (const auto& pci : record.touched) {
        if (!available.contains(pci.value) || !touched.insert(pci.value).second)
            return invalid("power.recovery", "journal touched list differs from devices");
    }
    // Crucial: inspect *all* touched cards before issuing even the first restore.
    for (const auto& saved : record.devices) {
        if (!touched.contains(saved.policy.device.pci.value))
            continue;
        auto live = probe(hardware, saved.policy.device.pci);
        if (!live)
            return live.error();
        if (auto failure = original_can_restore(saved, live.value()))
            return *failure;
    }
    return record;
}

} // namespace

v2::Result<PciAddress> PciAddress::parse(std::string_view text) {
    if (text.size() != 12 || text[4] != ':' || text[7] != ':' || text[10] != '.' || text[11] < '0' ||
        text[11] > '7') {
        return invalid("power.devices.bdf", "invalid PCI BDF");
    }
    for (size_t i = 0; i < text.size(); ++i) {
        if (i == 4 || i == 7 || i == 10 || i == 11)
            continue;
        const char ch = text[i];
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
            return invalid("power.devices.bdf", "invalid lowercase PCI BDF");
    }
    return PciAddress{std::string(text)};
}
v2::Result<Digest> Digest::parse(std::string_view text) {
    if (text.size() != 71 || !text.starts_with("sha256:"))
        return invalid("power.profile_digest", "invalid sha256 digest");
    for (auto ch : text.substr(7))
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
            return invalid("power.profile_digest", "invalid lowercase sha256 digest");
    return Digest{std::string(text)};
}
std::string Role::label() const {
    return kind == RoleKind::engine ? "engine" : "helper" + std::to_string(helper_index);
}
v2::Result<Role> Role::parse(std::string_view text) {
    if (text == "engine")
        return Role{};
    if (!text.starts_with("helper") || text.size() < 7)
        return invalid("power.devices.role", "invalid GPU role");
    uint64_t number = 0;
    for (char ch : text.substr(6)) {
        if (ch < '0' || ch > '9')
            return invalid("power.devices.role", "invalid helper index");
        const auto digit = static_cast<uint64_t>(ch - '0');
        if (number > (std::numeric_limits<uint32_t>::max() - digit) / 10U)
            return invalid("power.devices.role", "helper index overflow");
        number = (number * 10U) + digit;
    }
    if (text.size() > 7 && text[6] == '0')
        return invalid("power.devices.role", "noncanonical helper index");
    return Role{RoleKind::helper, static_cast<uint32_t>(number)};
}
bool Tuning::changes_overdrive() const {
    return voltage_offset || shader_ceiling || memory_ceiling || shader_offset;
}
bool Tuning::changes_hardware() const {
    return changes_overdrive() || board_cap;
}
const Capability& CardSnapshot::at(ControlId id) const {
    return controls.at(slot(id));
}

v2::Result<PowerPlan> bind(const v2::Manifest& verified, std::span<const SelectedGpu> selected) {
    const auto& input = verified.spec.plan.power;
    const auto& intent = verified.spec.intent.power;
    PowerPlan result;
    result.mode = input.mode == v2::PowerMode::kObserve ? Mode::observe : Mode::apply;
    result.profile = input.profile;
    if (result.profile != intent.profile || result.profile.empty())
        return invalid("spec.plan.power.profile", "resolved power profile differs from intent");
    auto profile = Digest::parse(input.profile_digest);
    if (!profile)
        return profile.error();
    auto manifest = Digest::parse(verified.metadata.content_digest);
    if (!manifest)
        return manifest.error();
    result.profile_digest = profile.value();
    result.manifest_digest = manifest.value();
    if (selected.empty() || selected.size() != input.devices.size())
        return unsafe("spec.plan.power.devices", "power devices differ from serving inventory");
    std::set<std::string> distinct;
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto& source = input.devices[i];
        auto role = Role::parse(source.role);
        auto pci = PciAddress::parse(source.bdf);
        if (!role)
            return role.error();
        if (!pci)
            return pci.error();
        if (role.value() != selected[i].role || pci.value() != selected[i].pci ||
            !distinct.insert(pci.value().value).second)
            return unsafe("spec.plan.power.devices", "role, BDF or order differs from selected inventory");
        const auto& s = source.settings;
        if (s.sclk_offset_mhz && s.sclk_max_mhz)
            return invalid("spec.plan.power.devices.settings.sclk_offset_mhz", "mutually exclusive shader controls");
        Tuning tuning;
        if (s.vddgfx_offset_mv)
            tuning.voltage_offset = Millivolts{*s.vddgfx_offset_mv};
        if (s.sclk_max_mhz)
            tuning.shader_ceiling = Megahertz{*s.sclk_max_mhz};
        if (s.sclk_offset_mhz)
            tuning.shader_offset = Megahertz{*s.sclk_offset_mhz};
        if (s.mclk_max_mhz)
            tuning.memory_ceiling = Megahertz{*s.mclk_max_mhz};
        if (s.power_cap_w)
            tuning.board_cap = Watts{*s.power_cap_w};
        if (result.mode == Mode::observe && tuning.changes_hardware())
            return invalid("spec.plan.power.devices.settings", "observe cannot request hardware writes");
        result.devices.push_back({selected[i], tuning});
    }
    return result;
}

v2::Result<std::vector<PowerCommand>> commands_for(const DevicePolicy& policy, const CardSnapshot& before,
                                                   bool restoring) {
    const auto& tuning = policy.tuning;
    const auto& pci = policy.device.pci;
    std::vector<PowerCommand> out;
    const auto issue = [&](CommandValue value) { out.push_back(PowerCommand{pci, before.hwmon, value}); };
    const auto original = [&](ControlId kind) -> v2::Result<int64_t> {
        const auto& measured = before.at(kind).current;
        if (!measured)
            return unsafe("power.original", pci.value + ": required original control is unreadable");
        return measured.value_or(0);
    };
    const bool signed_shader = before.at(ControlId::shader_offset).current.has_value();
    if (tuning.changes_overdrive() &&
        (signed_shader == before.at(ControlId::shader).current.has_value() ||
         (signed_shader && tuning.shader_ceiling) || (!signed_shader && tuning.shader_offset)))
        return unsafe("power.original", pci.value + ": incompatible or ambiguous shader control");
    if (signed_shader && tuning.changes_overdrive()) {
        // RDNA4 uses a signed one-argument shader command. Stage all
        // represented coupled controls to restore an interrupted commit too.
        auto shader = original(ControlId::shader_offset);
        auto memory = original(ControlId::memory);
        auto voltage = original(ControlId::voltage);
        if (!shader) return shader.error();
        if (!memory) return memory.error();
        if (!voltage) return voltage.error();
        issue(ShaderOffset{Megahertz{!restoring && tuning.shader_offset ?
            tuning.shader_offset->value : shader.value()}});
        issue(MemoryClock{Megahertz{!restoring && tuning.memory_ceiling ?
            tuning.memory_ceiling->value : memory.value()}});
        issue(VoltageOffset{Millivolts{!restoring && tuning.voltage_offset ?
            tuning.voltage_offset->value : voltage.value()}});
    } else if (restoring) {
        // An OverDrive commit couples the original shader row and VDDGFX.
        if (tuning.shader_ceiling || tuning.voltage_offset) {
            auto previous = original(ControlId::shader);
            if (!previous)
                return previous.error();
            issue(ShaderClock{Megahertz{previous.value()}});
        }
        if (tuning.memory_ceiling) {
            auto previous = original(ControlId::memory);
            if (!previous)
                return previous.error();
            issue(MemoryClock{Megahertz{previous.value()}});
        }
        if (tuning.changes_overdrive()) {
            auto previous = original(ControlId::voltage);
            if (!previous)
                return previous.error();
            issue(VoltageOffset{Millivolts{previous.value()}});
        }
    } else {
        if (tuning.shader_ceiling)
            issue(ShaderClock{*tuning.shader_ceiling});
        if (tuning.memory_ceiling)
            issue(MemoryClock{*tuning.memory_ceiling});
        if (tuning.voltage_offset) {
            issue(VoltageOffset{*tuning.voltage_offset});
        } else if (tuning.shader_ceiling || tuning.memory_ceiling) {
            auto previous = original(ControlId::voltage);
            if (!previous)
                return previous.error();
            issue(VoltageOffset{Millivolts{previous.value()}});
        }
    }
    if (tuning.changes_overdrive())
        issue(OverdriveCommit{});
    if (tuning.board_cap) {
        if (restoring) {
            auto previous = original(ControlId::board_cap);
            if (!previous)
                return previous.error();
            issue(BoardCap{Microwatts{previous.value()}});
        } else {
            if (tuning.board_cap->value <= 0 || tuning.board_cap->value > kIjson / kMicrowattsPerWatt)
                return invalid("power.board_cap", "board cap exceeds safely convertible range");
            issue(BoardCap{Microwatts{tuning.board_cap->value * kMicrowattsPerWatt}});
        }
    }
    return out;
}

constexpr std::array<std::pair<Stage, std::string_view>, 10> kStageLabels{
    {{Stage::fresh, "new"},
     {Stage::prepared, "prepared"},
     {Stage::changing, "applying"},
     {Stage::active, "applied"},
     {Stage::rolled_back, "rolled_back"},
     {Stage::rollback_failed, "rollback_failed"},
     {Stage::restored, "restored"},
     {Stage::restore_failed, "restore_failed"},
     {Stage::observe, "observe"},
     {Stage::unchanged, "apply_empty"}}};
std::string_view stage_name(Stage stage) {
    for (auto [candidate, label] : kStageLabels)
        if (candidate == stage)
            return label;
    return "invalid";
}
v2::Result<Stage> stage_from_name(std::string_view name) {
    for (auto [candidate, label] : kStageLabels)
        if (name == label)
            return candidate;
    return invalid("power.journal.stage", "unknown power transaction stage");
}
bool unresolved(Stage stage) {
    return stage == Stage::changing || stage == Stage::active || stage == Stage::rollback_failed ||
           stage == Stage::restore_failed;
}

v2::Status SessionPower::checked(const Predicate& predicate, std::string_view path, std::string_view refused) {
    const auto result = predicate.invoke(path);
    if (!result) {
        auto failure = result.error();
        failure.message = "predicate threw: " + failure.message;
        return failure;
    }
    if (result.value())
        return std::nullopt;
    return unsafe(std::string(path), std::string(refused));
}

SessionPower::SessionPower(PowerPlan plan, std::string run_id, Hardware& hardware, Records& records,
                           Predicate owns_gpu_lease, Predicate children_exited)
    : plan_(std::move(plan)), hardware_(hardware), records_(records), lease_(std::move(owns_gpu_lease)),
      children_exited_(std::move(children_exited)) {
    journal_.run_id = run_id;
    journal_.manifest_digest = plan_.manifest_digest;
    journal_.profile_digest = plan_.profile_digest;
    receipt_.run_id = std::move(run_id);
    receipt_.plan = plan_;
}
void SessionPower::record(EventKind kind, std::optional<PciAddress> pci, std::string reason) {
    Event event{kind, std::move(pci), std::move(reason)};
    journal_.events.push_back(event);
    receipt_.events.push_back(std::move(event));
}
v2::Status SessionPower::persist() {
    receipt_.stage = journal_.stage;
    try {
        if (auto err = records_.save(journal_))
            return err;
        return records_.save(receipt_);
    } catch (const std::exception& error) {
        return backend_exception("persist", error.what());
    } catch (...) {
        return backend_exception("persist", "unknown exception");
    }
}

// A resolved journal is the commit record. Persist the receipt first, then
// finalize the journal: any failure keeps its durable state unresolved.
v2::Status SessionPower::persist_resolved() {
    receipt_.stage = journal_.stage;
    try {
        if (auto err = records_.save(receipt_))
            return err;
        return records_.save(journal_);
    } catch (const std::exception& error) {
        return backend_exception("persist", error.what());
    } catch (...) {
        return backend_exception("persist", "unknown exception");
    }
}
v2::Status SessionPower::abort_apply(v2::ConfigError failure, std::optional<PciAddress> pci) {
    record(EventKind::apply_failed, std::move(pci), failure.message);
    const auto rollback = restore_touched(true);
    journal_.stage = rollback ? Stage::rollback_failed : Stage::rolled_back;
    record(EventKind::rollback, {}, rollback ? rollback->message : "");
    auto durable = rollback ? persist() : persist_resolved();
    if (durable) {
        // The durable journal is still unresolved (changing/active/
        // rollback_failed), regardless of the in-memory restoration result.
        journal_.stage = Stage::rollback_failed;
        return durable;
    }
    return failure;
}
v2::Result<Observation> SessionPower::snapshot(Moment at, ObservationState stage) {
    Observation result{at, stage, "", {}};
    for (const auto& device : plan_.devices) {
        auto live = probe(hardware_, device.device.pci);
        if (!live)
            return live.error();
        ObservedCard card;
        card.pci = device.device.pci;
        card.identity = live.value().identity;
        for (size_t index = 0; index < control_count; ++index)
            card.values[index] = live.value().controls[index].current;
        result.cards.push_back(std::move(card));
    }
    return result;
}

v2::Status SessionPower::preflight() {
    if (plan_.mode != Mode::apply && plan_.mode != Mode::observe)
        return invalid("power.mode", "invalid power mode");
    if (plan_.devices.empty())
        return invalid("power.devices", "empty serving GPU inventory");
    std::set<std::string> seen;
    for (const auto& d : plan_.devices) {
        auto parsed = PciAddress::parse(d.device.pci.value);
        if (!parsed || !seen.insert(d.device.pci.value).second)
            return invalid("power.devices", "invalid or duplicate PCI device");
        if (plan_.mode == Mode::observe && d.tuning.changes_hardware())
            return invalid("power.mode", "observe mode cannot write");
        if (!d.tuning.changes_hardware())
            continue;
        if (auto failure = checked(lease_, "power.lease", "GPU lease not verified for owner"))
            return failure;
        const auto current = probe(hardware_, d.device.pci);
        if (!current)
            return current.error();
        const auto& card = current.value();
        if (!card.amdgpu || card.pci != d.device.pci)
            return unsafe("power.devices", d.device.pci.value + ": no amdgpu backend");
        const auto check = [&](ControlId id, int64_t desired, std::string_view control) -> Status {
            if (!in_range(card.at(id), desired))
                return unsafe("power.devices", d.device.pci.value + ": " + std::string(control) +
                                                   " not writable or outside device range");
            return std::nullopt;
        };
        const auto& tuning = d.tuning;

        if (tuning.voltage_offset) {
            if (auto err = check(ControlId::voltage, tuning.voltage_offset->value, "voltage offset"))
                return err;
        }
        if (tuning.shader_ceiling) {
            if (auto err = check(ControlId::shader, tuning.shader_ceiling->value, "shader clock"))
                return err;
        }
        if (tuning.shader_offset) {
            if (auto err = check(ControlId::shader_offset, tuning.shader_offset->value, "shader offset"))
                return err;
        }
        if (tuning.memory_ceiling) {
            if (auto err = check(ControlId::memory, tuning.memory_ceiling->value, "memory clock"))
                return err;
        }
        if (tuning.board_cap) {
            const int64_t watts = tuning.board_cap->value;
            if (watts <= 0 || watts > kIjson / kMicrowattsPerWatt)
                return invalid("power.board_cap", "board cap outside safely convertible range");
            if (auto err = check(ControlId::board_cap, watts * kMicrowattsPerWatt, "board cap"))
                return err;
            if (!card.hwmon)
                return unsafe("power.devices", d.device.pci.value + ": no power1_cap hwmon");
        }
        SavedDevice entry{d, card};
        if (auto err = original_can_restore(entry, card))
            return err;
        journal_.devices.push_back(std::move(entry));
    }
    return std::nullopt;
}

v2::Status SessionPower::begin() {
    if (journal_.stage != Stage::fresh)
        return state_error("power session already begun");
    if (auto used = records_.contains(journal_.run_id); !used)
        return used.error();
    else if (used.value())
        return unsafe("power.journal", "run ID already has a power journal");
    auto unresolved_records = records_.unresolved_journals();
    if (!unresolved_records)
        return unresolved_records.error();
    if (!unresolved_records.value().empty())
        return unsafe("power.journal", "unresolved power journal must be recovered first");
    if (auto err = preflight())
        return err;
    auto starting = snapshot(Moment::start, ObservationState::observed);
    if (!starting)
        return starting.error();
    receipt_.observations.push_back(std::move(starting.value()));
    if (journal_.devices.empty()) {
        journal_.stage = plan_.mode == Mode::observe ? Stage::observe : Stage::unchanged;
        receipt_.stage = journal_.stage;
        return records_.save(receipt_); // observe: no journal, no writes
    }
    journal_.stage = Stage::prepared;
    record(EventKind::prepared);
    if (auto err = persist())
        return err;
    journal_.stage = Stage::changing;
    if (auto err = persist())
        return err;
    // PWR09 requires deterministic PCI BDF order, independent of manifest
    // role/selection order. The receipt retains the authenticated role order.
    std::vector<const SavedDevice*> ordered;
    ordered.reserve(journal_.devices.size());
    for (const auto& saved : journal_.devices)
        ordered.push_back(&saved);
    std::sort(ordered.begin(), ordered.end(), [](const SavedDevice* a, const SavedDevice* b) {
        return a->policy.device.pci.value < b->policy.device.pci.value;
    });
    bool attempted_write = false;
    for (const auto* selected : ordered) {
        const auto& saved = *selected;
        if (auto failure = checked(lease_, "power.lease", "GPU lease lost before device write"))
            return attempted_write ? abort_apply(*failure, saved.policy.device.pci) : failure;
        if (saved.before.at(ControlId::shader_offset).current) {
            auto current = probe(hardware_, saved.policy.device.pci);
            if (!current) return abort_apply(current.error(), saved.policy.device.pci);
            if (auto err = original_can_restore(saved, current.value()))
                return abort_apply(*err, saved.policy.device.pci);
            if (auto err = verify_readback(saved, current.value(), true))
                return abort_apply(*err, saved.policy.device.pci);
        }
        journal_.touched.push_back(saved.policy.device.pci);
        record(EventKind::touched, saved.policy.device.pci);
        if (auto err = persist()) {
            // The first control must never be touched until the journal and
            // receipt for that card have both been saved successfully.
            return attempted_write ? abort_apply(*err, saved.policy.device.pci) : err;
        }
        v2::Status fault;
        auto current = probe(hardware_, saved.policy.device.pci);
        if (!current)
            fault = current.error();
        else if (current.value().pci != saved.before.pci || current.value().hwmon != saved.before.hwmon ||
                 !current.value().amdgpu)
            fault = unsafe("power.devices", "hardware controller changed before writing");
        if (!fault && saved.before.at(ControlId::shader_offset).current) {
            fault = original_can_restore(saved, current.value());
            if (!fault && current.value().identity != saved.before.identity)
                fault = unsafe("power.devices", "physical identity changed before writing");
            if (!fault)
                fault = verify_readback(saved, current.value(), true);
        }
        if (!fault) {
            auto program = commands_for(saved.policy, saved.before, false);
            if (!program)
                fault = program.error();
            else
                for (const auto& cmd : program.value()) {
                    if (auto failure = checked(lease_, "power.lease", "GPU lease lost during power writes")) {
                        fault = failure;
                        break;
                    }
                    attempted_write = true; // even a failed call may have changed hardware
                    if (auto err = issue(hardware_, cmd)) {
                        fault = err;
                        break;
                    }
                }
        }
        if (!fault) {
            current = probe(hardware_, saved.policy.device.pci);
            if (!current)
                fault = current.error();
            else
                fault = verify_readback(saved, current.value(), false);
        }
        if (fault)
            return abort_apply(*fault, saved.policy.device.pci);
        record(EventKind::applied, saved.policy.device.pci);
    }
    for (const auto& saved : journal_.devices) {
        if (!saved.before.at(ControlId::shader_offset).current) continue;
        if (auto failure = checked(lease_, "power.lease", "GPU lease lost before final signed readback"))
            return abort_apply(*failure, saved.policy.device.pci);
        auto current = probe(hardware_, saved.policy.device.pci);
        if (!current) return abort_apply(current.error(), saved.policy.device.pci);
        if (auto err = verify_readback(saved, current.value(), false))
            return abort_apply(*err, saved.policy.device.pci);
    }
    auto applied = snapshot(Moment::post_apply, ObservationState::applied);
    if (!applied)
        return abort_apply(applied.error(), {});
    receipt_.observations.push_back(std::move(applied.value()));
    journal_.stage = Stage::active;
    if (auto err = persist())
        return abort_apply(*err, {});
    return std::nullopt;
}

v2::Status SessionPower::restore_touched(bool rollback) {
    // Prevalidate every original before restoring any of them. Controller
    // identity/driver ranges may have changed since the initial apply.
    std::vector<const SavedDevice*> reverse;
    for (auto it = journal_.touched.rbegin(); it != journal_.touched.rend(); ++it) {
        const auto found = std::find_if(journal_.devices.begin(), journal_.devices.end(),
                                        [&](const SavedDevice& d) { return d.policy.device.pci == *it; });
        if (found == journal_.devices.end())
            return state_error("touched card missing from journal");
        reverse.push_back(&*found);
    }
    for (const auto* saved : reverse) {
        if (auto failure = checked(lease_, "power.lease", "GPU lease not held during restoration"))
            return failure;
        auto current = probe(hardware_, saved->policy.device.pci);
        if (!current)
            return current.error();
        if (auto err = original_can_restore(*saved, current.value()))
            return err;
    }
    v2::Status first_error;
    for (const auto* saved : reverse) {
        v2::Status problem;
        auto program = commands_for(saved->policy, saved->before, true);
        if (!program)
            problem = program.error();
        else
            for (const auto& cmd : program.value()) {
                if (auto failure = checked(lease_, "power.lease", "GPU lease lost during restore")) {
                    problem = failure;
                    break;
                }
                if (auto err = issue(hardware_, cmd)) {
                    problem = err;
                    break;
                }
            }
        if (!problem) {
            auto now = probe(hardware_, saved->policy.device.pci);
            if (!now)
                problem = now.error();
            else
                problem = verify_readback(*saved, now.value(), true);
        }
        if (problem) {
            record(EventKind::restore_failed, saved->policy.device.pci, problem->message);
            if (!first_error)
                first_error = problem;
        } else {
            record(EventKind::restored, saved->policy.device.pci, rollback ? "rollback" : "shutdown");
        }
    }
    for (const auto* saved : reverse) {
        if (!saved->before.at(ControlId::shader_offset).current) continue;
        if (auto failure = checked(lease_, "power.lease", "GPU lease lost before final signed restore readback"))
            return failure;
        auto current = probe(hardware_, saved->policy.device.pci);
        if (!current) return current.error();
        if (auto err = verify_readback(*saved, current.value(), true))
            return err;
    }
    return first_error;
}

v2::Status SessionPower::finish() {
    if (journal_.stage == Stage::observe || journal_.stage == Stage::unchanged ||
        journal_.stage == Stage::rolled_back || journal_.stage == Stage::restored)
        return std::nullopt;
    if (!unresolved(journal_.stage))
        return state_error("power session has no active transaction");
    if (auto failure =
            checked(children_exited_, "power.children", "serving process groups may still be alive")) {
        record(EventKind::restore_refused, {},
               failure->code == Code::admission_refused ? "serving process groups not verified exited"
                                                        : failure->message);
        if (auto err = persist())
            return err;
        return failure;
    }
    if (auto failure = checked(lease_, "power.lease", "owner no longer holds GPU lease"))
        return failure;
    auto observed = snapshot(Moment::pre_restore, ObservationState::observed);
    if (!observed)
        return observed.error();
    receipt_.observations.push_back(std::move(observed.value()));
    auto failure = restore_touched(false);
    journal_.stage = failure ? Stage::restore_failed : Stage::restored;
    auto finished = snapshot(Moment::stop, failure ? ObservationState::unresolved : ObservationState::restored);
    if (!finished) {
        journal_.stage = Stage::restore_failed;
        record(EventKind::restore_failed, {}, finished.error().message);
        if (auto err = persist())
            return err;
        return finished.error();
    }
    if (failure)
        finished.value().reason = failure->message;
    receipt_.observations.push_back(std::move(finished.value()));
    auto durable = failure ? persist() : persist_resolved();
    if (durable) {
        journal_.stage = Stage::restore_failed;
        return durable;
    }
    return failure;
}

v2::Status recover(const fs::path& journal_path, Hardware& hardware, Records& records,
                   const SessionPower::Predicate& lease, const SessionPower::Predicate& children_exited) {
    if (auto failure = SessionPower::checked(children_exited, "power.recovery",
                                             "serving process groups not verified exited"))
        return failure;
    if (auto failure = SessionPower::checked(lease, "power.recovery", "GPU lease not verified for recovery"))
        return failure;
    auto read = records.load(journal_path);
    if (!read)
        return read.error();
    auto allowed = validate_recovery(std::move(read.value()), hardware);
    if (!allowed)
        return allowed.error();
    Journal record = std::move(allowed.value());
    v2::Status failure;
    Observation stopping{Moment::stop, ObservationState::restored, "", {}};
    for (auto it = record.touched.rbegin(); it != record.touched.rend(); ++it) {
        const auto found = std::find_if(record.devices.begin(), record.devices.end(),
                                        [&](const SavedDevice& d) { return d.policy.device.pci == *it; });
        if (found == record.devices.end())
            return state_error("invalid recovery touched list");
        v2::Status problem;
        auto program = commands_for(found->policy, found->before, true);
        if (!program)
            problem = program.error();
        else
            for (const auto& cmd : program.value()) {
                if (auto gate_failure =
                        SessionPower::checked(lease, "power.recovery", "GPU lease lost during recovery")) {
                    problem = gate_failure;
                    break;
                }
                if (auto err = issue(hardware, cmd)) {
                    problem = err;
                    break;
                }
            }
        if (!problem) {
            auto now = probe(hardware, found->policy.device.pci);
            if (!now) {
                problem = now.error();
            } else {
                problem = verify_readback(*found, now.value(), true);
                if (!problem) {
                    ObservedCard card;
                    card.pci = found->policy.device.pci;
                    for (size_t index = 0; index < control_count; ++index)
                        card.values[index] = now.value().controls[index].current;
                    stopping.cards.push_back(std::move(card));
                }
            }
        }
        record.events.push_back({problem ? EventKind::restore_failed : EventKind::restored,
                                 found->policy.device.pci, problem ? problem->message : ""});
        if (problem && !failure)
            failure = problem;
    }
    for (const auto& saved : record.devices) {
        if (!saved.before.at(ControlId::shader_offset).current ||
            std::find(record.touched.begin(), record.touched.end(), saved.policy.device.pci) == record.touched.end())
            continue;
        if (auto gate = SessionPower::checked(lease, "power.recovery", "GPU lease lost before final signed recovery readback")) {
            if (!failure) failure = gate;
            continue;
        }
        auto now = probe(hardware, saved.policy.device.pci);
        if (!now) {
            if (!failure) failure = now.error();
            continue;
        }
        if (auto err = verify_readback(saved, now.value(), true))
            if (!failure) failure = err;
    }
    record.stage = failure ? Stage::restore_failed : Stage::restored;
    stopping.state = failure ? ObservationState::unresolved : ObservationState::restored;
    if (failure)
        stopping.reason = failure->message;
    // Reconcile the prior manifest-bound receipt *before* a successful
    // journal is resolved. An inaccessible/invalid receipt keeps the journal
    // unresolved and prevents a misleading success from crash recovery.
    try {
        if (failure) {
            if (auto err = records.save(record))
                return err; // remains unresolved on any persistence failure
        }
        if (auto err = records.reconcile_recovery(record, stopping))
            return err;
        if (!failure) {
            if (auto err = records.save(record))
                return err;
        }
    } catch (const std::exception& error) {
        return backend_exception("recovery persistence", error.what());
    } catch (...) {
        return backend_exception("recovery persistence", "unknown exception");
    }
    return failure;
}

} // namespace bc::serving::power
