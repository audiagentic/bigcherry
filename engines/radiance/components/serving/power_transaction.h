#pragma once

// RSC02 S12: session-owned GPU power admission and restoration.
// Authority comes only from a manifest accepted by the shared v2 reader.
// Power is a transaction over selected PCI devices, not a subprocess/CLI action.
#include "callback_boundary.h"
#include "v2/error.h"
#include "v2/manifest.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace bc::serving::power {

namespace fs = std::filesystem;

struct PciAddress {
    std::string value;
    bool operator==(const PciAddress&) const = default;
    static v2::Result<PciAddress> parse(std::string_view text);
};

struct Digest {
    std::string value;
    static v2::Result<Digest> parse(std::string_view text);
};

template <class Tag> struct Quantity {
    int64_t value = 0;
    bool operator==(const Quantity&) const = default;
};
struct VoltageTag {};
struct FrequencyTag {};
struct WattsTag {};
struct MicrowattsTag {};
using Millivolts = Quantity<VoltageTag>;
using Megahertz = Quantity<FrequencyTag>;
using Watts = Quantity<WattsTag>;
using Microwatts = Quantity<MicrowattsTag>;

enum class Mode : uint8_t { observe, apply };
enum class RoleKind : uint8_t { engine, helper };
struct Role {
    RoleKind kind = RoleKind::engine;
    uint32_t helper_index = 0;
    std::string label() const;
    static v2::Result<Role> parse(std::string_view text);
    bool operator==(const Role&) const = default;
};

struct Tuning {
    std::optional<Millivolts> voltage_offset;
    std::optional<Megahertz> shader_ceiling;
    std::optional<Megahertz> memory_ceiling;
    std::optional<Watts> board_cap;
    std::optional<Megahertz> shader_offset;
    bool changes_hardware() const;
    bool changes_overdrive() const;
};
struct SelectedGpu {
    Role role;
    PciAddress pci;
};
struct DevicePolicy {
    SelectedGpu device;
    Tuning tuning;
};
struct PowerPlan {
    Mode mode = Mode::observe;
    std::string profile;
    Digest profile_digest;
    Digest manifest_digest;
    std::vector<DevicePolicy> devices;
};

// This function accepts a *typed* strict-reader-validated v2 manifest, not a
// duck-typed JSON dictionary. A different role, order or BDF refuses.
v2::Result<PowerPlan> bind(const v2::Manifest& verified, std::span<const SelectedGpu> selected);

enum class ControlId : uint8_t { voltage, shader, memory, board_cap, shader_offset, count };
constexpr size_t control_count = static_cast<size_t>(ControlId::count);
struct Capability {
    std::optional<int64_t> current;
    std::optional<int64_t> minimum;
    std::optional<int64_t> maximum;
    bool writable = false;
};
struct CardSnapshot {
    PciAddress pci;
    bool amdgpu = false;
    std::optional<std::string> hwmon; // "hwmonNNN", not a path supplied by a manifest
    // Kernel release, amdgpu version and PCI device/subsystem ids as one plain string compared by equality. A
    // receipt qualifies a setting only for the identity it was applied on.
    std::string identity;
    std::array<Capability, control_count> controls{};
    const Capability& at(ControlId id) const;
};
struct ShaderClock {
    Megahertz value;
};
struct ShaderOffset {
    Megahertz value;
};
struct MemoryClock {
    Megahertz value;
};
struct VoltageOffset {
    Millivolts value;
};
struct OverdriveCommit {};
struct BoardCap {
    Microwatts value;
};
using CommandValue = std::variant<ShaderClock, MemoryClock, VoltageOffset, OverdriveCommit, BoardCap, ShaderOffset>;
struct PowerCommand {
    PciAddress pci;
    std::optional<std::string> hwmon;
    CommandValue value;
};
struct SysfsWrite {
    fs::path path;
    std::string value;
    bool operator==(const SysfsWrite&) const = default;
};

// Rendering is the Linux adapter's only string conversion of typed commands.
// Paths and values cannot be supplied by the manifest.
v2::Result<SysfsWrite> render(const PowerCommand& command);
v2::Result<std::vector<PowerCommand>> commands_for(const DevicePolicy& policy, const CardSnapshot& before,
                                                   bool restoring);

class Hardware {
  public:
    virtual ~Hardware() = default;
    v2::Result<CardSnapshot> inspect(const PciAddress& pci) {
        return callback::invoke_backend("power.hardware.inspect", [&] { return do_inspect(pci); });
    }
    v2::Status execute(const PowerCommand& command) {
        return callback::invoke_backend("power.hardware.execute", [&] { return do_execute(command); });
    }

  private:
    virtual v2::Result<CardSnapshot> do_inspect(const PciAddress& pci) = 0;
    virtual v2::Status do_execute(const PowerCommand& command) = 0;
};

class LinuxHardware final : public Hardware {
  public:
    explicit LinuxHardware(fs::path root = "/") : root_(std::move(root)) {}

  private:
    v2::Result<CardSnapshot> do_inspect(const PciAddress& pci) override;
    v2::Status do_execute(const PowerCommand& command) override;
    fs::path root_;
};

enum class Stage : uint8_t {
    fresh,
    prepared,
    changing,
    active,
    rolled_back,
    rollback_failed,
    restored,
    restore_failed,
    observe,
    unchanged
};
std::string_view stage_name(Stage stage);
v2::Result<Stage> stage_from_name(std::string_view name);
bool unresolved(Stage stage);

enum class EventKind : uint8_t {
    prepared,
    touched,
    applied,
    apply_failed,
    restored,
    restore_failed,
    restore_refused,
    rollback
};
struct Event {
    EventKind kind;
    std::optional<PciAddress> pci;
    std::string reason;
};
struct SavedDevice {
    DevicePolicy policy;
    CardSnapshot before;
};
struct Journal {
    std::string run_id;
    Digest manifest_digest, profile_digest;
    Stage stage = Stage::fresh;
    std::vector<SavedDevice> devices;
    std::vector<PciAddress> touched;
    std::vector<Event> events;
};
enum class Moment : uint8_t { start, post_apply, pre_restore, stop, gfxoff };
enum class ObservationState : uint8_t {
    observed,
    applied,
    restored,
    unavailable,
    unresolved,
    gfxoff_active,
    gfxoff_absent,
    gfxoff_ambiguous,
    gfxoff_not_applicable,
    gfxoff_needs_context
};
struct ObservedCard {
    PciAddress pci;
    std::string identity; // CardSnapshot::identity at the observation
    std::array<std::optional<int64_t>, control_count> values{};
};
struct Observation {
    Moment at;
    ObservationState state;
    std::string reason;
    std::vector<ObservedCard> cards;
};
struct Verdict; // independently evaluated PWR06 paired-window diagnostic

struct Receipt {
    std::string run_id;
    PowerPlan plan;
    Stage stage = Stage::fresh;
    std::vector<Event> events;
    std::vector<Observation> observations;
    std::string workload; // the config that ran; evidence only, never a qualification match key
};

// Explicitly isolates persistence; fault tests can fail *any* durable write.
// Production journal replacements are fsync'ed and atomically renamed.
class Records {
  public:
    virtual ~Records() = default;
    v2::Status save(const Journal& journal) {
        return callback::invoke_backend("power.records.save_journal", [&] { return do_save(journal); });
    }
    v2::Status save(const Receipt& receipt) {
        return callback::invoke_backend("power.records.save_receipt", [&] { return do_save(receipt); });
    }
    v2::Result<std::vector<fs::path>> unresolved_journals() {
        return callback::invoke_backend("power.records.unresolved", [&] { return do_unresolved_journals(); });
    }
    v2::Result<bool> contains(std::string_view run_id) {
        return callback::invoke_backend("power.records.contains", [&] { return do_contains(run_id); });
    }
    v2::Result<Journal> load(const fs::path& path) {
        return callback::invoke_backend("power.records.load", [&] { return do_load(path); });
    }
    v2::Status reconcile_recovery(const Journal& journal, const Observation& stop) {
        return callback::invoke_backend("power.records.reconcile",
                                        [&] { return do_reconcile_recovery(journal, stop); });
    }
    fs::path journal_path(std::string_view run_id) const {
        // A path-only compatibility API; all throw types convert to a typed
        // Failure. No raw virtual call is available to the caller.
        return callback::Guarded<fs::path(std::string_view)>(
                   [this](std::string_view id) { return do_journal_path(id); })
            .require("power.records.journal_path", run_id);
    }

  private:
    virtual v2::Status do_save(const Journal&) = 0;
    virtual v2::Status do_save(const Receipt&) = 0;
    virtual v2::Result<std::vector<fs::path>> do_unresolved_journals() = 0;
    virtual v2::Result<bool> do_contains(std::string_view run_id) = 0;
    virtual v2::Result<Journal> do_load(const fs::path&) = 0;
    virtual v2::Status do_reconcile_recovery(const Journal&, const Observation& stop) = 0;
    virtual fs::path do_journal_path(std::string_view run_id) const = 0;
};
class DiskRecords final : public Records {
  public:
    DiskRecords(fs::path journals, fs::path receipts)
        : journal_dir_(std::move(journals)), receipt_dir_(std::move(receipts)) {}

  private:
    v2::Status do_save(const Journal& journal) override;
    v2::Status do_save(const Receipt& receipt) override;
    v2::Result<std::vector<fs::path>> do_unresolved_journals() override;
    v2::Result<bool> do_contains(std::string_view run_id) override;
    v2::Result<Journal> do_load(const fs::path& path) override;
    v2::Status do_reconcile_recovery(const Journal& journal, const Observation& stop) override;
    fs::path do_journal_path(std::string_view run_id) const override;

    fs::path journal_dir_, receipt_dir_;
};

// v2::Status is the serving component's existing optional typed ConfigError.
// A normal admission or stop refusal is a value, not an exception.
class SessionPower {
  public:
    // No friend or direct access to the underlying callable exists.
    using Predicate = callback::Guarded<bool()>;
    SessionPower(PowerPlan plan, std::string run_id, Hardware& hardware, Records& records,
                 Predicate owns_gpu_lease, Predicate children_exited);

    void set_workload(std::string name) {
        receipt_.workload = std::move(name);
    }
    v2::Status begin();  // full preflight and durable receipt/journal BEFORE writes
    v2::Status finish(); // no writes if any child group may still be alive
    v2::Status report_gfxoff(const Verdict& verdict); // advisory, persisted receipt
    Stage stage() const {
        return journal_.stage;
    }
    const Receipt& receipt() const {
        return receipt_;
    }

  private:
    PowerPlan plan_;
    Hardware& hardware_;
    Records& records_;
    Predicate lease_, children_exited_;
    // The sole invocation point for all externally supplied predicates.
    // False means admission refusal; exceptions become typed IO failures.
    static v2::Status checked(const Predicate& predicate, std::string_view path, std::string_view refused);
    friend v2::Status recover(const fs::path& journal_path, Hardware& hardware, Records& records,
                              const Predicate& owns_gpu_lease, const Predicate& children_exited);
    Journal journal_;
    Receipt receipt_;
    v2::Status persist();
    v2::Status persist_resolved();
    v2::Status abort_apply(v2::ConfigError failure, std::optional<PciAddress> pci);
    v2::Status preflight();
    v2::Result<Observation> snapshot(Moment at, ObservationState stage);
    v2::Status restore_touched(bool rollback);
    void record(EventKind kind, std::optional<PciAddress> pci = {}, std::string reason = {});
};

// Manual crash recovery: independent verified lease and all-child exit, all
// selected originals re-probed and range-checked before any restore write.
v2::Status recover(const fs::path& journal, Hardware& hardware, Records& records,
                   const SessionPower::Predicate& owns_gpu_lease,
                   const SessionPower::Predicate& children_exited);

// Diagnostic path, not a device controller. No thresholds for inferred GFXOFF.
enum class GfxoffState : uint8_t { active, absent, ambiguous, unavailable, not_applicable, needs_idle_context };
enum class WindowPhase : uint8_t { no_process, engine_loaded_idle };
enum class CaptureState : uint8_t { measured, unavailable, not_applicable };
struct Sample {
    Microwatts power;
    Millivolts voltage;
    uint8_t busy_percent = 0;
    std::chrono::system_clock::time_point timestamp;
};
struct Window {
    PciAddress pci;
    WindowPhase phase;
    std::string identity, sensor_identity;
    std::vector<Sample> samples;
    CaptureState acquisition = CaptureState::unavailable;
    std::string reason;
};
enum class GfxoffReason : uint8_t {
    matching_floor,
    elevated_signals,
    incomplete_window,
    no_engine_context,
    not_gfx12,
    failed_capture,
    identity_mismatch,
    floor_not_idle,
    time_order_mismatch,
    insufficient_samples,
    mixed_signals
};
struct Verdict {
    GfxoffState state;
    GfxoffReason reason;
};
Verdict evaluate(const std::optional<Window>& floor, const std::optional<Window>& idle);

class Sensor {
  public:
    virtual ~Sensor() = default;
    v2::Result<Window> measure(const PciAddress& pci, WindowPhase phase, size_t samples) {
        return callback::invoke_backend("power.sensor.measure",
                                        [&] { return do_measure(pci, phase, samples); });
    }

  private:
    virtual v2::Result<Window> do_measure(const PciAddress&, WindowPhase, size_t samples) = 0;
};
class LinuxSensor final : public Sensor {
  public:
    using Interval = callback::Guarded<void()>;
    explicit LinuxSensor(fs::path sys_root = "/sys", Interval between = {});

  private:
    v2::Result<Window> do_measure(const PciAddress& pci, WindowPhase phase, size_t samples) override;
    fs::path sys_root_;
    Interval between_;
};

enum class HostIssue : uint8_t {
    invalid_selection,
    missing_backend,
    missing_control,
    unsafe_range,
    cpu_energy_unreadable,
    probe_failed
};
struct HostFinding {
    HostIssue code;
    std::optional<PciAddress> pci;
    std::string detail;
};
std::vector<HostFinding> inspect_host(const PowerPlan& plan, Hardware& hardware,
                                      std::span<const fs::path> rapl_energy,
                                      const callback::Guarded<bool(const fs::path&)>& readable);

} // namespace bc::serving::power
