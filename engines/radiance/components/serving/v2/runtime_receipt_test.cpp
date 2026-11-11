#include "v2/runtime_receipt.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>

#include "../../moe-remote/moe_remote_shm.h"

using namespace bc::serving::v2;
using r3::moe_shm::PlacementControl;
using r3::moe_shm::PlacementJournal;
using r3::moe_shm::PlacementState;

static const std::string SHA(64, 'a');

static LivePolicyStartup make_startup(bool configured = true) {
  LivePolicyStartup s{};
  s.requested = configured ? LivePolicyRequested::kOn : LivePolicyRequested::kOff;
  s.configured = configured;
  s.reason_code = configured ? "configured" : "disabled_by_request";
  s.reason_detail = configured ? "configured" : "disabled";
  s.calibration_key = configured ? "live_cadence" : "";
  s.record_identity = configured ? "record-1" : "";
  s.effect_gate = configured;
  s.histogram_period_s = configured ? 1 : 0;
  s.cadence_ns = configured ? 1000 : 0;
  s.max_heat_age_ns = configured ? 2000 : 0;
  return s;
}

static PlacementControl make_control(uint64_t epoch = 7, uint64_t digest = 99,
                                    uint32_t fault = 0, uint32_t journal_index = 0,
                                    uint64_t txn = 42) {
  PlacementControl c{};
  c.active_epoch = epoch;
  c.active_digest = digest;
  c.state = static_cast<uint32_t>(PlacementState::ACTIVE);
  c.transaction_id = txn;
  c.journal_index = journal_index;
  c.fault_code = fault;
  c.fault_actor = fault ? 7 : 0;
  return c;
}

static PlacementJournal make_journal(uint64_t txn = 42, uint32_t current_cleanup = 0,
                                    uint32_t other_cleanup = 0, uint32_t current_fault = 0) {
  PlacementJournal j{};
  j.entry[0].transaction_id = txn;
  j.entry[0].cleanup_fault = current_cleanup;
  j.entry[0].cleanup_actor = current_cleanup ? 3 : 0;
  j.entry[0].fault = current_fault;
  // Stale other slot carries unrelated identity/fault.
  j.entry[1].transaction_id = 9999;
  j.entry[1].cleanup_fault = other_cleanup;
  j.entry[1].cleanup_actor = other_cleanup ? 9 : 0;
  return j;
}

int main() {
  const std::string digest = "sha256:" + SHA;

  // Active runtime with genuine telemetry.
  {
    auto control = make_control();
    auto journal = make_journal();
    auto ok = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, journal);
    assert(ok);
    const auto& doc = ok.value();
    assert(doc.at("runtime_status") == "active");
    assert(doc.at("placement_stopped") == false);
    assert(doc.at("degraded") == false);
    assert(doc.at("bounding").at("epoch") == 7);
    assert(doc.at("bounding").at("owner_generation_digest") == 99);
    assert(doc.at("startup_decision").at("requested") == "on");
    assert(doc.at("startup_decision").at("configured") == true);
    assert(doc.at("snapshot_count") == 1);
    assert(doc.at("snapshots").at(0).at("placement_stopped") == false);
  }

  // Missing telemetry (zero control) fails closed.
  {
    PlacementControl zero{};
    PlacementJournal zeroj{};
    auto bad = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, zero, zeroj);
    assert(!bad);
  }

  // Degraded from nonzero control.fault_code, not placement_stopped.
  {
    auto control = make_control(7, 99, 5);
    auto journal = make_journal();
    auto degraded = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, journal);
    assert(degraded);
    assert(degraded.value().at("runtime_status") == "degraded");
    assert(degraded.value().at("placement_stopped") == false);
    assert(degraded.value().at("degraded") == true);
  }

  // Placement stopped only from current matched journal cleanup fault.
  {
    auto control = make_control();
    auto journal = make_journal(42, 0, 0, 0); // current cleanup_fault? need set
    // make_journal signature: current_cleanup param third? adjust: call explicitly
    PlacementJournal j{};
    j.entry[0].transaction_id = 42;
    j.entry[0].cleanup_fault = 9;
    j.entry[0].cleanup_actor = 3;
    j.entry[1].transaction_id = 9999;
    j.entry[1].cleanup_fault = 0;
    auto stopped = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, j);
    assert(stopped);
    assert(stopped.value().at("runtime_status") == "placement_stopped");
    assert(stopped.value().at("placement_stopped") == true);
    assert(stopped.value().at("snapshots").at(0).at("placement_stopped") == true);
    assert(stopped.value().at("snapshots").at(0).at("cleanup_fault") == 9);
  }

  // Stale other slot cleanup fault is ignored.
  {
    auto control = make_control();
    PlacementJournal j{};
    j.entry[0].transaction_id = 42;
    j.entry[0].cleanup_fault = 0;
    j.entry[1].transaction_id = 9999;
    j.entry[1].cleanup_fault = 9;
    auto ok = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, j);
    assert(ok);
    assert(ok.value().at("runtime_status") == "active");
    assert(ok.value().at("placement_stopped") == false);
  }

  // control.fault_code nonzero => degraded, even if current cleanup zero.
  {
    auto control = make_control(7, 99, 5);
    auto journal = make_journal();
    auto degraded = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, journal);
    assert(degraded);
    assert(degraded.value().at("runtime_status") == "degraded");
    assert(degraded.value().at("placement_stopped") == false);
  }

  // Digest mismatch fails.
  {
    auto control = make_control();
    auto journal = make_journal();
    auto bad = publish_runtime_receipt("test-run", make_startup(), digest, 7, 100, control, journal);
    assert(!bad);
  }

  // Epoch mismatch fails.
  {
    auto control = make_control();
    auto journal = make_journal();
    auto bad = publish_runtime_receipt("test-run", make_startup(), digest, 8, 99, control, journal);
    assert(!bad);
  }

  // Generation/owner digest mismatch with control fails.
  {
    auto control = make_control(7, 99);
    auto journal = make_journal();
    auto bad = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, journal);
    // mutate control active_digest mismatch by changing control copy
    PlacementControl c2 = control;
    c2.active_digest = 100;
    auto bad2 = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, c2, journal);
    assert(!bad2);
  }

  // Mismatched journal transaction fails.
  {
    auto control = make_control();
    PlacementJournal j{};
    j.entry[0].transaction_id = 7; // mismatched
    j.entry[0].cleanup_fault = 0;
    auto bad = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, j);
    assert(!bad);
  }

  // Startup off is valid and preserved.
  {
    auto control = make_control();
    auto journal = make_journal();
    auto ok = publish_runtime_receipt("test-run", make_startup(false), digest, 7, 99, control, journal);
    assert(ok);
    assert(ok.value().at("startup_decision").at("requested") == "off");
    assert(ok.value().at("startup_decision").at("configured") == false);
  }

  // Startup configured inconsistency fails.
  {
    LivePolicyStartup bad_startup = make_startup(false);
    bad_startup.configured = true;
    auto control = make_control();
    auto journal = make_journal();
    auto bad = publish_runtime_receipt("test-run", bad_startup, digest, 7, 99, control, journal);
    assert(!bad);
  }

  // Manifest digest invalid fails.
  {
    auto control = make_control();
    auto journal = make_journal();
    auto bad = publish_runtime_receipt("test-run", make_startup(), std::string("sha256:") + std::string(63, 'a'), 7, 99, control, journal);
    assert(!bad);
  }

  // A healthy transition is in flight, not a degraded policy.
  {
    auto control = make_control();
    auto journal = make_journal();
    control.state = static_cast<uint32_t>(r3::moe_shm::PlacementState::RESERVING);
    auto in_flight = publish_runtime_receipt("test-run", make_startup(), digest, 7, 99, control, journal);
    assert(in_flight);
    assert(in_flight.value().at("runtime_status") == "transitioning");
    assert(in_flight.value().at("degraded") == false);
  }

  // The production reader sees one coherent control/journal pair inside the
  // worker seqlock. Synthetic segment uses the same validated v24 layout.
  {
    using namespace r3::moe_shm;
    const LayoutShape shape{16, 1, 2, 16, 8, 16};
    Layout layout{};
    assert(layout_for(shape, 1, slots_for(1, 1), &layout));
    void* mem = std::aligned_alloc(64, (layout.bytes + 63u) & ~uint64_t(63u));
    assert(mem);
    std::unique_ptr<void, decltype(&std::free)> owned(mem, &std::free);
    assert(init(mem, 1, shape, 1, 1));
    auto* hdr = static_cast<Header*>(mem);
    hdr->helper_count = 1;
    auto* control = placement_control(mem);
    auto* generation = generation_header(mem, 0);
    auto* journal = placement_journal(mem);
    control->seq = 2;
    control->transaction_id = 42;
    control->active_epoch = 7;
    control->journal_index = 0;
    generation->epoch = 7;
    generation->ready_helpers = 1;
    journal->entry[0].transaction_id = 42;
    journal->entry[0].before_epoch = 7;
    journal->entry[0].before_digest = control->active_digest;
    assert(valid_self(hdr, layout.bytes));
    auto live = [&] {
      return publish_live_runtime_receipt("test-run", make_startup(),
                                           digest, mem, layout.bytes);
    };
    auto good = live();
    assert(good);
    assert(good.value().at("run_id") == "test-run");
    assert(good.value().at("runtime_status") == "active");

    journal->entry[0].before_epoch = 6;
    assert(!live()); // stale journal epoch
    journal->entry[0].before_epoch = 7;
    journal->entry[0].before_digest += 1;
    assert(!live()); // mismatched owner generation
    journal->entry[0].before_digest = control->active_digest;
    journal->entry[1] = journal->entry[0];
    journal->entry[1].transaction_id = 123;
    control->journal_index = 1;
    assert(!live()); // stale alternate journal
    control->journal_index = 0;
    control->seq = 3;
    assert(!live()); // publication in flight
    control->seq = 2;
    auto* map = generation_map(mem, 0);
    const uint8_t old = map[0];
    map[0] = 0;
    assert(!live()); // control digest differs from owner bytes
    map[0] = old;
    assert(live());
    hdr->version++;
    assert(!live()); // invalid segment generation
  }

  return 0;
}