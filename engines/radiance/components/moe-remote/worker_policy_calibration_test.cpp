#include "worker_policy_calibration.h"
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <string>
#include <vector>

using namespace r3::policy;

int main() {
    // A calibration report line ends in a real newline and carries no backslash-n text: the launcher anchors its record search
    // at the start of a line, so a waiting line that ran into the next report hid a complete record for 80 minutes.
    {
        const std::string line = calibration_report_line("calibration waiting: drift_window (x)");
        assert(line.size() > 1 && line.back() == '\n');
        assert(line.find("\\n") == std::string::npos);
        const std::string joined = line + "RHP16_CALIBRATION_JSON {}\n";
        assert(joined.find("\nRHP16_CALIBRATION_JSON ") != std::string::npos);
    }
    // Use the hook's actual ring capacity and accepted source observations.
    DriftTracker drift(1, 4);
    Measured m;
    m.drift_capacity = drift.capacity();
    for (uint64_t obs = 1; obs <= 3; ++obs) {
        drift.add({double(5 - obs), double(obs)}, obs * 10);
        m.drift_intervals = drift.intervals();
        if (obs > 1) m.delta_gap_ns.add(1000000000.0);
        assert(!calibration_settled(m));
    }
    drift.add({1, 4}, 40);
    m.drift_intervals = drift.intervals();
    m.drift_horizon_observations = drift.horizon_observations();
    m.drift_half_life_observations = drift.half_life_observations();
    m.delta_gap_ns.add(1000000000.0);
    // Heat horizon alone is not convergence: a move and independent quiet
    // timing series must also be observed.
    assert(!calibration_settled(m));
    m.transaction_ns.add(10.0);
    m.tail.gain_ppm.add(1000.0);
    m.slowest_helper_copy_bps.add(1000000.0);
    m.tail_lane_service_ns_per_s.add(200000000.0);
    m.tail_lane_row_ns.add(1250.0);
    m.lane_lost_ns.add(100.0);
    m.hot_set_k = 2;
    m.marginal_error = 0.025;
    m.marginal_observations = 40;
    m.observation_rate_per_s = 10.0;

    // RHP19: serialized helper progress carries the exact retained ring,
    // accepted delta gap and observation-space horizon for the launcher.
    {
        const auto progress=calibration_drift_progress(m);
        assert(progress.at("capacity")==4);
        assert(progress.at("intervals")==4);
        assert(progress.at("horizon_observations")==30);
        assert(progress.at("observation_rate_per_s")==10);
        assert(progress.at("accepted_gap_ns")==1000000000ull);
        assert(progress.at("accepted_heat_deltas")==0);
    }

    // No input accepted => no watermark measurement or settled record.
    assert(!calibration_settled(m));
    m.accepted_heat_delta(true);
    m.effect_required_quiet_windows=2;
    m.effect.normalized_noise_ns.add(10.0);
    m.effect.normalized_noise_ns.add(-10.0);
    m.effect.quiet_delta_ns.add(5.0);
    m.effect.quiet_comparisons=2;
    m.effect.sampled_rows_per_s=20.0;
    m.effect_cadence_ns=1000000000ull;
    // A too-short published series must not certify calibration.
    m.effect_window_noise_ladder={{5,80000,3},{10,40000,3}};
    assert(!calibration_settled(m));
    const auto short_row=calibration_record(m,
        {std::string(64,'b'),"flashnext-4gpu","balanced"},"2026-11-06");
    assert(short_row.at("absent").at("effect_window_noise_ladder").is_string());
    m.effect_window_noise_ladder.push_back({20,20000,3});
    assert(calibration_settled(m));
    assert(m.drift_capacity == 4 && m.drift_horizon_observations == 30);
    // A counter discontinuity clears the horizon; the same cadence must not
    // accidentally certify a reset source.
    drift.add({2, 3}, 1);
    m.drift_intervals = drift.intervals();
    assert(!calibration_settled(m));

    m.drift_intervals = m.drift_capacity;
    m.transaction_ns.add(10.0);
    m.transaction_ns.add(30.0);
    m.hot_set_k = 2;
    m.marginal_error = 0.025;
    m.marginal_observations = 40;
    m.observation_rate_per_s = 10.0;

    const auto row = calibration_record(m, {std::string(64, 'a'), "flashnext-4gpu", "balanced"}, "2026-11-05");
    assert(row.at("measurements").at("txn_p50_ns") == 10);
    assert(row.at("measurements").at("txn_p95_ns") == 30);
    assert(row.at("measurements").at("hot_set_k") == 2);
    assert(row.at("measurements").at("heat_watermarked") == 1);
    Measured mixed = m;
    mixed.accepted_heat_delta(false);
    mixed.accepted_heat_delta(true); // once unwatermarked, always zero
    assert(calibration_record(mixed, {std::string(64, 'd'), "flashnext-4gpu", "balanced"},
                              "2026-11-06").at("measurements").at("heat_watermarked") == 0);
    Measured unobserved = m;
    unobserved.accepted_heat_deltas = 0;
    unobserved.accepted_unwatermarked_heat = false;
    assert(!calibration_settled(unobserved));
    const auto no_heat = calibration_record(unobserved,
        {std::string(64, 'e'), "flashnext-4gpu", "balanced"}, "2026-11-06");
    assert(no_heat.at("absent").at("heat_watermarked") == "no accepted heat deltas");
    assert(row.at("measurements").at("marginal_error_ppm") == 25000);
    assert(row.at("measurements").at("copy_bytes_per_s") == 1000000);
    assert(!row.at("absent").contains("copy_bytes_per_s"));
    assert(row.at("measurements").at("tail_lane_service_ns_per_s") == 200000000);
    assert(row.at("measurements").at("tail_lane_row_ns") == 1250);
    Measured no_requests=m;
    no_requests.tail_lane_row_ns=Series{64};
    assert(!calibration_settled(no_requests));
    const auto missing_request=calibration_record(no_requests,
        {std::string(64, 'c'), "flashnext-4gpu", "balanced"}, "2026-11-06");
    assert(missing_request.at("absent").at("tail_lane_row_ns").is_string());
    assert(!missing_request.at("measurements").contains("tail_lane_row_ns"));
    assert(row.at("measurements").at("lane_lost_ns") == 100);
    assert(row.at("measurements").at("effect_row_noise_ns") == 10);
    assert(row.at("measurements").at("effect_noise_max_ns") == 5);
    assert(row.at("measurements").at("effect_window_noise_ladder").size()==3);
    assert(row.at("measurements").at("effect_window_noise_ladder")[2][0]==20);
    for (const char* removed : {"effect_window_samples", "effect_quiet_windows",
                                "noise_false_revert_rate_ppm", "effect_noise_quantile_ns",
                                "effect_noise_floor_ns"}) {
        assert(!row.at("measurements").contains(removed));
        assert(!row.at("absent").contains(removed));
    }
    // Only the tail quiesce/reopen measurement is missing. The logged
    // condition must be exact, deduplicated, and the structural no-peer
    // path must still emit a partial record after one derived horizon.
    Measured only_lane=m;
    only_lane.lane_lost_ns=Series{64};
    only_lane.committed_transactions=1;
    only_lane.tail_committed_transactions=1;
    const auto missing_lane=calibration_missing_planner(only_lane);
    assert(missing_lane.size()==1 && missing_lane[0].name=="lane_lost_ns");
    CalibrationProgressTimeout planner_watch;
    std::string line;
    const uint64_t watch_start=1000000000ull;
    assert(planner_watch.observe(only_lane,missing_lane,watch_start,false,&line));
    assert(line=="calibration waiting: lane_lost_ns (no tail quiesce/reopen interval yet)");
    assert(!planner_watch.observe(only_lane,missing_lane,watch_start+1,false,&line));
    assert(!calibration_settled(only_lane,true,true));
    const uint64_t wait_gap=uint64_t(std::max({
        only_lane.delta_gap_ns.max(),double(only_lane.effect_cadence_ns),
        double(only_lane.effect_cadence_ns*2)}));
    const uint64_t wait_span=wait_gap*only_lane.drift_capacity;
    assert(!planner_watch.expired(watch_start+1,only_lane,
                                  only_lane.effect_cadence_ns,
                                  only_lane.effect_cadence_ns*2));
    Measured later_large_gap=only_lane;
    later_large_gap.delta_gap_ns.add(double(wait_span*10));
    assert(!planner_watch.expired(watch_start+wait_span-1,later_large_gap,
                                  only_lane.effect_cadence_ns,
                                  only_lane.effect_cadence_ns*2));
    assert(planner_watch.expired(watch_start+wait_span,later_large_gap,
                                 only_lane.effect_cadence_ns,
                                 only_lane.effect_cadence_ns*2));
    const auto partial_lane=calibration_record(only_lane,
        {std::string(64,'f'),"flashnext-4gpu","balanced"},"2026-11-06",
        true,"tail large lane has no peer reference on this layout",true);
    assert(partial_lane.at("absent").size()>=1);
    assert(partial_lane.at("absent").at("lane_lost_ns")==
           "no verified tail-helper quiesce-to-reopen interval");
    assert(!partial_lane.at("measurements").contains("lane_lost_ns"));
    assert(partial_lane.at("measurements").contains("copy_bytes_per_s"));
    assert(partial_lane.at("evidence").get<std::string>().find("partial calibration")!=std::string::npos);

    Measured no_copy=m;
    no_copy.slowest_helper_copy_bps=Series{64};
    const auto absent_copy=calibration_record(no_copy,
        {std::string(64, 'b'), "flashnext-4gpu", "balanced"}, "2026-11-06");
    assert(absent_copy.at("absent").at("copy_bytes_per_s").is_string());
    assert(!absent_copy.at("measurements").contains("copy_bytes_per_s"));
    assert(row.at("absent").at("abort_quiescent_ns") == "none observed");
    assert(row.at("absent").size() + row.at("measurements").size() == 19);

    // The relative peer can remain unpublished indefinitely. Wait for the
    // measured heat-gap horizon, not an arbitrary wall-clock timeout. All
    // non-effect evidence must still be present before emitting the record.
    Measured no_effect=m;
    no_effect.effect=EffectSamples{};
    no_effect.effect_window_noise_ladder.clear();
    no_effect.abort_ns.add(100.0); // every non-effect field now measured
    assert(!calibration_settled(no_effect));
    // Effect-only evidence is bounded even when no publisher becomes active.
    CalibrationProgressTimeout no_publishers;
    std::string effect_line;
    const uint64_t effect_start=1000000000ull;
    assert(no_publishers.observe(no_effect,{},effect_start,true,&effect_line));
    assert(effect_line=="calibration waiting: effect signal (published relative-service windows not converged)");
    assert(!no_publishers.observe(no_effect,{},effect_start+1,true,&effect_line));
    const uint64_t effect_span=uint64_t(CalibrationProgressTimeout::wait_span(
        no_effect,no_effect.effect_cadence_ns,2*no_effect.effect_cadence_ns));
    assert(!no_publishers.effect_expired(effect_start+1,no_effect,
        no_effect.effect_cadence_ns,2*no_effect.effect_cadence_ns));
    Measured later_effect_gap=no_effect;
    later_effect_gap.delta_gap_ns.add(double(effect_span*10));
    assert(!no_publishers.effect_expired(effect_start+effect_span-1,later_effect_gap,
        no_effect.effect_cadence_ns,2*no_effect.effect_cadence_ns));
    assert(no_publishers.effect_expired(effect_start+effect_span,later_effect_gap,
        no_effect.effect_cadence_ns,2*no_effect.effect_cadence_ns));
    assert(calibration_settled(no_effect,true));
    // A long pre-request/helper warm-up must never consume the outage budget.
    EffectSignalTimeout warming;
    const uint64_t start=1000000000ULL;
    const uint64_t gap=uint64_t(no_effect.delta_gap_ns.max());
    const uint64_t age=2*no_effect.effect_cadence_ns;
    const uint64_t period=std::max({gap,no_effect.effect_cadence_ns,age});
    const uint64_t horizon=period*std::max<size_t>(3,no_effect.drift_capacity);
    warming.observe(false,false,start,true);
    assert(!warming.expired(start+100*horizon,no_effect,no_effect.effect_cadence_ns,age));
    warming.observe(true,false,start+100*horizon,true);
    assert(!warming.expired(start+101*horizon-1,no_effect,no_effect.effect_cadence_ns,age));
    assert(warming.expired(start+101*horizon,no_effect,no_effect.effect_cadence_ns,age));
    EffectSignalTimeout unavailable;
    unavailable.observe(true,false,start,true);
    assert(!unavailable.expired(start+horizon-1,no_effect,no_effect.effect_cadence_ns,age));
    assert(!calibration_settled(no_effect,unavailable.expired(
        start+horizon-1,no_effect,no_effect.effect_cadence_ns,age)));
    assert(unavailable.expired(start+horizon,no_effect,no_effect.effect_cadence_ns,age));
    assert(calibration_settled(no_effect,true)); // planner ready; no effect wait
    const auto structural=calibration_record(no_effect,
        {std::string(64,'f'),"flashnext-4gpu","balanced"},"2026-11-06",true,
        "tail large lane has no peer reference on this layout");
    assert(structural.at("absent").at("effect_window_noise_ladder") ==
           "tail large lane has no peer reference on this layout");
    const auto fallback=calibration_record(no_effect,
        {std::string(64,'f'),"flashnext-4gpu","balanced"},"2026-11-06",true);
    for(const char* field : {"effect_rows_per_s","effect_noise_max_ns",
                             "effect_row_noise_ns","effect_window_noise_ladder"}) {
        assert(!fallback.at("measurements").contains(field));
        assert(fallback.at("absent").at(field).get<std::string>().find(
            "relative tail/peer service unavailable")!=std::string::npos);
    }
    // Partial means missing EFFECT ONLY, not discarding independently
    // observed transactions, accepted heat gaps, slowest ACK copy rate, etc.
    const auto& values=fallback.at("measurements");
    assert(values.at("txn_p50_ns")==10 && values.at("txn_p95_ns")==30);
    assert(values.at("copy_bytes_per_s")==1000000);
    assert(values.at("heat_gap_max_ns")==1000000000);
    assert(values.at("heat_watermarked")==1);
    assert(values.at("marginal_error_ppm")==25000);
    assert(values.at("abort_quiescent_ns")==100);
    assert(values.size()==15 && fallback.at("absent").size()==4);
    assert(!fallback.at("absent").contains("heat_gap_max_ns"));
    assert(!fallback.at("absent").contains("copy_bytes_per_s"));
    // Even if old effect data was once collected, an active outage cannot
    // claim it as fresh evidence. A resumed valid series clears the deadline.
    const auto stale=calibration_record(m,
        {std::string(64,'f'),"flashnext-4gpu","balanced"},"2026-11-06",true);
    assert(!stale.at("measurements").contains("effect_window_noise_ladder"));
    unavailable.observe(true,true,start+horizon,true);
    assert(!unavailable.expired(start+2*horizon,no_effect,no_effect.effect_cadence_ns,age));
    unavailable.observe(false,false,start+2*horizon,true);
    assert(!unavailable.expired(start+3*horizon-1,no_effect,no_effect.effect_cadence_ns,age));
    // No heat-gap measurement => no timeout can use a configured fallback
    // as if the feed had actually made progress.
    Measured no_gaps=no_effect;
    no_gaps.delta_gap_ns=Series{64};
    assert(!unavailable.expired(start+5*horizon,no_gaps,no_gaps.effect_cadence_ns,age));
    // Calib14 ordering: helper heartbeat reports waiting before any accepted
    // histogram delta; neither a baseline nor elapsed wall time arms a timer.
    // Even preloaded other measurements cannot manufacture heat evidence.
    const uint64_t epoch=1000000000ull;
    const uint64_t cadence=only_lane.effect_cadence_ns;
    const uint64_t age_bound=2*cadence;
    Measured pre_heat=only_lane;
    pre_heat.accepted_heat_deltas=0;
    pre_heat.delta_gap_ns=Series{256};
    pre_heat.drift_intervals=0;
    CalibrationProgressTimeout startup;
    auto pending=calibration_missing_planner(pre_heat);
    assert(startup.observe(pre_heat,pending,epoch,false,&line));
    assert(line.find("heat_watermarked (no accepted heat deltas)")!=std::string::npos);
    assert(!startup.observe(pre_heat,pending,epoch+1000000000000ull,false,&line));
    assert(!startup.expired(epoch+1000000000000ull,pre_heat,cadence,age_bound));
    assert(!startup.effect_expired(epoch+1000000000000ull,pre_heat,cadence,age_bound));
    assert(!calibration_settled(pre_heat,true,true));
    // The first accepted delta opens the feed but is not a measured gap:
    // no condition can expire on the second's initial heartbeat.
    pre_heat.accepted_heat_delta(true);
    pre_heat.drift_intervals=1;
    pending=calibration_missing_planner(pre_heat);
    startup.observe(pre_heat,pending,epoch+1000000000000ull,false,&line);
    assert(!startup.expired(epoch+2000000000000ull,pre_heat,cadence,age_bound));
    pre_heat.accepted_heat_delta(true); // second accepted delta establishes the gap
    pre_heat.drift_intervals=2;
    pre_heat.delta_gap_ns.add(double(cadence));
    const uint64_t basis_start=epoch+2000000000000ull;
    startup.observe(pre_heat,calibration_missing_planner(pre_heat),basis_start,false,&line);
    const uint64_t span=uint64_t(CalibrationProgressTimeout::wait_span(
        pre_heat,cadence,age_bound));
    assert(span>0);
    assert(!startup.expired(basis_start+span-1,pre_heat,cadence,age_bound));
    // A missing measured condition with no further progress eventually emits
    // exactly one partial calibration record with its true missing reason.
    assert(startup.expired(basis_start+span,pre_heat,cadence,age_bound));
    const auto stalled=calibration_record(pre_heat,
        {std::string(64,'h'),"flashnext-4gpu","balanced"},"2026-11-06",
        true,"tail large lane has no peer reference on this layout",true);
    assert(stalled.at("evidence").get<std::string>().find("partial calibration")!=std::string::npos);
    assert(stalled.at("absent").contains("lane_lost_ns"));

    // Copy ACK cannot be timed before a committed transaction. In contrast,
    // the lane-loss clock requires a COMMITTED move involving the tail;
    // committing only on other helpers is not its first opportunity.
    Measured no_tx=only_lane;
    no_tx.slowest_helper_copy_bps=Series{64};
    no_tx.committed_transactions=0;
    no_tx.tail_committed_transactions=0;
    CalibrationProgressTimeout tx_watch;
    auto missing_tx=calibration_missing_planner(no_tx);
    tx_watch.observe(no_tx,missing_tx,epoch,false,nullptr);
    assert(!tx_watch.expired(epoch+100*span,no_tx,cadence,age_bound));
    no_tx.committed_transactions=1;
    tx_watch.observe(no_tx,missing_tx,epoch+100*span,false,nullptr);
    assert(tx_watch.expired(epoch+101*span,no_tx,cadence,age_bound));
    Measured tail_tx=only_lane;
    tail_tx.tail_committed_transactions=0;
    CalibrationProgressTimeout tail_watch;
    tail_watch.observe(tail_tx,missing_lane,epoch,false,nullptr);
    assert(!tail_watch.expired(epoch+100*span,tail_tx,cadence,age_bound));
    tail_tx.tail_committed_transactions=1;
    tail_watch.observe(tail_tx,missing_lane,epoch+100*span,false,nullptr);
    assert(tail_watch.expired(epoch+101*span,tail_tx,cadence,age_bound));

    // Genuine progress in the unmet condition, not a heartbeat or another
    // counter, renews its frozen deadline. This applies to drift retention
    // as well as to effect-only quiet-window measurements.
    Measured drifting=m;
    drifting.drift_half_life_observations=0;
    drifting.drift_horizon_observations=100;
    CalibrationProgressTimeout drift_watch;
    auto missing_drift=calibration_missing_planner(drifting);
    drift_watch.observe(drifting,missing_drift,epoch,false,nullptr);
    assert(!drift_watch.expired(epoch+span-1,drifting,cadence,age_bound));
    drifting.drift_horizon_observations+=100;
    missing_drift=calibration_missing_planner(drifting);
    drift_watch.observe(drifting,missing_drift,epoch+span-1,false,nullptr);
    assert(!drift_watch.expired(epoch+span,drifting,cadence,age_bound));
    assert(drift_watch.expired(epoch+2*span-1,drifting,cadence,age_bound));
    CalibrationProgressTimeout effect_watch;
    effect_watch.observe(no_effect,{},epoch,true,nullptr);
    assert(!effect_watch.effect_expired(epoch+span-1,no_effect,cadence,age_bound));
    Measured growing_effect=no_effect;
    growing_effect.effect.quiet_comparisons++;
    effect_watch.observe(growing_effect,{},epoch+span-1,true,nullptr);
    assert(!effect_watch.effect_expired(epoch+span,growing_effect,cadence,age_bound));
    assert(effect_watch.effect_expired(epoch+2*span-1,growing_effect,cadence,age_bound));

    // The separate publisher timeout cannot run across the pre-feed phase.
    EffectSignalTimeout prefeed_effect;
    prefeed_effect.observe(true,false,epoch,false);
    assert(!prefeed_effect.expired(epoch+100*span,pre_heat,cadence,age_bound));
    prefeed_effect.observe(true,false,epoch+100*span,true);
    assert(!prefeed_effect.expired(epoch+100*span+1,pre_heat,cadence,age_bound));
    Measured entirely_missing{};
    assert(calibration_settled(entirely_missing,true)); // bounded timeout emits incomplete record
    assert(!calibration_settled(entirely_missing,true,true)); // structural no-peer awaits planner
    assert(calibration_settled(no_effect,true,true)); // structural no-peer with planner ready
    const auto partial=calibration_record(entirely_missing,
        {std::string(64,'g'),"flashnext-4gpu","balanced"},"2026-11-06",true);
    assert(partial.at("absent").contains("drift_half_life_observations"));
    assert(partial.at("absent").contains("effect_window_noise_ladder"));
}
