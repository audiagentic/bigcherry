// Included after the CPU v26 fixture/oracle definitions. Matrix of exact
// helper/rank ACKs in all lifecycle phases, including fault/missing/stale.
static void all_phase_faults(Fixture& f) {
    const AckPhase phases[] = {
        AckPhase::RESERVED, AckPhase::READY, AckPhase::QUIESCED,
        AckPhase::COMMITTED, AckPhase::RETIRED
    };
    for (AckPhase phase : phases) {
        for (const uint32_t bad_actor : {
                f.topology.helpers - 1, f.topology.helpers + f.topology.ranks - 1}) {
            for (int fault = 0; fault < 6; ++fault) {
                check(v::init(f.base, f.layout.bytes, f.shape, f.layers,
                              f.topology, manifest_digest) == v::Refusal::None,
                      "clean v26 before each fault scenario");
                cpu::Epoch e(f.segment, 19);
                check(e.begin(29), "cycle opened");
                if (phase != AckPhase::RESERVED) {
                    ack_phase(f, e, AckPhase::RESERVED);
                    check(e.step() == cpu::Progress::Advanced, "reserve before phase");
                }
                if (phase != AckPhase::RESERVED && phase != AckPhase::READY) {
                    ack_phase(f, e, AckPhase::READY);
                    check(e.step() == cpu::Progress::Advanced && e.quiesce(),
                          "prepared before phase");
                }
                if (phase == AckPhase::COMMITTED || phase == AckPhase::RETIRED) {
                    ack_phase(f, e, AckPhase::QUIESCED, 20);
                    check(e.step() == cpu::Progress::Advanced, "quiesced before phase");
                    auto* g = f.segment.generation(e.control().pending_slot);
                    g->epoch = e.control().pending_epoch;
                    g->digest = e.control().pending_digest;
                    g->state = uint32_t(GenerationState::ACTIVE);
                }
                if (phase == AckPhase::RETIRED) {
                    ack_phase(f, e, AckPhase::COMMITTED);
                    check(e.step() == cpu::Progress::Advanced, "committed before retirement");
                }
                ack_phase(f, e, phase, phase == AckPhase::QUIESCED ? 20 : 0);
                auto* a = f.segment.ack(bad_actor);
                if (fault == 0) *a = {}; // missing helper or rank
                if (fault == 1) --a->transaction_id; // earlier transaction
                if (fault == 2) ++a->epoch; // same transaction, wrong epoch
                if (fault == 3) a->digest ^= 1; // wrong source-map digest
                if (fault == 4) a->generation_slot ^= 1; // stale generation
                if (fault == 5) a->fault_code = 1; // current participant fault
                const auto progress = e.step();
                if (fault <= 1) {
                    check(progress == cpu::Progress::Waiting && e.refusal() == cpu::Refusal::None,
                          "lost/stale ACK waits in every lifecycle phase");
                    ack_phase(f, e, phase, phase == AckPhase::QUIESCED ? 20 : 0);
                    check(e.step() == cpu::Progress::Advanced, "exact recovery after lost ACK");
                } else {
                    check(progress == cpu::Progress::Refused &&
                          e.refusal() == (fault == 5 ? cpu::Refusal::ParticipantFault :
                                                     cpu::Refusal::WrongAckIdentity),
                          "wrong digest, epoch, generation or fault refused every phase");
                    ack_phase(f, e, phase, phase == AckPhase::QUIESCED ? 20 : 0);
                    check(e.step() == cpu::Progress::Refused, "refusal is terminal");
                }
            }
        }
    }
    check(v::init(f.base, f.layout.bytes, f.shape, f.layers,
                  f.topology, manifest_digest) == v::Refusal::None, "fresh drain test");
    cpu::Epoch e(f.segment, 40);
    check(e.begin(41), "begin drain test");
    ack_phase(f, e, AckPhase::RESERVED);
    check(e.step() == cpu::Progress::Advanced, "reserve drain test");
    ack_phase(f, e, AckPhase::READY);
    check(e.step() == cpu::Progress::Advanced && e.quiesce(), "ready drain test");
    ack_phase(f, e, AckPhase::QUIESCED, 30);
    ++f.segment.rank_ack(f.topology.ranks-1)->drained_step;
    check(e.step() == cpu::Progress::Refused &&
          e.refusal() == cpu::Refusal::Undrained,
          "all ACKs cannot fabricate a consistent drain cutoff");
}
