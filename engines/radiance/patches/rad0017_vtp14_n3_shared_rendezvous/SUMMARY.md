# rad0017_vtp14_n3_shared_rendezvous

**Status:** untested
**Plan item:** VTP14

Scope: the all-reduce rendezvous that a tensor-parallel world across libr3 (XTX) and libr4d (R9700) ranks needs.
Without it, such a world hangs in declare; that was reproduced on cards for TP2.

## Summary

Moves libr4d's all-reduce rendezvous, and with it libr3's, into one process-wide slot that the engine exports, so
ranks running different kernel libraries meet at one barrier. A failed share or a timeout poisons the rendezvous and
fails fast with the reason.

## Outcome

Card-free tests only so far. The shared slot is tested by `tests/bc_shared_slot_test.cpp`, and libr3's rewrite check
passes on the patched source. The mixed-card all-reduce is still to be proven on hardware under the gpu lease.

## Detail

See README.md and docs/planning/completed/run-dense-27b-virtual-tp/VTP14.md.
