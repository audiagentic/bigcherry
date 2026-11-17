# RSC03 slice 1b on hardware, 2026-11-02 (build 6f285de4, from d2224ad8; flashnext-4gpu, R9700 engine)

One gpu lease (released). Files: plan.live1.json, plan.live2.json, plan.live1-vs-live2.diff, 1a.log, 2.engine.log.

1a  bc-serve --plan-memory alone (engine, no helpers): exited 2 after 4.9 s, failing closed. The engine reached
    plan() without helpers but its start-up stopped "out of memory" ("weights-from-disk: 33.13 GiB of weights a rank
    are read from the container on disk"): with no helper-owned units the plan is a different one. Measuring needs
    the helpers' owner map (interim: serve flow; v1: owner map from the manifest).
1b  launcher start with --engine-arg --plan-memory FILE, twice: both wrote the plan and exited before weights
    ("nothing loaded or allocated past the plan"); launcher wall 54.6 s and 92.7 s (helper bring-up included).
    The two documents are identical except one live reading: the card's free VRAM differed by 8192 bytes,
    carried into vram.claimable, vram.elastic and vram.kv (each 8192 lower in run 2).
2   normal start and stop: every figure its startup log prints matches plan.live1 at the log's rounding.
GPU check: each after-step snapshot caught one exiting process of the run itself (its helper or engine) whose KFD
context had not yet been released; none remained afterwards.

Numbers (plan.live1.json; bytes):
  ngram_table 51200249856 (47.68 GiB, pinned, ram)      embedding 1271398400 (1.18 GiB)    host_arenas 11665408
  checkpoint slot 117938964 (112.48 MiB)  floor 8  want 16 -> wrong, see below
  kv_one_session 5557452800 (5.18 GiB)
  vram: claimable 31198251000 (29.06 GiB)  statics 5800083456 (5.40)  elastic 25398167544 (23.65)  experts 0
        kv 22812035064 (21.25)  stage 2586132480 (2.41)  kv_ceiling 31511797760  expert_ceiling 6351237120
  card: capacity 34208743424  free 33937612800  engine_expert_plane 6351237120  expert_unit_max 2535424
  full start (step 2): checkpoint store auto-sized 43 snapshots (4.72 GiB) of a want of 72 ("9 a sequence").

Finding (fixed in 5cfdf98e): rad0011 sized want_slots with geo_.checkpoint_interval, which configure_kv resolves
only after plan(); it now resolves the chunk geometry itself with chunk_geometry_resolve. Needs one re-measure.
Finding (for the lead): claimable/elastic/kv carry a live reading that moves by a few KiB between starts.

## Re-measure after the want_slots fix (build c4e58d6c, from 5cfdf98e; one gpu lease, released)
3   launcher start with --plan-memory: wrote plan.live3.json and exited before weights (launcher wall 56.6 s).
    checkpoints.want_slots = 72 (was 16), floor 8, slot 117938964: the engine's own "9 a sequence" for 8 sessions.
    Against plan.live1.json everything else is identical except the live reading again: vram.claimable, elastic and
    kv each 16384 bytes lower (the slice 3 check now holds claimable to the planned split, not to exact equality).
    KFD: none of the run's pids (647487 647743 648215 648377) kept a context 3 s after stop.

## A3: bc_plan_memory_check on real plans (2026-11-02; build c4e58d6c for the measurement, the check compiled from RSC03 9f8d8d6b's rad0011)
4   launcher start with --plan-memory: wrote plan.live4.json (launcher wall 112.9 s); KFD clear 3 s after stop; lease released.
    a3-check/driver: expected = plan.live3, live = plan.live4. Results in a3-check/result.txt:
    - real pair: pass;
    - expected n-gram table, embedding, checkpoint slot or one-session KV one byte smaller: each refused, naming the
      consumer, both byte counts and the remedy;
    - live claimable one byte short of the planned split: refused; exactly equal: pass.
Finding: the planned split (statics + kv + experts + stage) equals claimable exactly in every measurement (margin 0 in
live1..live4), because the engine's KV share is the remainder of the elastic plane. Claimable moved in 8 KiB steps over a
16 KiB range across the four starts (31,198,234,616 .. 31,198,251,000). So admission against a cached plan refuses any start
whose claimable reads lower than at measurement; the real pair passed only because live4 read 8 KiB above live3.
