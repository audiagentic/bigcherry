"""Drain Radiance's mover before shutdown frees its activation arena.

Port of engine-patches/0001 at the Radiance 1.3.0 pin 89cee7c.
"""

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="core/engine.cpp",
    description="Drain each rank's mover transfer stream before arena teardown",
    edits=(
        Edit(
            id="drain-mover-before-shutdown",
            anchor=r"^    for \(auto& r : ranks_\) r->ctx\.shutdown\(\);$",
            mode="replace",
            text="""    for (auto& r : ranks_) {
        /* The mover copies into arena-lent slab slots and the prefill stager's buffers at the
         * arena top: drain it before the arena below is freed. */
        if (rad_dev_set(r->device) >= 0)
            if (RadStream s = r->mover.transfer_stream()) rad_stream_sync(s);
        r->ctx.shutdown();
    }""",
            guard=r"if \(rad_dev_set\(r->device\) >= 0\)",
            expect_matches=1,
            rationale="Unique per-rank shutdown loop in Engine destructor",
        ),
    ),
)
