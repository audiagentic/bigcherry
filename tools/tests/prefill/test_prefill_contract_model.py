"""Model-only concurrency contract tests; not llama.cpp backend or GPU tests."""

from dataclasses import dataclass, field
import unittest


@dataclass(frozen=True)
class Task:
    name: str
    resource: str
    duration: int
    after: tuple[str, ...] = ()


class Timeline:
    def __init__(self):
        self.ends = {}
        self.resources = {}
        self.spans = {}

    def add(self, task):
        if task.name in self.ends or task.duration < 0:
            raise ValueError("duplicate task or negative duration")
        if any(dep not in self.ends for dep in task.after):
            raise ValueError("dependency not submitted")
        start = max([self.resources.get(task.resource, 0)] + [self.ends[d] for d in task.after])
        end = start + task.duration
        self.resources[task.resource] = end
        self.ends[task.name] = end
        self.spans[task.name] = (start, end)
        return start, end


class EventRing:
    def __init__(self, slots):
        self.slots = [{} for _ in range(slots)]
        self.generation = [0] * slots

    def acquire(self, slot, children):
        if self.slots[slot]:
            raise RuntimeError("slot still consumed by GPU")
        self.generation[slot] += 1
        self.slots[slot] = {child: False for child in children}
        return slot, self.generation[slot]

    def complete(self, ticket, child):
        slot, generation = ticket
        if generation != self.generation[slot] or child not in self.slots[slot]:
            raise RuntimeError("stale event or unknown consumer")
        self.slots[slot][child] = True

    def release(self, ticket):
        slot, generation = ticket
        if generation != self.generation[slot] or not self.slots[slot] or not all(self.slots[slot].values()):
            raise RuntimeError("incomplete fanout")
        self.slots[slot] = {}


class TileJoin:
    def __init__(self, ranks):
        self.ranks = ranks
        self.entries = {}
        self.identity = None

    def ready(self, rank, uid, t0, nt, shape, value):
        key = uid, t0, nt, shape
        if self.identity is not None and key != self.identity:
            raise RuntimeError("rank tile identity mismatch")
        if rank not in self.ranks or rank in self.entries:
            raise RuntimeError("unknown or duplicate rank")
        self.identity = key
        self.entries[rank] = value

    def reduce(self):
        if set(self.entries) != set(self.ranks):
            raise RuntimeError("consumer before all rank events")
        return sum(self.entries[rank] for rank in self.ranks)


@dataclass
class ExpertStage:
    predicted: set[int] = field(default_factory=set)
    delivered: set[int] = field(default_factory=set)

    def misses(self, required):
        return set(required) - self.delivered

    def execute(self, required):
        if self.misses(required):
            raise RuntimeError("missing exact expert weights")
        return tuple(sorted(required))


class ScratchLease:
    def __init__(self, available_bytes):
        self.available_bytes = available_bytes
        self.state = "idle"
        self.pending = set()

    def borrow(self, phase, bytes_needed):
        if phase != "prefill" or self.state != "idle" or bytes_needed > self.available_bytes:
            return False
        self.state = "borrowed"
        return True

    def restore(self, outstanding):
        if self.state != "borrowed":
            raise RuntimeError("no active lease")
        self.state = "restoring"
        self.pending = set(outstanding)

    def completed(self, event):
        if self.state != "restoring" or event not in self.pending:
            raise RuntimeError("stale restoration event")
        self.pending.remove(event)
        if not self.pending:
            self.state = "idle"

    def decode_ready(self):
        return self.state == "idle"


class PrefillContractTests(unittest.TestCase):
    @staticmethod
    def timeline(serial):
        t = Timeline()
        t.add(Task("prepare0", "host", 1))
        t.add(Task("submit0", "host", 0, ("prepare0",)))
        t.add(Task("target0", "target", 9, ("submit0",)))
        t.add(Task("stage0", "copy", 2, ("target0",)))
        t.add(Task("prepare1", "host", 2, ("stage0",) if serial else ("submit0",)))
        t.add(Task("submit1", "host", 0, ("prepare1",)))
        t.add(Task("target1", "target", 9, ("submit1",)))
        t.add(Task("catch0", "draft", 3, ("stage0", "submit1")))
        return t

    def test_nonblocking_host_prepare_and_draft_target_overlap(self):
        async_t = self.timeline(False)
        serial_t = self.timeline(True)
        self.assertLess(async_t.spans["prepare1"][0], async_t.spans["target0"][1])
        self.assertLess(async_t.spans["catch0"][0], async_t.spans["target1"][1])
        self.assertLess(max(async_t.ends.values()), max(serial_t.ends.values()))

    def test_ring_waits_for_every_child_and_rejects_stale_generation(self):
        ring = EventRing(2)
        a = ring.acquire(0, ("gfx1100_a", "gfx1100_b", "gfx1201"))
        ring.complete(a, "gfx1100_a")
        with self.assertRaises(RuntimeError):
            ring.release(a)
        with self.assertRaises(RuntimeError):
            ring.acquire(0, ("gfx1201",))
        for child in ("gfx1100_b", "gfx1201"):
            ring.complete(a, child)
        ring.release(a)
        b = ring.acquire(0, ("gfx1201",))
        with self.assertRaises(RuntimeError):
            ring.complete(a, "gfx1201")
        ring.complete(b, "gfx1201")
        ring.release(b)

    def test_range_collective_requires_all_ranks_matching_tile(self):
        join = TileJoin((0, 1, 2))
        join.ready(0, 7, 128, 64, (4096, 64), 1)
        join.ready(1, 7, 128, 64, (4096, 64), 2)
        with self.assertRaises(RuntimeError):
            join.reduce()
        with self.assertRaises(RuntimeError):
            join.ready(2, 7, 192, 64, (4096, 64), 3)
        join.ready(2, 7, 128, 64, (4096, 64), 3)
        self.assertEqual(join.reduce(), 6)

    def test_speculative_prefetch_cannot_skip_missed_experts(self):
        stage = ExpertStage(predicted={1, 7}, delivered={1, 7})
        self.assertEqual(stage.misses({1, 7, 9}), {9})
        with self.assertRaises(RuntimeError):
            stage.execute({1, 7, 9})
        stage.delivered.add(9)
        self.assertEqual(stage.execute({1, 7, 9}), (1, 7, 9))

    def test_phase_scratch_lease_restores_before_decode(self):
        lease = ScratchLease(256)
        self.assertFalse(lease.borrow("decode", 64))
        self.assertFalse(lease.borrow("prefill", 257))
        self.assertTrue(lease.borrow("prefill", 128))
        self.assertFalse(lease.borrow("prefill", 64))
        lease.restore(("gpu0", "gpu1"))
        lease.completed("gpu0")
        self.assertFalse(lease.decode_ready())
        with self.assertRaises(RuntimeError):
            lease.completed("gpu0")
        lease.completed("gpu1")
        self.assertTrue(lease.decode_ready())


if __name__ == "__main__":
    unittest.main()
