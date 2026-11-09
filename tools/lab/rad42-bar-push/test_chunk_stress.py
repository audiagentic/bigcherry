"""Deterministic card-free interleavings for the proposed per-chunk protocol.

This checks state/identity only. It must NOT be interpreted as proof of a
cross-device memory fence or real GPU publication.
"""
import random
import unittest

from chunk_contract import ChunkSlot, HandoffError, Phase, Ticket


class StressTest(unittest.TestCase):
    def test_chunk_reorder_and_generation_recycling(self):
        rng = random.Random(0xBA42050)
        chunks = 32
        capacity = 256 * 2560 * 2
        buffers = [ChunkSlot(capacity) for _ in range(chunks)]
        seen = set()
        # Every chunk owns a distinct ready word, but the same physical buffer
        # is recycled over request generations only AFTER consumer retirement.
        for epoch in (1, 2):
            for generation in range(1, 201):
                tickets = [Ticket(epoch, generation, i) for i in range(chunks)]
                for step in ("start", "gpu_result_ready", "publish", "consume", "retire"):
                    order = list(range(chunks))
                    rng.shuffle(order)
                    for i in order:
                        slot, ticket = buffers[i], tickets[i]
                        if step == "start":
                            slot.start(ticket)
                            with self.assertRaises(HandoffError):
                                slot.consume(ticket)
                        elif step == "gpu_result_ready":
                            slot.gpu_result_ready(ticket)
                        elif step == "publish":
                            with self.assertRaises(HandoffError):
                                slot.publish(ticket, 128, cpu_write_drained=False)
                            slot.publish(ticket, capacity, cpu_write_drained=True)
                            self.assertEqual(slot.bytes_written, capacity)
                        elif step == "consume":
                            slot.consume(ticket)
                            self.assertNotIn(ticket, seen)
                            seen.add(ticket)
                        elif step == "retire":
                            with self.assertRaises(HandoffError):
                                slot.consume(ticket)
                            slot.retire(ticket)
                self.assertEqual(len(seen), ((epoch - 1) * 200 + generation) * chunks)
        self.assertEqual(len(seen), 12800)
        self.assertTrue(all(slot.phase is Phase.RETIRED for slot in buffers))
        with self.assertRaises(HandoffError):
            buffers[0].start(Ticket(1, 201, 0))

    def test_faulted_chunk_requires_quiescent_recovery(self):
        chunks = [ChunkSlot(1024) for _ in range(3)]
        tickets = [Ticket(1, 1, i) for i in range(3)]
        for b, t in zip(chunks, tickets):
            b.start(t)
            b.gpu_result_ready(t)
        chunks[0].publish(tickets[0], 128, cpu_write_drained=True)
        chunks[0].consume(tickets[0])
        chunks[0].retire(tickets[0])
        chunks[1].fault(tickets[1])
        with self.assertRaises(HandoffError):
            chunks[1].start(Ticket(1, 2, 1))
        with self.assertRaises(HandoffError):
            chunks[1].recover(quiesced=False)
        chunks[1].recover(quiesced=True)
        chunks[1].start(Ticket(1, 2, 1))
        # An unrelated live chunk must not be silently recycled.
        with self.assertRaises(HandoffError):
            chunks[2].start(Ticket(1, 2, 2))

    def test_reject_invalid_allocations_and_tickets(self):
        with self.assertRaises(ValueError):
            ChunkSlot(0)
        s = ChunkSlot(128)
        for t in (Ticket(0, 1, 0), Ticket(1, 0, 0), Ticket(1, 1, -1)):
            with self.assertRaises(HandoffError):
                s.start(t)
        self.assertEqual(s.phase, Phase.FREE)


if __name__ == "__main__":
    unittest.main()
