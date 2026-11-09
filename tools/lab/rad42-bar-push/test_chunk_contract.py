"""No GPUs or ROCm imports: python3 -m unittest discover -p 'test_*.py'."""
import unittest

from chunk_contract import ChunkSlot, HandoffError, Phase, Ticket


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.slot = ChunkSlot(4096)
        self.first = Ticket(1, 1, 0)

    def prepare(self):
        self.slot.start(self.first)
        self.slot.gpu_result_ready(self.first)

    def test_success_and_next_generation(self):
        self.prepare()
        self.slot.publish(self.first, 4096, cpu_write_drained=True)
        self.slot.consume(self.first)
        self.slot.retire(self.first)
        self.slot.start(Ticket(1, 2, 0))
        self.assertEqual(self.slot.phase, Phase.GPU_WRITING)

    def test_no_publish_before_gpu_result(self):
        self.slot.start(self.first)
        with self.assertRaises(HandoffError):
            self.slot.publish(self.first, 12, cpu_write_drained=True)

    def test_no_publish_before_write_drain(self):
        self.prepare()
        with self.assertRaises(HandoffError):
            self.slot.publish(self.first, 12, cpu_write_drained=False)
        self.assertEqual(self.slot.phase, Phase.CPU_PUSHING)

    def test_overflow_or_zero_is_rejected(self):
        self.prepare()
        for size in (0, 4097, -1):
            with self.assertRaises(HandoffError):
                self.slot.publish(self.first, size, cpu_write_drained=True)

    def test_cannot_consume_before_ready(self):
        self.prepare()
        with self.assertRaises(HandoffError):
            self.slot.consume(self.first)

    def test_cannot_reuse_inflight_buffer(self):
        self.prepare()
        with self.assertRaises(HandoffError):
            self.slot.start(Ticket(1, 2, 0))

    def test_stale_or_wrong_chunk_never_opens_slot(self):
        self.prepare()
        with self.assertRaises(HandoffError):
            self.slot.publish(Ticket(1, 1, 1), 32, cpu_write_drained=True)
        with self.assertRaises(HandoffError):
            self.slot.publish(Ticket(1, 0, 0), 32, cpu_write_drained=True)

    def test_no_replay_even_after_retirement(self):
        self.prepare()
        self.slot.publish(self.first, 12, cpu_write_drained=True)
        self.slot.consume(self.first)
        self.slot.retire(self.first)
        with self.assertRaises(HandoffError):
            self.slot.start(self.first)
        with self.assertRaises(HandoffError):
            self.slot.start(Ticket(0, 2, 0))
        with self.assertRaises(HandoffError):
            self.slot.start(Ticket(1, 0, 0))

    def test_fault_requires_quiescence(self):
        self.prepare()
        self.slot.fault(self.first)
        with self.assertRaises(HandoffError):
            self.slot.consume(self.first)
        with self.assertRaises(HandoffError):
            self.slot.recover(quiesced=False)
        self.slot.recover(quiesced=True)
        self.slot.start(Ticket(2, 1, 0))

    def test_older_epoch_rejected(self):
        self.prepare()
        self.slot.publish(self.first, 64, cpu_write_drained=True)
        self.slot.consume(self.first)
        self.slot.retire(self.first)
        self.slot.start(Ticket(2, 1, 0))
        self.slot.gpu_result_ready(Ticket(2, 1, 0))
        self.slot.publish(Ticket(2, 1, 0), 64, cpu_write_drained=True)
        self.slot.consume(Ticket(2, 1, 0))
        self.slot.retire(Ticket(2, 1, 0))
        with self.assertRaises(HandoffError):
            self.slot.start(Ticket(1, 9, 0))


if __name__ == "__main__":
    unittest.main()
