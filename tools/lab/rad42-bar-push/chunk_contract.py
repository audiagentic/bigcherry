"""Card-free model of a proposed CPU-BAR chunk handoff (NOT production protocol).

A request is identified by placement epoch, generation and chunk number. This
model deliberately rejects early publication, stale tickets, and slot reuse
before the consumer retires. Real CPU/GPU memory visibility requires hardware
proof; this model tests only the lifecycle.
"""
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class Ticket:
    epoch: int
    generation: int
    chunk: int

    def valid(self):
        return self.epoch > 0 and self.generation > 0 and self.chunk >= 0


class Phase(Enum):
    FREE = auto()
    GPU_WRITING = auto()
    CPU_PUSHING = auto()
    READY = auto()
    CONSUMING = auto()
    RETIRED = auto()
    FAULT = auto()


class HandoffError(RuntimeError):
    pass


class ChunkSlot:
    """One buffer slot, with explicit producer and consumer ownership."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.phase = Phase.FREE
        self.ticket = None
        self.bytes_written = 0

    def _expect(self, ticket, phase):
        if ticket != self.ticket or self.phase != phase:
            raise HandoffError(
                f"expected {self.ticket!r} at {phase.name}, "
                f"got {ticket!r} at {self.phase.name}"
            )

    def start(self, ticket):
        if not ticket.valid() or self.phase not in (Phase.FREE, Phase.RETIRED):
            raise HandoffError("invalid ticket or buffer still in use")
        if self.ticket is not None and (
            ticket.epoch < self.ticket.epoch or
            (ticket.epoch == self.ticket.epoch and
             ticket.generation <= self.ticket.generation)
        ):
            raise HandoffError("stale generation")
        self.ticket = ticket
        self.phase = Phase.GPU_WRITING
        self.bytes_written = 0

    def gpu_result_ready(self, ticket):
        self._expect(ticket, Phase.GPU_WRITING)
        self.phase = Phase.CPU_PUSHING

    def publish(self, ticket, bytes_written, *, cpu_write_drained):
        self._expect(ticket, Phase.CPU_PUSHING)
        if not cpu_write_drained:
            raise HandoffError("cannot publish before CPU BAR writes drain")
        if bytes_written <= 0 or bytes_written > self.capacity:
            raise HandoffError("invalid payload length")
        self.bytes_written = bytes_written
        self.phase = Phase.READY

    def consume(self, ticket):
        self._expect(ticket, Phase.READY)
        self.phase = Phase.CONSUMING

    def retire(self, ticket):
        self._expect(ticket, Phase.CONSUMING)
        self.phase = Phase.RETIRED

    def fault(self, ticket):
        if ticket != self.ticket or self.phase in (Phase.FREE, Phase.RETIRED):
            raise HandoffError("cannot fault a foreign or retired ticket")
        self.phase = Phase.FAULT

    def recover(self, *, quiesced):
        if self.phase != Phase.FAULT or not quiesced:
            raise HandoffError("recovery requires fault and complete quiescence")
        self.phase = Phase.RETIRED
