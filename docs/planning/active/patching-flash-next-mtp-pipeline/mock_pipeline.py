#!/usr/bin/env python3
"""FMTP control/state oracle.

Models:
- cold rounds that draft a front before target verification;
- hit rounds that reuse a promoted front, then replay that known front through MTP
  under target verification to reconstruct live draft state;
- bridge-gated promotion;
- recursive draft hidden state that intentionally differs from a fresh target reseed;
- stale epoch/frontier/base rejection and short-tail rejection;
- same-thread target-enqueue -> MTP-work -> target-sync timing.

It deliberately does not simulate llama.cpp KV layout or HIP kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random


def h64(*parts: object) -> int:
    data = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def target_hidden(prefix: list[int]) -> int:
    return h64("target", tuple(prefix))


def advance_hidden(hidden: int, token: int, pos: int) -> int:
    return h64("hidden", hidden, token, pos)


def propose(prefix: list[int], truth: list[int], hidden: int, p_correct: float, salt: int) -> int | None:
    pos = len(prefix)
    if pos >= len(truth):
        return None

    raw = h64("proposal", salt, pos, hidden, tuple(prefix[-4:]))
    if prefix == truth[:pos] and raw / 2**64 < p_correct:
        return truth[pos]

    tok = (truth[pos] + 1 + raw % 1009) % 10007
    return (tok + 1) % 10007 if tok == truth[pos] else tok


def generate(
    prefix: list[int],
    truth: list[int],
    hidden: int,
    count: int,
    p_correct: float,
    salt: int,
) -> tuple[list[int], int]:
    cur = list(prefix)
    out: list[int] = []
    for _ in range(count):
        tok = propose(cur, truth, hidden, p_correct, salt)
        if tok is None:
            break
        pos = len(cur)
        hidden = advance_hidden(hidden, tok, pos)
        cur.append(tok)
        out.append(tok)
    return out, hidden


def replay_known(prefix: list[int], known: list[int], hidden: int) -> int:
    """Advance MTP state with fixed proposal tokens; no token is committed here."""
    pos = len(prefix)
    for tok in known:
        hidden = advance_hidden(hidden, tok, pos)
        pos += 1
    return hidden


def accepted(prefix: list[int], draft: list[int], truth: list[int]) -> int:
    n = 0
    for tok in draft:
        pos = len(prefix) + n
        if pos >= len(truth) or tok != truth[pos]:
            break
        n += 1
    return n


@dataclass(frozen=True)
class Ahead:
    epoch: int
    base: int
    front_len: int
    bridge: int | None
    tail: tuple[int, ...]


def run_trial(
    length: int,
    block: int,
    p_correct: float,
    salt: int,
    min_tail: int,
) -> tuple[int, int, int, int, int, int]:
    truth = [random.Random(salt * 1000003 + i).randrange(10000) for i in range(length)]
    rng = random.Random(17000000 + salt)

    committed: list[int] = []
    epoch = 1
    front: list[int] | None = None
    front_source = "fresh"

    promotions = 0
    flushes = 0
    stale_rejected = 0
    promoted_replays = 0
    hidden_mismatch_promotions = 0
    rounds = 0

    while len(committed) < len(truth):
        rounds += 1
        base = len(committed)

        if front_source == "fresh":
            # Cold/flush round: front drafting is serial and produces a live MTP frontier.
            front, h_after_front = generate(
                committed, truth, target_hidden(committed), block, p_correct, salt
            )
        else:
            # Hit round: the promoted tokens exist but their old speculative KV was discarded.
            # Reconstruct the live MTP branch from the authoritative target seed while the
            # target verifies the promoted front.
            assert front
            h_after_front = replay_known(committed, front, target_hidden(committed))
            promoted_replays += 1

        if not front:
            committed.append(truth[len(committed)])
            epoch += 1
            front_source = "fresh"
            front = None
            continue

        # Ahead work always starts from a live state after the current front.
        # Randomly shorten it to model p-min / bounded continuation.
        n_ahead = block + 1
        if rng.random() < 0.15:
            n_ahead = rng.randrange(1, block + 2)

        chain, _ = generate(
            committed + front, truth, h_after_front, n_ahead, p_correct, salt
        )

        # Inject occasional metadata faults that must fail closed.
        base_fault = 1 if rng.random() < 0.01 else 0
        ahead = Ahead(
            epoch=epoch,
            base=base + base_fault,
            front_len=len(front),
            bridge=chain[0] if chain else None,
            tail=tuple(chain[1:]),
        )

        stale = rng.random() < 0.02
        current_epoch = epoch + 1 if stale else epoch

        n_acc = accepted(committed, front, truth)
        promoted = False

        if n_acc == len(front):
            committed.extend(front)
            if len(committed) < len(truth):
                target_extra = truth[len(committed)]
                committed.append(target_extra)

                promote = (
                    ahead.epoch == current_epoch
                    and ahead.base + ahead.front_len == len(committed) - 1
                    and ahead.bridge == target_extra
                    and len(ahead.tail) >= min_tail
                )

                if promote:
                    # Important: a reused tail need not equal what a fresh next-round MTP
                    # reseed would have proposed. Target verification still preserves output.
                    fresh_tail, _ = generate(
                        committed,
                        truth,
                        target_hidden(committed),
                        len(ahead.tail),
                        p_correct,
                        salt,
                    )
                    if list(ahead.tail) != fresh_tail:
                        hidden_mismatch_promotions += 1

                    front = list(ahead.tail)
                    front_source = "promoted"
                    promotions += 1
                    promoted = True
        else:
            committed.extend(front[:n_acc])
            if len(committed) < len(truth):
                committed.append(truth[len(committed)])

        if stale:
            stale_rejected += 1

        # Every parent round is fenced. A promoted child gets a new generation id.
        epoch = max(epoch, current_epoch) + 1

        if not promoted:
            flushes += 1
            front_source = "fresh"
            front = None

    assert committed == truth
    return (
        promotions,
        flushes,
        stale_rejected,
        promoted_replays,
        hidden_mismatch_promotions,
        rounds,
    )


def overlap_timing_cases() -> int:
    """Prove the same-thread schedule has the same ideal device overlap as a worker."""
    rng = random.Random(424242)
    cases = 10000
    for _ in range(cases):
        target_us = rng.randint(50, 5000)
        mtp_us = rng.randint(20, 3000)

        serial = target_us + mtp_us
        same_thread = max(target_us, mtp_us)  # target async submit, MTP work, target sync
        overhang = max(0, mtp_us - target_us)

        assert same_thread <= serial
        assert same_thread - target_us == overhang
    return cases


def edge_cases() -> None:
    # All-correct proposals must sustain promoted rounds and replay them.
    p, _, _, r, _, _ = run_trial(96, 3, 1.0, 1, 1)
    assert p > 0 and r > 0

    # Fully wrong proposals repeatedly flush but cannot alter target output.
    _, f, _, _, _, _ = run_trial(64, 3, 0.0, 2, 1)
    assert f > 0

    # Exercise v1 front depths and minimum-tail filtering.
    for block in range(1, 9):
        for min_tail in range(1, min(3, block) + 1):
            run_trial(64, block, 0.7, 1000 + 10 * block + min_tail, min_tail)


def main() -> None:
    edge_cases()
    timing_cases = overlap_timing_cases()

    totals = [0] * 6
    trials = 5000

    for i in range(trials):
        cfg = random.Random(100000 + i)
        vals = run_trial(
            length=cfg.randint(32, 300),
            block=cfg.randint(1, 8),
            p_correct=cfg.uniform(0.25, 0.98),
            salt=i,
            min_tail=cfg.randint(1, 3),
        )
        totals = [a + b for a, b in zip(totals, vals)]

    promotions, flushes, stale, replays, hidden_diff, rounds = totals
    assert replays == promotions
    assert hidden_diff > 0

    print(
        "PASS "
        f"trials={trials} rounds={rounds} promotions={promotions} "
        f"promoted_replays={replays} hidden_mismatch_promotions={hidden_diff} "
        f"flushes={flushes} stale_rejected={stale} timing_cases={timing_cases}"
    )


if __name__ == "__main__":
    main()
