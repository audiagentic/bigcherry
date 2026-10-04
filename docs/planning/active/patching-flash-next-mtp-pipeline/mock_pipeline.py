#!/usr/bin/env python3
"""FMTP04 control-state oracle.

Models only the semantic control rule:
- target verifies a front draft;
- a concurrent MTP continuation produces bridge + tail;
- tail is reusable only on full-front acceptance + bridge equality;
- epoch invalidation rejects stale completion.

This deliberately does not simulate llama.cpp KV/recurrent memory or HIP execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random


@dataclass(frozen=True)
class Ahead:
    epoch: int
    base: int
    bridge: int | None
    tail: tuple[int, ...]


def guess_chain(
    prefix: list[int],
    truth: list[int],
    count: int,
    p_correct: float,
    salt: int,
    tag: str,
) -> list[int]:
    out: list[int] = []
    for j in range(count):
        pos = len(prefix) + j
        if pos >= len(truth):
            break

        h = hashlib.blake2b(
            f"{salt}|{tag}|{prefix}|{out}|{pos}".encode(), digest_size=8
        ).digest()
        raw = int.from_bytes(h, "little")
        u = raw / 2**64

        # Once the speculative prefix diverges, the continuation is deliberately
        # treated as unrelated to truth; it cannot become valid again by chance.
        prefix_is_true = prefix + out == truth[:pos]
        if prefix_is_true and u < p_correct:
            tok = truth[pos]
        else:
            tok = (truth[pos] + 1 + raw % 1009) % 10007
            if tok == truth[pos]:
                tok = (tok + 1) % 10007
        out.append(tok)
    return out


def accepted(prefix: list[int], draft: list[int], truth: list[int]) -> int:
    n = 0
    for tok in draft:
        pos = len(prefix) + n
        if pos >= len(truth) or tok != truth[pos]:
            break
        n += 1
    return n


def run_trial(length: int, block: int, p_correct: float, salt: int) -> tuple[int, int, int]:
    truth = [random.Random(salt * 1000003 + i).randrange(10000) for i in range(length)]

    committed: list[int] = []
    epoch = 0
    promotions = 0
    flushes = 0
    stale_rejected = 0

    def fresh_front() -> list[int]:
        return guess_chain(
            committed,
            truth,
            block,
            p_correct,
            salt,
            f"fresh:{epoch}:{len(committed)}",
        )

    front = fresh_front()

    while len(committed) < len(truth):
        # Continue beyond the current front. Token 0 predicts the target-extra
        # token; only token 1 onward can become the next verification draft.
        chain = guess_chain(
            committed + front,
            truth,
            block + 1,
            p_correct,
            salt,
            f"ahead:{epoch}:{len(committed)}",
        )
        ahead = Ahead(
            epoch=epoch,
            base=len(committed),
            bridge=chain[0] if chain else None,
            tail=tuple(chain[1:]),
        )

        n_acc = accepted(committed, front, truth)

        if n_acc == len(front) and front:
            committed.extend(front)

            if len(committed) < len(truth):
                # Target is authoritative for the extra token produced by a
                # verification step after a fully accepted draft.
                target_extra = truth[len(committed)]
                committed.append(target_extra)

                promote = (
                    ahead.epoch == epoch
                    and ahead.base + len(front) == len(committed) - 1
                    and ahead.bridge == target_extra
                    and bool(ahead.tail)
                )
                if promote:
                    front = list(ahead.tail)
                    promotions += 1
                    continue
        else:
            # Partial acceptance commits only the matching prefix, then the
            # target supplies the correction/next authoritative token.
            committed.extend(front[:n_acc])
            if len(committed) < len(truth):
                committed.append(truth[len(committed)])

        # Rejection, bridge mismatch or unusable tail invalidates this branch.
        old_epoch = epoch
        epoch += 1
        flushes += 1

        # Simulate a completion arriving after invalidation. It must be stale.
        assert ahead.epoch == old_epoch
        assert ahead.epoch != epoch
        stale_rejected += 1

        front = fresh_front()

    assert committed == truth
    return promotions, flushes, stale_rejected


def edge_cases() -> None:
    # Fully correct proposal stream must exercise promotion.
    p, _, _ = run_trial(length=64, block=3, p_correct=1.0, salt=1)
    assert p > 0

    # Fully wrong proposals must repeatedly flush yet preserve target truth.
    _, f, s = run_trial(length=64, block=3, p_correct=0.0, salt=2)
    assert f > 0 and s == f

    # Exercise all intended v1 front sizes, including depth one.
    for block in range(1, 9):
        run_trial(length=50, block=block, p_correct=0.7, salt=100 + block)


def main() -> None:
    edge_cases()

    trials = 5000
    promotions = 0
    flushes = 0
    stale_rejected = 0

    for i in range(trials):
        cfg = random.Random(100000 + i)
        p, f, s = run_trial(
            length=cfg.randint(32, 300),
            block=cfg.randint(1, 8),
            p_correct=cfg.uniform(0.25, 0.98),
            salt=i,
        )
        promotions += p
        flushes += f
        stale_rejected += s

    assert stale_rejected == flushes
    print(
        f"PASS trials={trials} promotions={promotions} "
        f"flushes={flushes} stale_rejected={stale_rejected}"
    )


if __name__ == "__main__":
    main()
