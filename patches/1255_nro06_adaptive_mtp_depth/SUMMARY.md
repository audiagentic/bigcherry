# 1255_nro06_adaptive_mtp_depth

**Status:** untested
**Group:** nasone-rdna
**Plan item:** NRO06

## What it does

Adds the pure adaptive MTP depth controller state machine beside the existing MTP implementation. It is not yet connected to a new CLI/speculative mode.

## Why

The controller can be exhaustively tested without changing runtime behavior before plumbing adaptive draft caps through requests and sequence state.

## Upstream

Local staged adaptation of nasone commit `10579a7365a3bc86c4f8e41aaab20e73e1571e5e`.
