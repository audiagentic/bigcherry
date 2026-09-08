# 1257_nro08_topk_wave32

**Status:** untested
**Group:** nasone-rdna
**Plan item:** NRO08

## What it does

Adds an unwired wave32 TOP-1 reduction primitive on top of NRO07.

## Why

The source follow-up reduces LDS/barrier traffic with wave32 shuffles. It must be validated separately from the hybrid TOP_K algorithm itself.

## Upstream

Local staged adaptation of nasone commit `7f1d25f7e05cb34053d317a0f72f601d846c8662`.
