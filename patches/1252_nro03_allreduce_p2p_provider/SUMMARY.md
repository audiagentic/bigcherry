# 1252_nro03_allreduce_p2p_provider

**Status:** untested
**Group:** nasone-rdna
**Plan item:** NRO03

## What it does

Adds a disabled P2P request flag and a source-current peer-copy helper. No AllReduce dispatch uses it yet.

## Why

The source fork's fixed issuer is unsafe for BigCherry's observed gfx1100 directionality. The draft encodes source-current push semantics before transport scheduling is implemented.

## Upstream

Local correctness-first adaptation of nasone commit `7c5bb5cb991670676b89cddc5077c9456c5cf70e`.
