# 1250_nro01_allreduce_q8_wire

**Status:** untested
**Group:** nasone-rdna
**Plan item:** NRO01

## What it does

Adds disabled-by-default Q8_0 wire-format primitives and threshold state to the internal AllReduce implementation. The draft does not yet route live reductions through Q8; that wiring is intentionally held until synthetic numerical fixtures are in place.

## Why

Q8_0 can reduce PCIe wire bytes for large FP32 reductions, but it is lossy and must be qualified independently from residual fusion or P2P transport.

## Upstream

Local atomic adaptation informed by nasone32/llama.cpp-RDNA3-7900xtx-opt commit `e06dcf6300718227cb8cfda9e61fb12ccb693418` on BigCherry pin `b10705`.
