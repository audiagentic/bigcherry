---
id: RAD27
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-27T02:15:28.682410+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: S
---

# Engine reports a dead helper as the request error (not out of memory / device error)

## Description

After a helper dies the engine's GPU wait traps; the engine then labels the failed request 'out of memory', 'device error' or 'shape this kernel does not serve' (secondary errors after the queue fault). Add a supported way for the moe-remote plugin to give the engine a terminal reason (helper index, pid) so the request error and log name the helper.

## Steps

1. Find the engine's terminal-error path (radiance server).
2. Patch (rad0002 or a new patch) a plugin hook for a terminal reason.
3. Kill test checks the message names the helper.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

rad20-helper-kill.sh responses name the dead helper in every scenario.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-10-27T02:15:28.682410+00:00 (created-by): Created by agent
