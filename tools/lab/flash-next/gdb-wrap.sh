#!/bin/bash
# Run a server under gdb so a hang can be inspected: on SIGINT gdb stops the server and prints every thread's stack.
# dev has ptrace_scope=1, so gdb can only trace a process it started; attaching later is refused.
# Use as the WRAP of long-ctx-profile.sh:   WRAP=tools/lab/flash-next/gdb-wrap.sh ... long-ctx-profile.sh <server> <out> timing
# then, when it hangs:                      pkill -INT -x llama-server        (the stacks go to the server log)
exec gdb -q -batch -ex "handle SIGINT stop print nopass" -ex "handle SIGPIPE nostop noprint pass" -ex run \
    -ex "thread apply all bt 20" --args "$@"
