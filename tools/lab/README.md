# Tooling lab

`tools/lab/<plan-topic>/` is for temporary, plan-owned investigation. Lab code is not a Python package, is not evidence authority, and must never be imported by production.

Each experiment must state its question, inputs, outputs, GPU/build requirements, canonical-state mutation, safety notes, and disposition. Generated outputs belong under `artifacts/lab/<experiment>/`.

Lab scripts must be self-contained: they may import third-party/runtime libraries, but must not be imported by `bigcherry` production modules, tests, or maintained analysis. Do not add `tools/lab/__init__.py`; lab discovery is filesystem- and template-based, not package-based. Keep one experiment per plan-owned directory and record the plan item in its README.

Do not commit copies of upstream llama.cpp source under a lab directory. Reviewers and CI should materialize the repository's pinned `vendor/llama.cpp` checkout and inspect the exact pinned revision; committed `vendor-b*` snapshots become stale at the next pin bump and duplicate upstream source.

When the question is closed, delete the experiment or graduate only the durable capability to maintained analysis/product code. Graduation requires a separate implementation and validation slice; copying a lab script into production is not graduation.
