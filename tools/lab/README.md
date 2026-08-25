# Tooling lab

`tools/lab/<plan-topic>/` is for temporary, plan-owned investigation. Lab code is not a Python package, is not evidence authority, and must never be imported by production.

Each experiment must state its question, inputs, outputs, GPU/build requirements, canonical-state mutation, safety notes, and disposition. Generated outputs belong under `artifacts/lab/<experiment>/`.

When the question is closed, delete the experiment or graduate only the durable capability to maintained analysis/product code.
