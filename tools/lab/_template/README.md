# Experiment template

Plan item: <HI/RD/RE/...>
Status: active
Owner: <name or team>
Question state: open | answered | abandoned

## Question

What exact question does this experiment answer?

## Inputs

...

## Outputs

Generated outputs go under `artifacts/lab/<experiment>/`.

## Runtime

GPU required: yes/no
Real compilation required: yes/no
Mutates canonical BigCherry state: yes/no

## Safety

- Canonical-state mutation: none unless explicitly justified here.
- Do not import this experiment from `bigcherry` production, tests, or maintained analysis.
- Stop conditions and cleanup steps: ...

## Disposition

When complete, record the result and choose exactly one:

- delete the experiment and its generated outputs
- graduate the durable capability to maintained analysis
- graduate the durable capability to product

Graduation is a separate implementation/validation change; this directory is never evidence authority.
