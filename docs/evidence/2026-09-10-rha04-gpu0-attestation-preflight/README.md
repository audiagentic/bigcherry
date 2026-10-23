# RHA04 GPU0 identity-bound attestation preflight

Date: 2026-09-10  
Role: Brutus build server, resolved through environment settings  
Status: **attestation verified; no throughput claim**

This is the untimed diagnostic preflight introduced by the RHA04 capture
contract. It used the same binary, model, production arguments, and device
visibility as the intended timed cell, with only `--verbosity 5` added. The
timed server-bench arm was not run in this artifact.

The server reported the expected physical identity:

- backend: `ROCm`
- architecture: `gfx1100`
- PCI locator: `0000:03:00.0`
- binary SHA-256: `249618ad89ba98aa2e8c658022f48b9ff5f17e03194b5fcba54139e5e7908ce6`
- model SHA-256: `8ae8fd04e30c5a4af3ef72654729fb0d7f33615cac1ba335dd88084a56287934`
- correlation ID: `7631a484104b4b6aa5cb9a3317b17021`

The preflight shut down cleanly through POSIX SIGINT (`requested=true`,
`forced=false`, return code 0). The selected binary returned HTTP 404 for the
shutdown endpoint in an earlier attempt; the implementation now follows the
arm's declared shutdown method, so this is a valid clean preflight rather than
a forced-termination result.

Raw artifacts:

- `preflight.json` — SHA-256 `db474e9897705bc3953a825e1efe338b2163bdd1322487b2413907920343349e`
- `server.log` — SHA-256 `32b44e7220a4ce3e133f608d84d2844762392b8fd08e0fa70824bac633e9ed1a`

This evidence authorizes physical identity admission for a matching timed arm;
it is not a performance result and does not complete RHA04's matrix.
