# RHP01-RHP06 CPU-only mock evidence

Source: `tools/lab/radiance/heat-placement/heat_mock.mjs`, executed as a pure JavaScript function against the identical committed source by the planning author; 23 assertions passed. Test scope is **synthetic mechanics**, not actual Radiance/Brutus GPU performance or numerical fidelity.

Replay on an installed Node environment:

```bash
node tools/lab/radiance/heat-placement/heat_mock.mjs
python3 -m unittest tools/tests/radiance/test_heat_placement_mock.py
```

Expected fixed result: `mock-results.json`. No runtime service was started, no GPU kernels executed, and no new inference feature is implemented by these plans.

Mock design omissions needing hardware validation: precise Radiance multi-pass bank writes, GPU completion generation, host run-ahead loss rate, mapping of protected/MTP experts, helper quant/dequant parity, actual GPU memory/transfer budget, atomic placement cutover, 32K prefill performance and model KL.
