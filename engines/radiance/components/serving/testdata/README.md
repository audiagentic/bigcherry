# RSC02 S12 power acceptance fixtures

power_qualified_inputs.json contains four checked-in synthetic hardware snapshots and power policies adapted from the PWR09 fake-driver cases. They are not evidence of a leased physical GPU run.

Regeneration command from repository root:

    PYTHONPATH=tools python3 tools/tests/hardware/generate_cpp_power_oracle.py

The generator produces power_qualified_golden.json with observable apply writes, restore writes and manifest-bound receipt/request projection. C++ tests load both JSON files as data. Neither the test executable nor bc-serve imports, invokes or links Python.

The C++ implementation derives from the RSC11 S12/v2 manifest requirements, rather than the Python classes. Contract-facing JSON is serialized only by the durable record adapter. The core uses typed v2::Manifest, PCI addresses, roles, physical units, variant power commands, transaction stages, v2::Status/Result and injected hardware/record/sensor interfaces.

Compatibility decisions requiring integration review:
- Device operation order follows the manifest's exact selected role/BDF order, not Python's incidental lexicographic BDF sort. All provided fixtures have equal orders. Differently ordered manifests require an explicit owner decision before hardware rollout.
- Recovery preflights each touched card's original values and controller identity against the current driver-reported ranges before any restore write. Corrupt, unreadable and symlinked journals refuse. This intentionally strengthens Python behaviour.
- Observe produces no power journal or device writes, but a manifest-bound receipt. Applied profiles require independently verified child exit, plus verified lease ownership for each write.
- GFXOFF requires time-ordered no-process and engine-loaded-idle windows on one gfx12 card with matched driver/sensor identity and unanimous independent signals, with no arbitrary numeric thresholds.

Never regenerate golden fixtures from live hardware or use them as runtime dependencies. Legacy Python/shell remains only as a behavioural reference until coordinated RSC02 step-6 retirement.
