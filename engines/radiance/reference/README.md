# Radiance reference inventories (inert data)

This directory holds engine-specific design inventories, **not** runtime
configuration. It is deliberately outside `engines/radiance/configs/`:
`bc-serve --configs engines/radiance/configs` must never ingest these records.

`serving-settings-inventory.toml` is the historical RSC01/RSC04 audited
classification of launcher, service, helper and engine settings. Its original
path was `config/serving-settings.toml`. It is not read by production
serving. The actual strict native C++ serving schema and parser under
`engines/radiance/components/serving/` are authoritative; never generate
runtime defaults from this inventory or treat an entry as active simply
because it appears here.

When changing or retiring the inventory, compare it against the current
C++ schema and the RSC04/RSC12 ownership plans; document any superseded
entries, link to a reviewed decision, and keep historical findings
available. Do not turn this directory into a second configuration authority.
