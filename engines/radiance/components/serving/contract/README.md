# Serving manifest v1 contract registry

`serving-v1.toml` is the one authoritative description of the serving manifest `bigcherry/v1` wire contract
(specification: `docs/design/serving-manifest-v1/spec.md`). Everything else about the contract is generated from it by
`generate.cpp` into `../generated/`, and committed:

| Generated file | What |
|---|---|
| `serving-v1.schema.json` | JSON Schema 2020-12. Also committed as `docs/design/serving-manifest-v1/serving-manifest.v1.schema.json` (identical bytes). |
| `serving_types.h` | The typed model (`bc::serving::v1`): one struct per object, enum classes with wire-name tables, `kFields` (every field's path, kind, class, requiredness, Radiance binding and doc), `kFingerprint`. |
| `serving-fields.md` | The field reference. |
| `serving_codec.h`, `serving_codec.cpp` | One strict decoder per struct (`v1/error.h` codes, human paths) and `apply()` for each patch def. |
| `contract.fingerprint` | Two lines: `schema sha256:...` and `semantics sha256:...` (below). |

Two fingerprints, both pinned in `serving_types.h` (`kFingerprint`, `kSemanticFingerprint`):

- **schema**: the RFC 8785 JCS SHA-256 of the validation schema with the annotation keywords (`title`, `description`,
  `$comment`, `examples`, `default`, `deprecated`, `readOnly`, `writeOnly`) removed from *schema positions only*: the
  walk enters subschemas (`properties`, `items`, `allOf`, `if`, ...) and never literal instance data, so a change to an
  object-valued `const` or `enum` changes it. It identifies what validates.
- **semantics**: the same schema plus what the contract freezes that the schema does not express: every field's
  provenance class, the lifecycle, the Radiance bindings (`radiance_option`, `radiance_config`), the reader's error
  codes (`error_class`, `mismatch`) and the calibration key order. It identifies the closed v1 contract. C++ names and docs are excluded (renaming generated code is not a wire
  change, and a consumer would not compile).

v1 is closed (spec §8): a change to either fingerprint is a `bigcherry/v2` change. Correcting documentation changes
neither and is allowed.

## Regenerate

```
cmake --build <serving build> --target bc_contract_gen
<serving build>/bc_contract_gen engines/radiance/components/serving/contract/serving-v1.toml \
    engines/radiance/components/serving/generated
cp engines/radiance/components/serving/generated/serving-v1.schema.json \
    docs/design/serving-manifest-v1/serving-manifest.v1.schema.json
```

CI runs `bc_contract_check` (a card-free component test): it regenerates in memory and fails, naming the file and the
first differing line, when any committed copy differs; it also compiles the committed `serving_types.h` and checks its
fingerprint. `bc_contract_test` covers the loader's refusals. The Python contract test
(`tools/tests/core/test_serving_manifest_v1_contract.py`) checks the two schema copies are byte-identical and validates
the examples against them.

## Format

All numbers are integers. A float anywhere in the registry, a `number` kind, or a float inside a `constraints` fragment is
refused. Every table's keys are closed: an unknown key is refused, naming it.

- `[contract]`: `api_version`, `kind`, `contract_version` (must match the `api_version` suffix), `schema_dialect`,
  `schema_id`, `title`, `namespace` (of the typed model), `lifecycle` (every value's provenance lifecycle).
- `[[class_default]]`: `path`, `class` (`envelope`, `policy`, `fact`, `derived`, `tuned`, `diagnostic`). The longest
  matching path prefix gives a field its provenance class; a path has one rule. Every value in the document must be
  covered. `envelope` is not a provenance class: it marks the integrity fields (`apiVersion`, `kind`, `metadata`,
  `provenance`), which identify the document rather than configure it, carry no provenance record of their own and are
  covered only by the root `""` record (spec §7). The other classes are exactly the provenance record's `class` values.
- `[[calibration_key]]`: `key`, `doc`, and the value's type as `value_kind` (`ref`, `integer`, `string`, `enum`,
  `boolean`) with `value_ref`, `value_minimum`, `value_maximum`, `value_exclusive_maximum`, `value_min_length`,
  `value_pattern`, `value_values`. The schema's calibration `key` enum and its per-key value rules come from this table.
- `[[def]]`: a shared type under `$defs`: `name`, `doc`, and `kind` = `integer` or `string` (with scalar constraints),
  `object` (with `cpp`; its members are `[[field]]`s under `$defs/<name>/`), or `patch` (`of` = an object def, `cpp`):
  every member of `of`, recursively, made optional, each object a nonempty partial map (spec §2.2). A patch def has no
  fields of its own.
- `[[field]]`: one schema node, parents before children, members in document order. `path` is a JSON Pointer from the
  document root (`""` is the root), or `$defs/<name>/...` inside an object def; `[]` is an array's item and `*` a map's
  value. Keys:
  - `kind`: `object` (needs `cpp`, a C++ type name; optional `min_properties`, `constraints`), `map` (`key_ref`,
    `required_keys`, `min_properties`; its value is the `[[field]]` `<path>/*`), `array` (`item_kind`: `object`, whose
    item is the `[[field]]` `<path>/[]`, `tuple` with `tuple` = element defs, or a scalar kind with `item_ref`,
    `item_min_length`, `item_minimum`, `item_values`, `item_pattern`; `min_items`, `max_items`, `unique_items`),
    `integer` (`minimum`, `maximum`, `exclusive_maximum`, `const`), `string` (`min_length`, `pattern`, `format`,
    `const`), `boolean`, `enum` (string `values`, `cpp_type`), `int_enum` (integer `values`), `ref` (`ref` = a def),
    `calibration_key` and `calibration_value` (the two members of `/spec/calibration/[]`).
  - `doc` (required, nonempty); `optional` (not in the parent's `required`); `nullable` (also accepts `null`; not for
    containers); `cpp_member` (when the JSON name is not a C++ identifier, e.g. `class`); `radiance_option` and
    `radiance_config` (the engine flag and native `Config` destination the field binds to, spec §3).
  - `error_class`: the decoder's code for every fault in this subtree (`provenance_invalid`, `calibration_invalid`,
    `ownership_mismatch`, `plan_mismatch`, `invalid_value`, `unsupported_version`); `mismatch`: a const field's code when
    the value differs (default `invalid_value`).
  - `constraints`: an object's extra rules the typed model does not express (cross-field `if`/`then`), as a JSON array
    in a TOML literal string, emitted verbatim as the object's `allOf`.

C++ names are checked before anything is generated: struct and enum names are unique (`CalibrationKey`,
`CalibrationValue` and `FieldInfo` are reserved); two enum fields share a `cpp_type` only with identical values; every
enumerator (from an enum value or a calibration key) is a C++ identifier and unique within its enum, which has at most
256 values; members are unique within their struct; the namespace is identifiers joined by `::`. Referenced defs fit
their use: a map's `key_ref` is a string def, tuple elements are integer defs, a calibration value refers to an integer
or string def. Counts are not negative. A tuple item takes no scalar `item_*` attribute.
