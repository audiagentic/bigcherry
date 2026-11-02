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
| `contract.fingerprint` | `sha256:` of the RFC 8785 JCS bytes of the schema with every `description` removed. |

The fingerprint follows what validates, not the documentation: editing a `doc` changes the schema's descriptions and the
reference but not the fingerprint; any change to a type, bound, requiredness or enum changes it. v1 is closed (spec §8):
such a change is a `bigcherry/v2` change.

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
  matching path prefix gives a field its provenance class. Every value in the document must be covered.
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

C++ type names (structs, enums) must be unique; `CalibrationKey`, `CalibrationValue` and `FieldInfo` are reserved. Two
enum fields may share a `cpp_type` only with identical values.
