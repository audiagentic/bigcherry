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
- **semantics**: the same schema plus the declared contract data the schema does not express: the contract identity
  (`api_version`, `kind`, `contract_version`, which the loader also requires to equal the `/apiVersion`, `/kind` and
  `/metadata/contract_version` consts), every field's provenance class, the lifecycle, the Radiance bindings
  (`radiance_option`, `radiance_config`), the reader's error codes (`error_class`, `mismatch`) and the calibration key
  order. It identifies the declared v1 contract. C++ names and docs are excluded (renaming generated code is not a wire
  change, and a consumer would not compile).

v1 is closed (spec §8): a change to either fingerprint is a `bigcherry/v2` change. Correcting documentation changes
neither and is allowed. The fingerprints identify declared contract data, not behaviour: the frozen procedures (JCS
digest, provenance coverage and inheritance, semantic checks, native binding behaviour; spec §3, §7, §8) are held by
the reader's conformance tests and reviewed implementation, not by these hashes.

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
  covered. `envelope` is a registry classification, not a provenance record class: it marks the integrity fields
  (`apiVersion`, `kind`, `metadata`, `provenance`), which identify the document rather than configure it. It decides
  nothing about provenance coverage, which follows spec §7 (only `metadata.content_digest` and the provenance records
  themselves are exempt); how a record for an identity field such as `metadata.producer.build_sha256` is classed is
  settled with the coverage check (RSC08 slice 2 follow-up). The other classes are exactly the record's `class` values.
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
    `integer` (`minimum`, `maximum`, `exclusive_maximum`, `const`), `string` (`min_length`, `max_length`, `pattern`,
    `format`, `const`; a `pattern` requires a `max_length`, spec §11 amendment 7), `boolean`, `enum` (string `values`, `cpp_type`), `int_enum` (integer `values`), `ref` (`ref` = a def),
    `calibration_key` and `calibration_value` (the two members of `/spec/calibration/[]`).
  - `doc` (required, nonempty); `optional` (not in the parent's `required`); `nullable` (also accepts `null`; not for
    containers); `cpp_member` (when the JSON name is not a C++ identifier, e.g. `class`); `radiance_option` and
    `radiance_config` (the engine flag and native `Config` destination the field binds to, spec §3).
  - `default` and `default_source` (together or neither; scalar fields, or a ref to a scalar def): the schema default
    (spec §7, §11 amendment 9) and where it comes from, `radiance:Config.<member>` (must equal the field's own
    `radiance_config`) or `setting:R3_<NAME>`. `bc_contract_gen RADIANCE_SRC COMPONENT_TOML...` and `bc_contract_check`
    compare the value with the `Config` initialiser in `core/rad_core.h` or the setting's `default` and fail on any
    difference; an initialiser that is not a simple literal is no source, so such a field has no default. The default
    is a schema annotation (the validation fingerprint ignores it) and part of the semantic fingerprint.
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

## Inactive v2 scaffold (RSC11 S9)

`serving-v2.toml` is the **new**, single-source v2 registry. The production `bigcherry/v1` generator output, schema fingerprints, reader and writer remain unchanged until the one S13 activation. The version-specific generated C++ types/codec, Python codec, schema, field reference and fingerprints live under `../generated/v2/`; the documented copies are in `docs/design/serving-manifest-v2/`. V2 code has its own error types; it does not translate or accept v1 manifests.

Regenerate by invoking `bc_contract_gen` with `contract/serving-v2.toml` and `generated/v2`, then copy `serving-v2.schema.json` and `serving-fields.md` to their documented v2 equivalents. `bc_contract_check` compares both v1 and v2 committed outputs against the registry, including the copies in docs; v2 generation never mutates a v1 output. `bc_v2_codec_test` checks the compiled C++ codec; `tools/tests/core/test_serving_manifest_v2_contract.py` checks the generated Python codec, synthetic fixtures, closed schema, duplicate rejection, role/BDF semantics and version isolation.

V2-only `[[field]]` attributes `origins` and `confidences` declare permitted provenance source and evidence quality; omitted attributes mean unrestricted. Values are validated against the registry's v2 provenance enums, emitted in the C++ field table/Python constraints and included in the v2 semantic fingerprint. RSC03 resource budgets and RSC09 handoff fields must be derived or calibrated, never profile estimates. See `docs/design/serving-manifest-v2/spec.md` for all v2 additions and pending S8/S10/S13 obligations.

## Inactive v2 successor (RSC11 S9)

`serving-v2.toml` is the single v2 registry. `bc_contract_gen` produces committed `generated/v2/{serving-v2.schema.json,serving_types.h,serving_codec.h,serving_codec.cpp,serving_codec.py,serving-fields.md,contract.fingerprint}`; `bc_contract_check` byte-compares all generated artifacts and the v2 documentation mirrors. Regenerate with:

```
<serving build>/bc_contract_gen engines/radiance/components/serving/contract/serving-v2.toml engines/radiance/components/serving/generated/v2
```

Then copy the generated schema and field reference to `docs/design/serving-manifest-v2/` (schema destination: `serving-manifest.v2.schema.json`). The S8/S10 type interface and S5/S11 separate admission-refusal wire are documented in `docs/design/serving-manifest-v2/spec.md`. These artifacts are inactive until S13; v1 generation, its fingerprints, and production readers remain unchanged.
