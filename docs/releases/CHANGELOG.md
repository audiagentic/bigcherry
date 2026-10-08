# Changelog

## [11474.2.0](https://github.com/audiagentic/bigcherry/compare/bc-11474.1.0...bc-11474.2.0) (2026-10-08)


### New and changed

* add slice worktree lifecycle commands ([#34](https://github.com/audiagentic/bigcherry/issues/34)) ([bbce6b6](https://github.com/audiagentic/bigcherry/commit/bbce6b6eb9dcf9559051e9c02074a0451a6279e5))
* automate qualified patch promotion ([#11](https://github.com/audiagentic/bigcherry/issues/11)) ([bd80a8d](https://github.com/audiagentic/bigcherry/commit/bd80a8d824697e963e58617d8ceb43b0ab3af183))
* execute slice lab runs in detached queued worktrees ([#35](https://github.com/audiagentic/bigcherry/issues/35)) ([a03fd83](https://github.com/audiagentic/bigcherry/commit/a03fd83d232f49ffc54268632e1aa54ec4f0aab0))
* **lab:** reference lane run of llama-server on the R9700 alone with a 4-bit 27B ([#75](https://github.com/audiagentic/bigcherry/issues/75)) ([ef52141](https://github.com/audiagentic/bigcherry/commit/ef5214128379aecf38a2182530ea8b50f15f7892))
* **patch:** add Q8_0 few-tile MMQ Stream-K qualification ([#22](https://github.com/audiagentic/bigcherry/issues/22)) ([4b1dfc4](https://github.com/audiagentic/bigcherry/commit/4b1dfc4854d168b3d250b7435f3d2bb9c272f52c))
* share primary checkout state across worktrees ([#33](https://github.com/audiagentic/bigcherry/issues/33)) ([cfc8d9b](https://github.com/audiagentic/bigcherry/commit/cfc8d9b1167c8aacd961d16261495d81ed440f2d))


### Fixes

* **ci:** scope the experiment audit to changed experiments ([#71](https://github.com/audiagentic/bigcherry/issues/71)) ([e76f349](https://github.com/audiagentic/bigcherry/commit/e76f349cb29521710d6ee473f18c4fc5bf3f4f9e))
* enforce LF line endings for tracked text ([#32](https://github.com/audiagentic/bigcherry/issues/32)) ([ff6bc5a](https://github.com/audiagentic/bigcherry/commit/ff6bc5ad3c5c4f6671882df5c6c0de6b0af4b140))
* **lab:** queue finds built binaries where the build publishes them ([#72](https://github.com/audiagentic/bigcherry/issues/72)) ([3dc67c6](https://github.com/audiagentic/bigcherry/commit/3dc67c64275cc4e621da43c8581881f609f0abec))
* make slice finish resumable and carry primary changes ([#58](https://github.com/audiagentic/bigcherry/issues/58)) ([ab8feab](https://github.com/audiagentic/bigcherry/commit/ab8feabade9a50f7046b93647bebaf61a8813c19))
* **patch:** 1328 aux expert backend works at b11474 (split-state cache, mirrored merge) ([#61](https://github.com/audiagentic/bigcherry/issues/61)) ([64c1ca7](https://github.com/audiagentic/bigcherry/commit/64c1ca71f3238171a3b76f9e1b011153efe9ef42))
* **patch:** 1340 arena plans are keyed by the output set, OUTPUT flags reach the per-device tensors ([#63](https://github.com/audiagentic/bigcherry/issues/63)) ([64c15de](https://github.com/audiagentic/bigcherry/commit/64c15dec117d680e8e5555c31b3c6bdf042b6923))
* **patch:** reconcile remaining experiment audit after BPB01 merge ([#59](https://github.com/audiagentic/bigcherry/issues/59)) ([9b8ec87](https://github.com/audiagentic/bigcherry/commit/9b8ec879fb4c4be030e82e2af0dd7a2fbce6c1a8))
* **patch:** reject adaptive MTP depth (1255, 1268) at b11474 ([#51](https://github.com/audiagentic/bigcherry/issues/51)) ([d5fb657](https://github.com/audiagentic/bigcherry/commit/d5fb657a5075581df0e65d42c967d39a8ffbd63a))
* slice commands read git output as UTF-8; --carry fails instead of carrying nothing ([#74](https://github.com/audiagentic/bigcherry/issues/74)) ([fca841a](https://github.com/audiagentic/bigcherry/commit/fca841a2b3af317a54eb1b236fa2d94f2da47208))
* **test:** mechanics tests read the pinned vendor source, not deleted lab fixtures ([#69](https://github.com/audiagentic/bigcherry/issues/69)) ([52c3eb5](https://github.com/audiagentic/bigcherry/commit/52c3eb5443b08239602b8f40471a12ee2c72c2d7))

## [11474.1.0](https://github.com/audiagentic/bigcherry/compare/bc-11474.0.0...bc-11474.1.0) (2026-10-08)


### New and changed

* **patch:** defer MTP prompt catch-up ([#13](https://github.com/audiagentic/bigcherry/issues/13)) ([b43235a](https://github.com/audiagentic/bigcherry/commit/b43235a62822dff785db221962806533d095bd7b))
* **patch:** promote MTP look-ahead (1321, 1322), deferred catch-up in the production set (1348), diagnostics 1316/1329/1331; reject 1210/1252/1337/1338, retire 1330 ([#27](https://github.com/audiagentic/bigcherry/issues/27)) ([e240483](https://github.com/audiagentic/bigcherry/commit/e240483439312202926d3064597d25cbd4cb374e))
* **patch:** promote MTP tracing diagnostics (1315, 1317, 1318) ([#23](https://github.com/audiagentic/bigcherry/issues/23)) ([6cf9f5c](https://github.com/audiagentic/bigcherry/commit/6cf9f5ce0d173b64ae74b7ebda1424fd6b3b3d44))


### Fixes

* **patch:** make 1316 node hash safe on Meta buffers ([#24](https://github.com/audiagentic/bigcherry/issues/24)) ([1e1c0da](https://github.com/audiagentic/bigcherry/commit/1e1c0da93ceab530c2d67c2763c65f7dc15e4d0f))

## [11474.0.0](https://github.com/audiagentic/bigcherry/compare/bc-11402.0.0...bc-11474.0.0) (2026-10-07)


### Maintenance

* release notes bc-11474.0.0 ([45ab24c](https://github.com/audiagentic/bigcherry/commit/45ab24c808fd79f7119d9c437009621dae88629d))

## [11402.0.0](https://github.com/audiagentic/bigcherry/compare/bc-11126.0.0...bc-11402.0.0) (2026-10-05)


### Performance

* **patching:** avoid duplicate overlay target reads ([cdac46e](https://github.com/audiagentic/bigcherry/commit/cdac46e9e300728fea4dc385bf743363879e6222))


### Maintenance

* release notes bc-b11402 ([1e0b1da](https://github.com/audiagentic/bigcherry/commit/1e0b1da3f996a89456fdf61c14c4f9b43e9f15ef))
