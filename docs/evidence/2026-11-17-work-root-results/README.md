# Result files of finished runs and work-root experiments (2026-11-17)

Small result files (scored results, summaries, run manifests, logs named by an acceptance criterion, benchmark and
probe sources) migrated from the work root so the originals could be deleted: nothing is left unmanaged on the
filesystem. The recorded reference logits, captures, model-sized and binary files, rebuildable build output and the
logs of finished runs that no plan item cites were not kept. Relative paths below the folders match the original
location under the work root (`runs/`, `flashnext/`, `tmp/` and top-level experiment folders).
Absolute paths inside the files were replaced by placeholders: `<work-root>`, `<models-root>`, `<dev-root>`,
`<data-root>`, `<repo>`, `<home>`, `<mnt>`; nothing else was edited.

| Area | Entries kept | Files |
|---|---:|---:|
| flashnext | 162 | 162 |
| root | 34 | 268 |
| runs | 126 | 705 |
| tmp | 3 | 3 |

Selection rule: a run or file is kept when it holds a scored result or source that the repository does not already
carry, or when a plan item's acceptance names the file (for example the engine logs named by RHP01 and RHP19).
