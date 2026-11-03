# Documentation

BigCherry's documentation is organised by the part of the project it
describes.

| Location | Contents |
| --- | --- |
| [Reference](reference/README.md) | Engine architecture, dispatch, components, patch contracts, testing and build detail |
| [Radiance](reference/engines/radiance/README.md) | Primary engine: heterogeneous AMD kernels, remote MoE and model results |
| [llama.cpp](reference/engines/llamacpp/README.md) | Active secondary engine: measured HIP dispatch and patch catalogue |
| [Planning](planning/) | Active technical designs, decisions and implementation status |
| [Evidence](evidence/) | Retained measurements and qualification results |
| [Releases](releases/) | Release notes and recorded changes |
| [Archive](archive/) | Superseded designs and historical findings |

The technical detail for each patch or component lives with its source,
under `engines/llamacpp/patches/`, `engines/radiance/patches/` or
`engines/radiance/components/`. The
[patch and component documentation format](reference/DOC_STRUCTURE.md)
defines the section headings used by the generated catalogues.

The [dated Radiance lab history](archive/radiance/LAB_HISTORY.md) and
[old port/design notes](archive/radiance/) are archived; the current
[model pages](reference/engines/radiance/README.md#best-measured-models)
show results for the pinned engine.

Transient traces and large run outputs are stored separately under
`artifacts/`; they are not part of the maintained reference.
