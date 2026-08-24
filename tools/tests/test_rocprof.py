from pathlib import Path

from bigcherry import rocprof


def test_wrap_command_prefixes_rocprofv3(tmp_path):
    wrapped = rocprof.wrap_command(
        ["./llama-server", "-m", "model.gguf"],
        output_directory=tmp_path,
    )
    assert wrapped[:2] == ["rocprofv3", "--kernel-trace"]
    assert wrapped[-4:] == ["--", "./llama-server", "-m", "model.gguf"]
    assert str(tmp_path) in wrapped


def test_find_kernel_trace_locates_nested_file(tmp_path):
    nested = tmp_path / "brutus"
    nested.mkdir()
    target = nested / "12345_kernel_trace.csv"
    target.write_text("Kind\n", encoding="utf-8")
    assert rocprof.find_kernel_trace(tmp_path) == target


def test_find_kernel_trace_raises_when_missing(tmp_path):
    try:
        rocprof.find_kernel_trace(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "kernel_trace.csv" in str(exc)


def test_find_kernel_trace_raises_when_ambiguous(tmp_path):
    for pid in ("1", "2"):
        f = tmp_path / f"{pid}_kernel_trace.csv"
        f.write_text("Kind\n", encoding="utf-8")
    try:
        rocprof.find_kernel_trace(tmp_path)
        assert False, "expected ValueError for ambiguous match"
    except ValueError as exc:
        assert "exactly one" in str(exc)


def _write_trace(path: Path, rows: list[dict[str, str]]) -> None:
    import csv
    fieldnames = ["Kind", "Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_kernel_trace_skips_non_dispatch_rows(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path, [
        {"Kind": "MEMORY_COPY", "Agent_Id": "Agent 1", "Kernel_Name": "copy",
         "Start_Timestamp": "0", "End_Timestamp": "10"},
        {"Kind": "KERNEL_DISPATCH", "Agent_Id": "Agent 1", "Kernel_Name": "mul_mat_vec_q",
         "Start_Timestamp": "0", "End_Timestamp": "100"},
    ])
    dispatches = rocprof.load_kernel_trace(path)
    assert len(dispatches) == 1
    assert dispatches[0].kernel_name == "mul_mat_vec_q"
    assert dispatches[0].duration_ns == 100


def test_classify_kernel_real_names():
    # Real kernel names observed in a live trace on Brutus, 2026-08-25.
    assert rocprof.classify_kernel("void mul_mat_vec_q<(ggml_type)14") == "mmvq"
    assert rocprof.classify_kernel("void mul_mat_q<(ggml_type)8") == "mmq"
    assert rocprof.classify_kernel("quantize_q8_1(float const*") == "quantize"
    assert rocprof.classify_kernel("void rms_norm_f32<1024") == "norm"
    assert rocprof.classify_kernel("void flash_attn_tile<256") == "flash_attn"
    assert rocprof.classify_kernel("ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)") == "rccl"
    assert rocprof.classify_kernel("__amd_rocclr_fillBufferAligned") == "copy"
    assert rocprof.classify_kernel("totally_unknown_kernel_xyz") == "other"


def test_union_busy_ns_merges_overlapping_intervals():
    # Two overlapping intervals on the same agent must count once, not twice.
    busy = rocprof._union_busy_ns([(0, 100), (50, 150), (200, 250)])
    assert busy == 200  # [0,150) union [200,250) = 150 + 50


def test_summarize_family_and_agent_aggregates(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path, [
        {"Kind": "KERNEL_DISPATCH", "Agent_Id": "Agent 1", "Kernel_Name": "mul_mat_vec_q<(ggml_type)14",
         "Start_Timestamp": "0", "End_Timestamp": "100"},
        {"Kind": "KERNEL_DISPATCH", "Agent_Id": "Agent 1", "Kernel_Name": "mul_mat_vec_q<(ggml_type)8",
         "Start_Timestamp": "50", "End_Timestamp": "200"},
        {"Kind": "KERNEL_DISPATCH", "Agent_Id": "Agent 2", "Kernel_Name": "quantize_q8_1",
         "Start_Timestamp": "0", "End_Timestamp": "30"},
    ])
    dispatches = rocprof.load_kernel_trace(path)
    families, agents = rocprof.summarize(dispatches)

    assert families["mmvq"].call_count == 2
    assert families["mmvq"].total_time_ns == 100 + 150
    assert families["quantize"].call_count == 1

    assert agents["Agent 1"].busy_ns == 200  # union of [0,100) and [50,200)
    assert agents["Agent 1"].span_ns == 200
    assert agents["Agent 2"].busy_ns == 30


def test_format_summary_is_readable(tmp_path):
    path = tmp_path / "trace.csv"
    _write_trace(path, [
        {"Kind": "KERNEL_DISPATCH", "Agent_Id": "Agent 1", "Kernel_Name": "mul_mat_vec_q<(ggml_type)14",
         "Start_Timestamp": "0", "End_Timestamp": "100"},
    ])
    families, agents = rocprof.summarize(rocprof.load_kernel_trace(path))
    text = rocprof.format_summary(families, agents)
    assert "mmvq" in text
    assert "Agent 1" in text
