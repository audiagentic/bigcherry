#!/usr/bin/env bash
# HI36/HI35 b10502 GPU pipeline on Brutus.
#
# GPU policy: GPU2 (R9700 gfx1201 32GB) ONLY.
#   GPU0/1 (7900 XTX) = parallel agent's dual-XTX MTP server. DO NOT TOUCH.
#   GPU3 (6900 XT)    = kept free.
#
# Tree policy:
#   - The parallel agent's MTP server runs in its own container; its work
#     does NOT conflict with host-side builds. Host build dirs are ~/bc-build-*.
#   - S2 rebuilds ~/bc-build-record-4b against the SHARED tree at b10502
#     (that IS the pending A5 acceptance build for it; audit already passes).
#   - S4 regenerates the workload-max catalog from the 27B inventory IN the
#     shared tree and incrementally rebuilds ~/bc-build-tune-4b and
#     ~/bc-build-replay (already configured workload-max; only the signature
#     file path and generated catalog sources change).
#
# Stages (serial, GPU-idle prechecked):
#   S1 kernel-fraction traces, 27B dense, STALE build 22dc605 (kernel mix of a
#      dense 27B workload is unchanged across the pin window - host-side
#      changes only; labeled as such in the analysis doc)
#   S2 rebuild record-4b build at b10502 (shared tree)
#   S3 record 27B MTP workload -> inventory (27B types/widths)
#   S4 isolated tree: apply + generate workload-max + build tune & replay
#   S5 tune run 1 + run 2 (stability pair, identical policy)
#   S6 export replay cache from run 1
#   S7 replay baseline (same workload) + miss-log (different workload)
set -u
BC=/mnt/vault/development/llmhosts/bigcherry
BIN=$BC/artifacts/h36-pipeline
LOG=$BIN/logs
STATUS=$BIN/status
MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
REC_BUILD=~/bc-build-record-4b
TUNE_BUILD=~/bc-build-tune-4b
REPLAY_BUILD=~/bc-build-replay
mkdir -p "$LOG" "$STATUS"
# single-instance guard: a second launch exits instead of colliding on GPU2/builds
exec 9>"$BIN/.lock"
flock -n 9 || {
	echo "another pipeline instance holds $BIN/.lock; exiting" >&2
	exit 1
}
MASTER=$LOG/master.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }
fail() {
	say "FAILED: $*"
	echo "FAILED: $*" >"$STATUS/current"
	exit 1
}

gpu2_free() {
	# $NF = trailing number on each line (e.g. "GPU[2] : GPU use (%): 0" -> 0)
	local use vram
	use=$(rocm-smi --showuse 2>/dev/null | awk '/^GPU\[2\]/{print $NF; exit}')
	vram=$(rocm-smi --showmemuse 2>/dev/null | awk '/^GPU\[2\]/{print $NF; exit}')
	case "$use" in '' | *[!0-9]*) return 1 ;; esac
	case "$vram" in '' | *[!0-9]*) return 1 ;; esac
	[ "$use" -lt 10 ] && [ "$vram" -lt 10 ]
}
wait_gpu2() {
	local i
	for i in $(seq 1 60); do
		gpu2_free && return 0
		sleep 30
	done
	return 1
}
wait_port() {
	# $1=port  (up to ~8 min; model load dominates)
	local i
	for i in $(seq 1 96); do
		curl -s --max-time 2 "http://127.0.0.1:$1/health" | grep -q ok && return 0
		sleep 5
	done
	return 1
}

### S1: kernel-fraction traces (stale build, native dispatch, dense 27B)
if [ -f "$STATUS/S1_done" ]; then
	say "S1 already done (marker); skipping"
else
	say "S1 start: kernel-fraction traces (27B dense, stale build 22dc605)"
	echo S1_running >"$STATUS/current"
	if ! wait_gpu2; then fail "S1: GPU2 not free after 30 min"; fi
	HIP_VISIBLE_DEVICES=2 rocprofv3 --kernel-trace --output-format csv \
		-d "$BIN/kf-prefill" \
		-- $REC_BUILD/bin/llama-bench -m "$MODEL" -p 2048 -n 0 -r 3 -ngl 99 -fa on \
		>"$LOG/s1-prefill.log" 2>&1 || fail "S1 prefill trace (tail: $(tail -5 "$LOG/s1-prefill.log"))"
	if ! wait_gpu2; then fail "S1: GPU2 busy between traces"; fi
	HIP_VISIBLE_DEVICES=2 rocprofv3 --kernel-trace --output-format csv \
		-d "$BIN/kf-decode" \
		-- $REC_BUILD/bin/llama-bench -m "$MODEL" -p 0 -n 256 -r 3 -ngl 99 -fa on \
		>"$LOG/s1-decode.log" 2>&1 || fail "S1 decode trace (tail: $(tail -5 "$LOG/s1-decode.log"))"
	say "S1 done: prefill CSVs=$(find "$BIN"/kf-prefill -name '*_kernel_trace.csv' 2>/dev/null | wc -l) decode CSVs=$(find "$BIN"/kf-decode -name '*_kernel_trace.csv' 2>/dev/null | wc -l)"
	touch "$STATUS/S1_done"
	echo S1_done >"$STATUS/current"
fi

### S2: rebuild shared-tree record build at b10502
if [ -f "$STATUS/S2_done" ]; then
	say "S2 already done (marker); skipping"
else
	say "S2 start: rebuild $REC_BUILD at b10502 (shared tree; audit already passed today)"
	echo S2_running >"$STATUS/current"
	cmake --build "$REC_BUILD" -j"$(nproc)" >"$LOG/s2-build.log" 2>&1 ||
		fail "S2 record build (tail: $(tail -8 "$LOG/s2-build.log"))"
	"$REC_BUILD/bin/llama-bench" --version 2>/dev/null | tail -1 >>"$LOG/s2-build.log"
	say "S2 done"
	touch "$STATUS/S2_done"
	echo S2_done >"$STATUS/current"
fi

### S3: record the 27B MTP workload -> inventory
# NOTE: record mode writes the record file to the EXACT GGML_HIP_DISPATCH_DB
# path (no .measurements.jsonl suffix -- that convention is tune mode only).
if [ -f "$STATUS/S3_done" ]; then
	say "S3 already done (marker); skipping"
else
	say "S3 start: record 27B MTP workload"
	echo S3_running >"$STATUS/current"
	DB=$BIN/record-27b.jsonl
	if [ ! -s "$DB" ]; then
		HIP_VISIBLE_DEVICES=2 GGML_HIP_DISPATCH_MODE=record GGML_HIP_DISPATCH_DB="$DB" \
			nohup "$REC_BUILD/bin/llama-server" -m "$MODEL" \
			--port 42101 --host 127.0.0.1 -ngl 99 -fa on \
			--ctx-size 8192 --batch-size 512 --ubatch-size 256 \
			--spec-type draft-mtp --spec-draft-n-max 4 \
			--jinja --parallel 1 --no-webui \
			>"$LOG/s3-server.log" 2>&1 &
		SRV=$!
		if ! wait_port 42101; then
			kill "$SRV" 2>/dev/null
			fail "S3 server unhealthy (tail: $(tail -8 "$LOG/s3-server.log"))"
		fi
		(cd /mnt/vault/development/llmhosts/llamacpp && timeout 1500 python3 bench/run_bench.py \
			--bench-type server-bench \
			--server-url http://127.0.0.1:42101 \
			--model Qwen3.8-27B-Q8_0 \
			--bench-configs default \
			--toggles '{"repetitions":3}') >"$LOG/s3-bench.log" 2>&1 ||
			say "S3 bench exit $? (checking record output anyway)"
		curl -s -X POST http://127.0.0.1:42101/shutdown >/dev/null 2>&1
		sleep 5
		kill "$SRV" 2>/dev/null
	else
		say "S3 record file already present ($(wc -l <"$DB") lines); skipping capture"
	fi
	[ -s "$DB" ] || fail "S3 no record output"
	say "S3 recorded $(wc -l <"$DB") lines"
	(cd "$BC/tools" && python3 -m bigcherry inventory record "$DB" \
		--inventory "$BIN/27b-inventory.json" --database "$BIN/27b-inventory.sqlite") \
		>"$LOG/s3-inventory.log" 2>&1 || fail "S3 inventory conversion ($(tail -3 "$LOG/s3-inventory.log"))"
	say "S3 inventory: $(cat "$BIN/27b-inventory.json")"
	touch "$STATUS/S3_done"
	echo S3_done >"$STATUS/current"
fi

### S4: workload-max catalog (shared tree) + incremental tune/replay rebuilds
if [ -f "$STATUS/S4_done" ]; then
	say "S4 already done (marker); skipping"
else
	say "S4 start: generate workload-max catalog from 27B inventory (shared tree)"
	echo S4_running >"$STATUS/current"
	(cd "$BC/tools" && python3 -m bigcherry generate --variant-set workload-max \
		--arch gfx1201 --inventory "$BIN/27b-inventory.json") \
		>"$LOG/s4-generate.log" 2>&1 || fail "S4 generate ($(tail -5 "$LOG/s4-generate.log"))"
	say "S4 generate done: $(grep -m1 -o 'candidate_count[^,}]*' "$BC"/artifacts/0adcc3bb*/hip-autotune-build-descriptor.json 2>/dev/null || tail -2 "$LOG/s4-generate.log")"
	cmake -S "$BC/vendor/llama.cpp" -B "$TUNE_BUILD" \
		-DGGML_HIP_AUTOTUNE_SIGNATURE_FILE="$BIN/27b-inventory.json" \
		>"$LOG/s4-tune-reconf.log" 2>&1 || fail "S4 tune reconfigure ($(tail -8 "$LOG/s4-tune-reconf.log"))"
	cmake --build "$TUNE_BUILD" -j"$(nproc)" \
		>"$LOG/s4-tune-build.log" 2>&1 ||
		fail "S4 tune build ($(tail -8 "$LOG/s4-tune-build.log"))"
	cmake -S "$BC/vendor/llama.cpp" -B "$REPLAY_BUILD" \
		-DGGML_HIP_AUTOTUNE_SIGNATURE_FILE="$BIN/27b-inventory.json" \
		>"$LOG/s4-replay-reconf.log" 2>&1 || fail "S4 replay reconfigure ($(tail -8 "$LOG/s4-replay-reconf.log"))"
	cmake --build "$REPLAY_BUILD" -j"$(nproc)" \
		>"$LOG/s4-replay-build.log" 2>&1 ||
		fail "S4 replay build ($(tail -8 "$LOG/s4-replay-build.log"))"
	say "S4 done"
	touch "$STATUS/S4_done"
	echo S4_done >"$STATUS/current"
fi

### S5: tune runs 1 + 2 (stability pair, identical policy)
tune_run() {
	local tag=$1 db=$2
	# Resume-safe: a complete measurements file from a prior run is kept.
	if [ -s "$db.measurements.jsonl" ]; then
		say "S5 $tag measurements already present ($(wc -l <"$db.measurements.jsonl") lines); skipping"
		return 0
	fi
	rm -f "$db"
	HIP_VISIBLE_DEVICES=2 GGML_HIP_DISPATCH_MODE=tune GGML_HIP_DISPATCH_DB="$db" \
		GGML_HIP_TUNE_SCREEN_SAMPLES=3 GGML_HIP_TUNE_FINAL_SAMPLES=15 \
		GGML_CUDA_DISABLE_GRAPHS=1 \
		nohup "$TUNE_BUILD/bin/llama-server" -m "$MODEL" \
		--port 42102 --host 127.0.0.1 -ngl 99 -fa on \
		--ctx-size 8192 --batch-size 512 --ubatch-size 256 \
		--spec-type draft-mtp --spec-draft-n-max 4 \
		--jinja --parallel 1 --no-webui \
		>"$LOG/s5-$tag-server.log" 2>&1 &
	local srv=$!
	if ! wait_port 42102; then
		kill "$srv" 2>/dev/null
		return 1
	fi
	(cd /mnt/vault/development/llmhosts/llamacpp && timeout 7200 python3 bench/run_bench.py \
		--bench-type server-bench \
		--server-url http://127.0.0.1:42102 \
		--model Qwen3.8-27B-Q8_0 \
		--bench-configs default \
		--toggles '{"repetitions":3}') >"$LOG/s5-$tag-bench.log" 2>&1
	curl -s -X POST http://127.0.0.1:42102/shutdown >/dev/null 2>&1
	kill "$srv" 2>/dev/null
	# The tuner flushes measurements at atexit (atomic temp->rename). Poll for
	# process exit + file appearance instead of a fixed sleep (S5 run-1 race).
	local i
	for i in $(seq 1 120); do
		kill -0 "$srv" 2>/dev/null || break
		sleep 1
	done
	for i in $(seq 1 60); do
		[ -s "$db.measurements.jsonl" ] && break
		sleep 1
	done
	[ -s "$db.measurements.jsonl" ]
}
if [ -f "$STATUS/S5_done" ]; then
	say "S5 already done (marker); skipping"
else
	say "S5 start: tune run 1"
	echo S5_running >"$STATUS/current"
	if ! wait_gpu2; then fail "S5: GPU2 busy before run 1"; fi
	tune_run t1 "$BIN/tune-t1.jsonl" || fail "S5 tune run 1 (server: $(tail -5 "$LOG/s5-t1-server.log") bench: $(tail -5 "$LOG/s5-t1-bench.log"))"
	say "S5 run 1 done: $(wc -l <"$BIN/tune-t1.jsonl.measurements.jsonl") measurement lines"
	if ! wait_gpu2; then fail "S5: GPU2 busy before run 2"; fi
	say "S5 start: tune run 2 (stability pair)"
	tune_run t2 "$BIN/tune-t2.jsonl" || fail "S5 tune run 2"
	say "S5 run 2 done: $(wc -l <"$BIN/tune-t2.jsonl.measurements.jsonl") measurement lines"
	touch "$STATUS/S5_done"
	echo S5_done >"$STATUS/current"
fi

### S6: export replay cache from run 1 (against the ISOLATED build's manifest)
if [ -f "$STATUS/S6_done" ]; then
	say "S6 already done (marker); skipping"
else
	say "S6 start: promote (BH over combined A+B) + export replay cache"
	echo S6_running >"$STATUS/current"
	MANIFEST=$(ls "$BC"/artifacts/0adcc3bb*/hip-autotune-manifest.json 2>/dev/null | head -1)
	[ -n "$MANIFEST" ] || fail "S6 shared-tree manifest not found"
	# Export requires promotion_status=promoted. The J: tree tooling (5020c68)
	# rejects duplicate hypothesis identities, so promote run A alone; run B is
	# the stability comparison, not a promotion input.
	(cd "$BC/tools" && python3 -m bigcherry tune-promote \
		"$BIN/tune-t1.jsonl.measurements.jsonl" \
		--output "$BIN/t1-promoted.measurements.jsonl") \
		>"$LOG/s6-promote.log" 2>&1 || fail "S6 promote ($(tail -8 "$LOG/s6-promote.log"))"
	say "S6 promote: $(tail -1 "$LOG/s6-promote.log")"
	(cd "$BC/tools" && python3 -m bigcherry.replay_cache \
		"$BIN/t1-promoted.measurements.jsonl" \
		--manifest "$MANIFEST" --output "$BIN/dispatch-27b.cache") \
		>"$LOG/s6-export.log" 2>&1 || fail "S6 export ($(tail -8 "$LOG/s6-export.log"))"
	say "S6 done: $(ls -la "$BIN/dispatch-27b.cache" | awk '{print $5, $9}')"
	touch "$STATUS/S6_done"
	echo S6_done >"$STATUS/current"
fi

### S7: replay baseline + miss-log
replay_run() {
	local tag=$1 port=$2 cfg=$3 toggles=$4 extra=$5 ctx=${6:-8192} kvt=${7:--ctk f16 -ctv f16}
	rm -f "$BIN/cov-$tag.json" "$BIN/miss-$tag.jsonl"
	# shellcheck disable=SC2086
	env $extra HIP_VISIBLE_DEVICES=2 GGML_HIP_DISPATCH_MODE=replay \
		GGML_HIP_DISPATCH_CACHE="$BIN/dispatch-27b.cache" \
		GGML_HIP_DISPATCH_COVERAGE="$BIN/cov-$tag.json" \
		nohup "$REPLAY_BUILD/bin/llama-server" -m "$MODEL" \
		--port "$port" --host 127.0.0.1 -ngl 99 -fa on \
		--ctx-size "$ctx" $kvt --batch-size 512 --ubatch-size 256 \
		--spec-type draft-mtp --spec-draft-n-max 4 \
		--jinja --parallel 1 --no-webui \
		>"$LOG/s7-$tag-server.log" 2>&1 &
	local srv=$!
	if ! wait_port "$port"; then
		kill "$srv" 2>/dev/null
		return 1
	fi
	(cd /mnt/vault/development/llmhosts/llamacpp && timeout 1800 python3 bench/run_bench.py \
		--bench-type server-bench \
		--server-url "http://127.0.0.1:$port" \
		--model Qwen3.8-27B-Q8_0 \
		--bench-configs "$cfg" \
		--toggles "$toggles") >"$LOG/s7-$tag-bench.log" 2>&1
	curl -s -X POST "http://127.0.0.1:$port/shutdown" >/dev/null 2>&1
	kill "$srv" 2>/dev/null
	local i
	for i in $(seq 1 120); do
		kill -0 "$srv" 2>/dev/null || break
		sleep 1
	done
	for i in $(seq 1 60); do
		[ -s "$BIN/cov-$tag.json" ] && break
		sleep 1
	done
	[ -s "$BIN/cov-$tag.json" ]
}
if [ -f "$STATUS/S7_done" ]; then
	say "S7 already done (marker); skipping"
else
	say "S7 start: replay baseline (same workload)"
	echo S7_running >"$STATUS/current"
	if ! wait_gpu2; then fail "S7: GPU2 busy before baseline"; fi
	if [ -s "$BIN/cov-baseline.json" ]; then
		say "S7 baseline cov already present; skipping"
	else
		replay_run baseline 42103 default '{"repetitions":3}' "" || fail "S7 baseline replay (server: $(tail -5 "$LOG/s7-baseline-server.log") bench: $(tail -5 "$LOG/s7-baseline-bench.log"))"
	fi
	say "S7 baseline: $(cat "$BIN/cov-baseline.json")"
	if ! wait_gpu2; then fail "S7: GPU2 busy before miss run"; fi
	say "S7 start: replay miss-log (different workload: long-prompt-12k)"
	if [ -s "$BIN/miss-misslog.jsonl" ] && [ -s "$BIN/cov-misslog.json" ]; then
		say "S7 misslog artifacts already present; skipping"
	else
		rm -f "$BIN/miss-misslog.jsonl"
		# 12288-token prompt needs ctx > 8192; q8 KV keeps 16k ctx inside 32GB.
		replay_run misslog 42104 long-prompt-12k '{"repetitions":1}' \
			"GGML_HIP_DISPATCH_MISS=native-record GGML_HIP_DISPATCH_MISS_LOG=$BIN/miss-misslog.jsonl" \
			16384 "-ctk q8_0 -ctv q8_0" ||
			fail "S7 miss-log replay (server: $(tail -5 "$LOG/s7-misslog-server.log") bench: $(tail -5 "$LOG/s7-misslog-bench.log"))"
	fi
	say "S7 misslog: $(cat "$BIN/cov-misslog.json")"
	if [ -s "$BIN/miss-misslog.jsonl" ]; then
		say "S7 miss log: $(wc -l <"$BIN/miss-misslog.jsonl") entries"
	fi
	touch "$STATUS/S7_done"
fi
say "ALL STAGES DONE"
echo ALL_DONE >"$STATUS/current"
