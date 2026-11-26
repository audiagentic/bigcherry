#!/usr/bin/env python3
"""Image-input probe for a running radiance server: does a request with a picture get a right answer, or a clear refusal?

The picture is drawn here (stdlib only): a red disc on the left and a green square on the right on a blue field, 448 x 448.
It is sent as a `data:` URL in an OpenAI chat message with two questions; the answer must name both colours. A server built
without the media decoder, or started text-only (`vision.mm_max_patches = 0`), refuses the request: that is reported, with the
server's own message, and is the expected result of those modes (`--expect refused`).

  python3 tools/lab/radiance/image_probe.py --url http://127.0.0.1:18450 --run-dir RUN --expect answered
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SIZE = 448


def pixel(x: int, y: int, width: int, height: int) -> tuple[int, int, int]:
    cx, cy, radius = width // 4, height // 2, width // 6
    if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
        return (220, 30, 30)
    if 3 * width // 4 - radius <= x <= 3 * width // 4 + radius and cy - radius <= y <= cy + radius:
        return (30, 190, 40)
    return (20, 40, 150)


def bmp(width: int = SIZE, height: int = SIZE) -> bytes:
    """The same picture as a 24-bit BMP (bottom-up BGR rows padded to 4 bytes): stdlib only, and a decoder every FFmpeg build has."""
    pad = (-3 * width) % 4
    rows = b"".join(b"".join(bytes(pixel(x, y, width, height)[::-1]) for x in range(width)) + b"\0" * pad
                    for y in range(height - 1, -1, -1))
    header = b"BM" + struct.pack("<IHHI", 54 + len(rows), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    return header + info + rows


def jpeg(width: int = SIZE, height: int = SIZE) -> bytes:
    """The same picture as a JPEG (needs Pillow, which the stdlib lacks)."""
    import io

    from PIL import Image
    image = Image.new("RGB", (width, height))
    image.putdata([pixel(x, y, width, height) for y in range(height) for x in range(width)])
    out = io.BytesIO()
    image.save(out, "JPEG", quality=92)
    return out.getvalue()


def png(width: int = SIZE, height: int = SIZE) -> bytes:
    """The same picture as an RGB PNG. A build whose FFmpeg lacks the png decoder (the agent container's /opt/ffmpeg6 does)
    refuses it with "the media holds no video stream"."""
    rows = []
    for y in range(height):
        row = bytearray([0])  # filter type 0
        for x in range(width):
            row += bytes(pixel(x, y, width, height))
        rows.append(bytes(row))

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = zlib.crc32(kind + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def ask(url: str, model: str, timeout: float, kind: str = "bmp",
        image_bytes: bytes | None = None) -> tuple[int, dict, float]:
    raw = image_bytes if image_bytes is not None else {"bmp": bmp, "jpeg": jpeg, "png": png}[kind]()
    data_url = f"data:image/{kind};base64," + base64.b64encode(raw).decode()
    body = {"model": model, "max_tokens": 64, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "What colour is the round shape and what colour is the square? Answer in one short sentence."}]}]}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read()), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", "replace")
        try:
            payload = json.loads(text)
        except ValueError:
            payload = {"error": {"message": text[:300]}}
        return error.code, payload, time.perf_counter() - started


def judge(status: int, payload: dict) -> tuple[str, str]:
    """("answered" | "wrong" | "refused", the text that decided it)."""
    if status != 200:
        message = (payload.get("error") or {}).get("message") or json.dumps(payload)[:200]
        return "refused", f"HTTP {status}: {message}"
    answer = (payload["choices"][0]["message"].get("content") or "").strip()
    lowered = answer.lower()
    right = "red" in lowered and "green" in lowered
    return ("answered" if right else "wrong"), answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="native bc-serve run directory for immutable measurement identity")
    parser.add_argument("--work-root", type=Path,
                        help="BC_WORK_ROOT MRC01 record store (defaults to environment)")
    parser.add_argument("--model", default=None, help="the served model name (default: the first id of /v1/models)")
    parser.add_argument("--expect", choices=("answered", "refused"), default="answered")
    parser.add_argument("--format", dest="kind", choices=("bmp", "jpeg", "png"), default="bmp")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", help="write the result as JSON here")
    args = parser.parse_args(argv)
    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from bigcherry.bench.radiance_lab_records import verify_live_target
    from bigcherry.bench.served_model import served_model

    verify_live_target(args.run_dir, args.url)
    args.model = args.model or served_model(args.url)
    actual_image_bytes = {"bmp": bmp, "jpeg": jpeg, "png": png}[args.kind]()
    status, payload, seconds = ask(args.url, args.model, args.timeout, args.kind,
                                   image_bytes=actual_image_bytes)
    verdict, text = judge(status, payload)
    result = {"format": args.kind, "expected": args.expect, "verdict": verdict, "http_status": status, "seconds": round(seconds, 2), "text": text}
    from bigcherry.bench.radiance_lab_records import publish_probe

    digest = publish_probe(
        args.run_dir, kind="image", result=result,
        definition={"format": args.kind, "expected": args.expect,
                    "image_width": SIZE, "image_height": SIZE},
        argv=[str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        extra_inputs={"image_payload": actual_image_bytes},
        work_root=args.work_root,
    )
    print(json.dumps({"result": result, "record_sha256": digest}))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"record_sha256": digest}, handle)
    return 0 if verdict == args.expect else 1


if __name__ == "__main__":
    sys.exit(main())
