import struct, sys, time, subprocess
d = "/sys/kernel/debug/dri/0"
def u32(n):
    try:
        b = open(f"{d}/{n}", "rb", buffering=0).read(4)
        return struct.unpack("<I", b)[0] if len(b) == 4 else None
    except OSError as e:
        return f"err:{e.errno}"
label, secs = sys.argv[1], int(sys.argv[2])
for _ in range(secs):
    p = subprocess.run(["amd-smi", "metric", "-g", "0", "-p", "-u"], capture_output=True, text=True).stdout
    w = next((l.split()[1] for l in p.splitlines() if "SOCKET_POWER" in l), "?")
    a = next((l.split()[1] for l in p.splitlines() if "GFX_ACTIVITY" in l), "?")
    print(label, time.monotonic(), "gfxoff_enabled", u32("amdgpu_gfxoff"), "gfxoff_status", u32("amdgpu_gfxoff_status"), "W", w, "gfx", a, flush=True)
    time.sleep(1)
