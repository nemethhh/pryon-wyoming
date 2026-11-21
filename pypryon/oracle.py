"""Run the pinned pryon_test oracle under qemu-user and parse its detections.

Mechanism (validated in Task 2, see .superpowers/sdd/pypryon-plan/task-2-report.md):
the ARM harness ``build/oracle/pryon_test`` (built by ``tools/oracle/build.sh``, byte
pinned) is executed with ``unshare --user --map-root-user --pid --fork`` +
``qemu-arm -L <donut rootfs>`` exactly as echo-dot-assist's ``tools/qrun.sh`` does —
the PID namespace is required because bionic mutexes store a 16-bit owner tid and
deadlock when the host tid exceeds 65535. Audio is fed through stdin in one go with
``-x 1000`` so the harness's real-time pacing sleep is collapsed (pryon drops audio
when its backlog exceeds ~7 s, so the feed must not be slower than the decoder; the
Task 2 smoke run with ``-x 1000`` reproduced the vendor-recorded detections).

CLI: ``python -m pypryon.oracle <raw>`` prints the parsed detections as JSON.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORACLE_BIN = REPO_ROOT / "build" / "oracle" / "pryon_test"

# Rootfs-internal default manifest from tools/oracle/pryon_test.c (DEFAULT_MANIFEST);
# the harness resolves it inside the rootfs because qemu-user's -L prefix maps guest
# absolute paths into the rootfs (observed in the Task 2 smoke run). An explicit
# model_dir is passed as a host path, which qemu uses as-is when the rootfs has no
# such prefixed path.
ROOTFS_DEFAULT_MANIFEST = "/system/local/models/keyword/en-US/ALEXA/pryon.manifest"

# pryon_test consumes raw s16le mono 16 kHz audio (its "format:" line prints
# enc=0 rate=16000 bits=16 ch=1; the pacing sleep divides by 16000).
SAMPLE_RATE_HZ = 16000

# Feed speed: pryon_test sleeps n/16000/speed seconds per chunk, so -x 1000 makes
# the pacing negligible (Task 2's unpaced smoke-run value; brief mandates no throttling).
FEED_SPEED = "1000"

# pryon_test.c keeps at most 8 client properties (props[8]) and -p takes name=value
# with an integer atoll() conversion.
MAX_CLIENT_PROPS = 8

# Exit codes of pryon_test.c: 0 = detections were printed, 3 = no result callbacks.
ORACLE_EXIT_OK = 0
ORACLE_EXIT_NO_DETECTIONS = 3

# stdout RESULT line, printed by our own harness tools/oracle/pryon_test.c (on_result):
# "RESULT decoder=%s keyword=%s type=%d begin=%llu end=%llu (%.2fs..%.2fs) meta=%u"
_RESULT_RE = re.compile(
    r"^RESULT decoder=\S+ keyword=(?P<keyword>\S+) type=(?P<type>\d+) "
    r"begin=(?P<begin>\d+) end=(?P<end>\d+) \(\d+\.\d+s\.\.\d+\.\d+s\) meta=\d+$"
)

# Engine-side KWS recognition log line (stderr), observed in the Task 2 pinned-binary
# smoke run (build/oracle/smoke.stderr lines 503/524) and in echo-dot-assist's
# FINDINGS.md records; field names/separators are the vendor logger's format:
# pryon[<level>] <TAG>: DecoderID=<id>: recognition_id=<id>-<n>: {KWS}
#   segmentation_score=<f>,classifier_score=<f>,active_threshold=<f>,detection_type=<word>
_KWS_RE = re.compile(
    r"^pryon\[\d+\] \w+: DecoderID=\S+: recognition_id=(?P<rid>[^:]+): \{KWS\} "
    r"segmentation_score=[^,]+,classifier_score=(?P<score>[^,]+),"
    r"active_threshold=(?P<threshold>[^,]+),detection_type=(?P<dtype>\S+)$"
)

# RESULT `type` int ↔ KWS detection_type word. Provenance: vendor
# pryon-native/LEDGER.md §"Decoder: score and accept/near-miss rule (2026-11-11)":
# "Accept (RESULT type=2)" and "Near-miss (type=0)"; thresholds accept 0.923 /
# notify 0.69 come from the model's kw.cfg.json. Unknown type codes are kept as the
# KWS word without error (the map is only a consistency check).
_DETECTION_TYPE_BY_CODE = {2: "Accept", 0: "NearMiss"}


class OracleError(RuntimeError):
    """The oracle could not be run, or its output could not be parsed."""


def _eda_dir() -> Path:
    """The echo-dot-assist checkout: $EDA_DIR, else ../echo-dot-assist next to this repo."""
    env = os.environ.get("EDA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT.parent / "echo-dot-assist").resolve()


def parse_detections(output: str) -> list[dict]:
    """Parse concatenated pryon_test stdout+stderr text into detection dicts.

    RESULT lines (stdout) and KWS recognition log lines (stderr) pair up by ordinal:
    the n-th RESULT matches the KWS line whose recognition_id ends in ``-n``
    (dec-0/dec-1 in the Task 2 smoke run). Returns dicts with keys ``keyword,
    detection_type, classifier_score, begin_sample, end_sample, active_threshold``.
    """
    results: list[re.Match[str]] = []
    kws: dict[int, re.Match[str]] = {}
    for line in output.splitlines():
        m = _RESULT_RE.match(line)
        if m:
            results.append(m)
            continue
        m = _KWS_RE.match(line)
        if m:
            decoder, sep, idx = m.group("rid").rpartition("-")
            if not decoder or not sep or not idx.isdigit():
                raise OracleError(f"unparsable recognition_id in KWS log line: {line!r}")
            i = int(idx)
            if i in kws:
                raise OracleError(f"duplicate KWS detection line for recognition index {i}: {line!r}")
            kws[i] = m
    if len(results) != len(kws):
        raise OracleError(
            f"RESULT lines ({len(results)}) do not match KWS detection lines ({len(kws)}): "
            "the oracle output streams are out of sync"
        )
    detections = []
    for i, r in enumerate(results):
        k = kws[i]
        dtype = k.group("dtype")
        expected = _DETECTION_TYPE_BY_CODE.get(int(r.group("type")))
        if expected is not None and expected != dtype:
            raise OracleError(
                f"RESULT type={r.group('type')} (i.e. {expected}) contradicts KWS "
                f"detection_type={dtype} for detection #{i}"
            )
        detections.append(
            {
                "keyword": r.group("keyword"),
                "detection_type": dtype,
                "classifier_score": float(k.group("score")),
                "begin_sample": int(r.group("begin")),
                "end_sample": int(r.group("end")),
                "active_threshold": float(k.group("threshold")),
            }
        )
    return detections


def run_oracle(
    raw_path: Path,
    model_dir: Path | None = None,
    client_props: dict[str, int] | None = None,
    *,
    timeout_s: float | None = None,
) -> list[dict]:
    """Feed a raw s16le 16 kHz mono clip through the pinned oracle, parse detections.

    raw_path: audio clip on the host (fed via stdin, so no guest path translation).
    model_dir: model directory holding pryon.manifest; None uses the harness default
    (the ALEXA model inside the donut rootfs).
    client_props: client properties pushed before the audio as ``-p name=value``.
    timeout_s: subprocess timeout; None scales with clip length (10x audio duration,
    floor 120 s — the feed is unpaced, so run time is qemu decode time) so a long
    clip is never clipped by qrun.sh's 120 s default.
    """
    props = client_props or {}
    for name, value in props.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise OracleError(
                f"client property {name}={value!r} must be an int "
                "(pryon_test -p takes name=value with an integer value)"
            )
    if len(props) > MAX_CLIENT_PROPS:
        raise OracleError(f"at most {MAX_CLIENT_PROPS} client properties (pryon_test.c props[8])")

    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"audio clip not found: {raw_path}")
    if not ORACLE_BIN.is_file():
        raise OracleError(f"oracle binary missing: {ORACLE_BIN} — build it with tools/oracle/build.sh")
    if shutil.which("qemu-arm") is None:
        raise OracleError("qemu-arm not found on PATH (echo-dot-assist keeps it in ~/.local/bin)")
    if shutil.which("unshare") is None:
        raise OracleError("unshare not found on PATH (needed for the PID-namespace qemu wrapper)")
    rootfs = _eda_dir() / "firmware" / "donut" / "rootfs"
    if not rootfs.is_dir():
        raise OracleError(
            f"donut firmware rootfs not found: {rootfs} "
            f"(EDA_DIR={os.environ.get('EDA_DIR') or 'unset'}; "
            "set EDA_DIR to the echo-dot-assist checkout)"
        )

    args = [
        "unshare", "--user", "--map-root-user", "--pid", "--fork", "--kill-child",
        # --kill-child (beyond qrun.sh) so a hard timeout cannot orphan qemu inside
        # its private PID namespace.
        "qemu-arm", "-L", str(rootfs),
        "-E", "LD_LIBRARY_PATH=/system/lib:/system/vendor/lib",
        str(ORACLE_BIN), "-x", FEED_SPEED,
    ]
    if model_dir is not None:
        args += ["-m", str(Path(model_dir) / "pryon.manifest")]
    for name, value in props.items():
        args += ["-p", f"{name}={value}"]

    if timeout_s is None:
        timeout_s = max(120.0, 10.0 * raw_path.stat().st_size / 2 / SAMPLE_RATE_HZ)
    try:
        with raw_path.open("rb") as audio:
            proc = subprocess.run(args, stdin=audio, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise OracleError(f"oracle run timed out after {timeout_s:.0f}s feeding {raw_path}") from exc

    output = (
        proc.stdout.decode("utf-8", "replace")
        + "\n"
        + proc.stderr.decode("utf-8", "replace")
    )
    if proc.returncode not in (ORACLE_EXIT_OK, ORACLE_EXIT_NO_DETECTIONS):
        tail = "\n".join(output.splitlines()[-20:])
        raise OracleError(f"oracle exited {proc.returncode}; last output:\n{tail}")
    return parse_detections(output)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print("usage: python -m pypryon.oracle <raw>\n  <raw>: s16le mono 16 kHz clip", file=sys.stderr)
        return 2
    detections = run_oracle(Path(argv[0]))
    print(json.dumps(detections, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
