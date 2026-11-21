"""Run the pinned pryon_test oracle (real libpryon under qemu-arm) and parse its detections.

The ARM harness ``build/oracle/pryon_test`` (built by ``tools/oracle/build.sh``) runs under
``unshare --user --map-root-user --pid --fork`` + ``qemu-arm -L <rootfs>``; the PID namespace
is needed because bionic mutexes store a 16-bit owner tid. The rootfs (an unpacked Echo
system image carrying libpryon.so) comes from ``PYPRYON_ORACLE_ROOTFS``.

CLI: ``python tools/oracle/pryon_oracle.py <clip.raw>`` prints the detections as JSON.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_BIN = REPO_ROOT / "build" / "oracle" / "pryon_test"

ROOTFS_DEFAULT_MANIFEST = "/system/local/models/keyword/en-US/ALEXA/pryon.manifest"
SAMPLE_RATE_HZ = 16000
# pryon_test sleeps n/16000/speed per chunk; libpryon drops audio past a ~7 s backlog
FEED_SPEED = "1000"
MAX_CLIENT_PROPS = 8  # pryon_test.c props[8]
ORACLE_EXIT_OK = 0
ORACLE_EXIT_NO_DETECTIONS = 3

# tools/oracle/pryon_test.c on_result()
_RESULT_RE = re.compile(
    r"^RESULT decoder=\S+ keyword=(?P<keyword>\S+) type=(?P<type>\d+) "
    r"begin=(?P<begin>\d+) end=(?P<end>\d+) \(\d+\.\d+s\.\.\d+\.\d+s\) meta=\d+$"
)

# libpryon's own KWS log line on stderr
_KWS_RE = re.compile(
    r"^pryon\[\d+\] \w+: DecoderID=\S+: recognition_id=(?P<rid>[^:]+): \{KWS\} "
    r"segmentation_score=[^,]+,classifier_score=(?P<score>[^,]+),"
    r"active_threshold=(?P<threshold>[^,]+),detection_type=(?P<dtype>\S+)$"
)

# vendor LEDGER §Decoder: RESULT type 2 = Accept, 0 = Near-miss
_DETECTION_TYPE_BY_CODE = {2: "Accept", 0: "NearMiss"}


class OracleError(RuntimeError):
    """The oracle could not be run, or its output could not be parsed."""


def _rootfs() -> Path:
    env = os.environ.get("PYPRYON_ORACLE_ROOTFS")
    if not env:
        raise OracleError("set PYPRYON_ORACLE_ROOTFS to an unpacked Echo rootfs carrying libpryon.so")
    return Path(env).expanduser().resolve()


def parse_detections(output: str) -> list[dict]:
    """Parse concatenated pryon_test stdout+stderr text into detection dicts.

    RESULT lines (stdout) and KWS recognition log lines (stderr) pair up by ordinal:
    the n-th RESULT matches the KWS line whose recognition_id ends in ``-n``
    Returns dicts with keys ``keyword,
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
    model_dir: model directory holding pryon.manifest; None uses the ALEXA model
    inside the rootfs.
    client_props: client properties pushed before the audio as ``-p name=value``.
    timeout_s: subprocess timeout; None scales with clip length (10x audio duration,
    floor 120 s — the feed is unpaced, so run time is qemu decode time).
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
        raise OracleError("qemu-arm not found on PATH")
    if shutil.which("unshare") is None:
        raise OracleError("unshare not found on PATH (needed for the PID-namespace qemu wrapper)")
    rootfs = _rootfs()
    if not rootfs.is_dir():
        raise OracleError(f"oracle rootfs not found: {rootfs} (PYPRYON_ORACLE_ROOTFS)")

    args = [
        "unshare", "--user", "--map-root-user", "--pid", "--fork", "--kill-child",
        # a hard timeout must not orphan qemu inside its private PID namespace
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
        print("usage: tools/oracle/pryon_oracle.py <raw>\n  <raw>: s16le mono 16 kHz clip", file=sys.stderr)
        return 2
    detections = run_oracle(Path(argv[0]))
    print(json.dumps(detections, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
