# pypryon

Run Amazon's Pryon wake-word models — the keyword dirs shipped in Echo
firmware and DAVS downloads — on any machine with Python and stock
`onnxruntime`. No ARM library, no Amazon code: the shipped ONNX's custom
`Hybrid*` kernels are rewritten into standard ops implementing the measured
CPUNK quantization contract, and the whole pipeline (LFBE front end, 25-lane
LSTM ring, kw.cfg orchestra decoder) is reimplemented in numpy.

Detections are validated against the **real** `libpryon.so` under
qemu-arm (the pinned `pryon_test` oracle): identical events on the test
clip set, scores within 1e-3 for events clear of the thresholds
(knife-edge events within 1e-2 of a threshold are logged as `boundary`,
per the design's margin rule).

## Usage

```python
from pypryon import Spotter

sp = Spotter("…/keyword/en-US/ALEXA/")   # an extracted model directory
for chunk in mic_chunks_16k_s16:         # int16 mono 16 kHz, any size
    for det in sp.feed(chunk):
        print(det.keyword, det.type_name, det.score,
              det.begin_sample, det.end_sample)
sp.scores_tail()                          # {"ALEXA": 0.93, "STOP": 0.01}
sp.set_client_property("AlarmState", 1)   # kw.cfg threshold overrides
```

Input contract: **int16, mono, 16 kHz**, 1-D arrays. A sample rate cannot
be detected from raw PCM — anything else silently shifts every index and
score.

Beside [openWakeWord](https://github.com/dscripka/openWakeWord):

```python
from pypryon.oww import PryonModel

m = PryonModel("…/ALEXA/")                # predict(x) → {"ALEXA": 0.93, …}
scores = [m.predict(frame) for frame in frames_80ms]
dets = m.detections                       # consumed on read
```

CLI:

```
pypryon run   <model_dir> <audio.(raw|wav16k)> [--scores]
pypryon convert <model_dir>               # portable ONNX → .pypryon-cache/
```

`run` prints one line per detection; `convert` writes the stock-ONNX
version of the graph to `.pypryon-cache/` under the current directory
(never next to the model — model dirs are firmware extractions and often
read-only).

## Getting a model directory

Extract your own firmware (no model bytes ship with this package):

1. **FTVDB**: fetch a device OTA bin (e.g. Echo Dot "donut" NS65741).
2. **payload extraction**: `payload-dumper-go` / Android `payload_dumper`
   unpacks the OTA payload to images.
3. **debugfs** (or `mount -o loop`): pull
   `/system/local/models/keyword/en-US/ALEXA/` out of the system image —
   it should contain `pryon.manifest`, `pryon.config`, `kw.cfg.json`,
   `words.shrunk.txt`, `nemort_scorer_config.json` and
   `runtime_stateless_standard.streaming.fixed.int16.onnx`.

DAVS-downloaded model dirs (the multi-graph NTT-fusion generation) are a
follow-up (M4) and not yet supported.

## Legal

Personal-use extraction of models your own devices run is the premise
(same stance as the rest of this repo, EFF DMCA §1201 voice-assistant
exemption). This package is **code only**: no Amazon-derived bytes — no
models, weights, firmware or audio — are redistributed here or belong in
its git history. Test fixtures are derived data (parsed detection JSON);
synthetic clips are generated deterministically by `tools/make_clips.py`.

## Provenance & credits

- The RE substrate — LFBE semantics, the CPUNK per-row quantization rule,
  the 25-lane LSTM ring contract, the decoder rule — is vendored from
  PR #171 of the echo-dot-assist project (BenStein1), pinned at commit
  `dd56117` under `vendor/pryon-native/` with its
  [LEDGER](vendor/pryon-native/LEDGER.md) and third-party notices; MIT.
  Every engine-derived constant in `pypryon/` carries a provenance comment
  (vendor LEDGER section, model file, or oracle log).
- The oracle harness (pinned `pryon_test` build, qemu-arm + qrun.sh) and
  firmware come from the sibling `echo-dot-assist` checkout.
- Original work here: the stock-ONNX port of the Hybrid* kernels
  (`pypryon/graph.py`), the numpy LFBE front end, the streaming runner and
  decoder, the kwSampleStartIndex derivation, and the packaged API.

See `recovery-notes/pypryon-design.md` in the parent repo for the design,
and `docs/` there for the RE record.
