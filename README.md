# nemotron-streaming-asr

Real-time **offline** streaming ASR engine for [NVIDIA Nemotron 3.5 Streaming ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) on Apple Silicon, built on MLX.

A stateful, cache-aware streaming pipeline (FastConformer-RNNT with language-ID prompt conditioning) for live microphone input. The recognition math is **identical** to the reference implementation — only the storage of streaming state changed, so the engine is O(n) with no recompute.

The pipeline mirrors the official model card and processor configuration: hop 160, n_fft 512, 16 kHz, 128 mel bins, 128 language prompts, and the same cache-aware, non-overlapping chunk streaming as NeMo's reference implementation.

## Architecture

```
Microphone
    │
    ▼
StreamingAudioBuffer      bounded PCM history → (1, chunk_mel, 128) log-mel chunks
    │
    ▼
StreamingEncoder          stateful cache-aware FastConformer (attn/conv/mel caches)
    │
    ▼
StreamingDecoder          stateful greedy RNN-T (hidden state, hypothesis)
    │
    ▼
NemotronStreamingSession  thin orchestrator; owns language, chunk size, reset()
```

```
src/nemotron_streaming_asr/
├── pipeline/
│   ├── audio_buffer.py    StreamingAudioBuffer (bounded, trimming, tail flush)
│   ├── encoder.py         StreamingEncoder + stateless reference impl
│   ├── decoder.py         StreamingDecoder
│   └── session.py         NemotronStreamingSession (+ latency operating points)
├── benchmark/
│   ├── stats.py           PerformanceStats (thread-safe collector, rolling/final reports)
│   ├── system.py          CPU %, RSS, unified memory, MLX memory
│   └── runner.py          StreamingBenchmark + CLI
├── apps/
│   ├── mic.py             console live demo (+ LatencyProbe)
│   └── dictation/         WhisperFlow-style desktop app (hotkey → mic → insert)
│       ├── hotkey.py      GlobalHotkey (pynput, hold-to-talk)
│       ├── microphone.py  MicrophoneRecorder (20 ms blocks)
│       ├── transcript.py  LiveTranscriptController (newest cumulative text)
│       ├── text_insertion.py  TextInsertionService (clipboard-safe ⌘V paste)
│       └── app.py         DictationApp (session lifecycle + console UI)
└── utils/tokenizer.py     standalone vocabulary helpers
tests/                     pytest suite (equivalence, buffer, session, chunk config, stats, dictation, integration)
examples/                  legacy prototype
data/                      sample audio (gitignored)
```

## Install

```bash
# runtime
python -m venv .venv && source .venv/bin/activate
uv pip install -e .                     # or: pip install -e .

# tests
uv pip install -e '.[test]'
```

The package requires the `mlx-audio` git dependency pinned in `pyproject.toml`
(mirrored by `requirements.txt`, the frozen environment lock).

## Usage

```python
from mlx_audio.stt import load
from nemotron_streaming_asr import NemotronStreamingSession

model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
session = NemotronStreamingSession(model, language="en-US")

session.feed(pcm_block)              # raw mono float32 PCM, any block size
for result in session.step():        # one cumulative AlignedResult per chunk
    print(result.text)

# push-to-talk: end of utterance
for result in session.finish():      # flushes the trailing partial chunk
    print(result.text)

# start a new utterance without reloading the model
session.reset()
```

`language` is a prompt key from the model's 128-locale dictionary, e.g. `"en-US"`,
`"de-DE"`, or `"auto"` (automatic language detection; the model then emits a
`<xx-XX>` language tag after the terminal punctuation, which
`utils/tokenizer.py` can strip or expose).

Live microphone demo:

```bash
nemotron-mic
# or: python -m nemotron_streaming_asr.apps.mic
```

## Latency: configurable operating points

Per the [NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
latency is set by `att_context_size = [left, right]` in **80 ms frames**
(`right` = lookahead). All five operating points are trained and selectable at
inference time — no retraining:

| att_context_size | lookahead | chunk | chunk latency |
|---|---|---|---|
| `[56, 0]`  | 0  | 1 frame  | 80 ms   |
| `[56, 1]`  | 1  | 2 frames | 160 ms  |
| `[56, 3]`  | 3  | 4 frames | 320 ms  |
| `[56, 6]`  | 6  | 7 frames | 560 ms  |
| `[56,13]`  | 13 | 14 frames | 1120 ms (default, matches the MLX port) |

```python
session = NemotronStreamingSession(model, language="en-US",
                                   att_context_size=[56, 3])   # 320 ms chunks
```

`nemotron-mic --lookahead 3` and `nemotron-benchmark --lookahead 3` expose the
same choice from the CLIs (choices `0|1|3|6|13`, default `13`). Every operating
point is regression-tested to be **frame-identical to the reference encoder**
and **token-identical to offline decoding** (`tests/test_chunk_config.py`).

Measured on the real 0.6B model (M5):

| lookahead | chunk latency | processing/chunk | first words after speech onset |
|---|---|---|---|
| 13 | 1120 ms | ~87 ms | ~1194 ms → `'Im sure you'` |
| 3  | 320 ms  | ~51 ms | ~975 ms  → `'Im sure'` |
| 0  | 80 ms   | ~40 ms | ~810 ms  → `'im'` |

Notes on what these numbers mean:

- The gap from **speech onset to first visible words** is bounded at ~0.8 s even
  at 80 ms chunks: the RNN-T decoder only emits the first token once enough
  speech frames have accumulated for acoustic confidence. Smaller chunks reach
  that point sooner and surface earlier *partial* words; the final transcript is
  unchanged (results are cumulative).
- Smaller chunks update the terminal more often (e.g. 3.1 vs 0.9 chunks/s) with
  less processing per chunk; the model card's WER tables show slightly higher
  accuracy at larger chunk sizes (the latency-accuracy Pareto curve).

The mic demo's `LatencyProbe` reports the live user-perceived gap:

```
[latency] speech onset detected        412 ms after mic start
[latency] first words after          1229 ms of speech | 1641 ms from mic start
[latency]   -> 'Im sure you'
```

(Disable with `--no-latency`; tune the energy threshold with `--vad-threshold`.)

## Desktop dictation app (WhisperFlow-style)

Hold the global hotkey (default **⌘⌥**), speak, release — the live cumulative
transcript is shown while you talk and the final transcript is **pasted at the
current cursor**:

```bash
nemotron-dictation
# or: python -m nemotron_streaming_asr.apps.dictation

nemotron-dictation --lookahead 3      # 320 ms chunks (snappier partials)
nemotron-dictation --no-insert        # print only, no paste
nemotron-dictation --hotkey cmd+shift # custom hotkey
```

```
Ready. Hold ⌘⌥ (Cmd+Option) and speak; release to insert.
Listening...
I'm sure
I'm sure you have a lot
...
✓ I'm sure you have a lot of questions and im gonna try to answer them all here like
Ready for next recording.
```

Layering (each layer is swappable):

```
GlobalHotkey ──press/release──▶ DictationApp (recording worker)
                                      │
                                      ▼
MicrophoneRecorder ──20 ms PCM──▶ NemotronStreamingSession (black box)
                                      │
                                      ▼
LiveTranscriptController ──▶ UI (console now, overlay later)
                                      │
                                      ▼
TextInsertionService (clipboard-safe ⌘V paste at cursor)
```

- A **fresh session is created per recording** (hotkey press) and destroyed
  after the final text is inserted — no state leaks between recordings.
- The transcript is cumulative: only the newest text is displayed; partials
  overwrite the current line.
- **Text insertion** uses native macOS APIs: `NSPasteboard` (clipboard
  snapshot / set / restore) plus a synthetic **⌘V** via `CGEventPost`. The
  user's clipboard is preserved and restored afterward.
- **Permissions** (macOS System Settings):
  - *Input Monitoring* — for the global hotkey (pynput listener),
  - *Accessibility* — for the synthetic ⌘V paste (`CGEventPost`).
  The app warns at startup if paste permission is missing.
- Console UI is the default; the UI is a thin listener on
  `LiveTranscriptController`, so a floating overlay can replace it later.

## Benchmarking

```bash
nemotron-benchmark --durations 60 300 900 --language en-US
nemotron-benchmark --durations 10 --no-realtime          # burst/throughput mode
nemotron-benchmark --audio some.wav --rolling 2
nemotron-benchmark --lookahead 3 --durations 60          # 320 ms chunks
```

Instrumentation is **optional and zero-impact**: attach `PerformanceStats` to the
session and every stage is timed with `time.perf_counter_ns()`, synchronized via
`mx.eval` so timings reflect real execution:

```python
from nemotron_streaming_asr import PerformanceStats, NemotronStreamingSession

stats = PerformanceStats(enabled=True, rolling_interval_s=5.0)
session = NemotronStreamingSession(model, language="en-US", stats=stats)
# ... feed / step ...
stats.stop()
stats.print_final_report()
```

Measured per emitted result: audio feed, mel extraction, encoder, decoder,
session step, end-to-end and token latency; plus `mx.eval` / `mx.clear_cache`
costs, throughput (chunks/tokens/words per second), rolling 5 s reports, and
memory stability (waveform, caches, Python heap, MLX memory) for 1/5/15-minute
continuous runs. The final report includes **Time to first words** (audio start
-> first non-empty result). With `stats=None` or
`stats=PerformanceStats(enabled=False)` the engine behaves exactly as the
uninstrumented build (verified by the test suite).

## Tests

```bash
python -m pytest tests/ -q                          # unit + equivalence (tiny model)
python -m pytest tests/ -m integration              # real model + sample WAV
python -m pytest tests/ -m slow                     # benchmark harness smoke
```

The equivalence suite proves the stateful pipeline is bit/token-identical to
the reference implementation: encoder frames match `stream_encode_chunks()`
exactly (`max|diff| = 0`), decoder tokens match `_decode_prompted_chunks()`,
live buffer mel matches offline extraction, a 40 s live session is
token-identical to the offline reference including the final flush, and all
five latency operating points match their offline equivalents.

## Notes

- `stream_encode_chunks()` / `stream_encode()` are kept as the stateless
  reference implementation for regression testing (do not delete).
- The audio buffer trims processed PCM aggressively (keeps only the STFT window
  + one preemphasis sample, ~1.2 s); mel values are bit-identical to offline
  extraction.
- MLX exposes no GPU utilization percentage; `benchmark/system.py` reports MLX
  active/peak/cache memory and device info instead.
- **Model & license**: the MLX port we run is the multilingual
  `nemotron-3.5-asr-streaming-0.6b-8bit` (40 language-locales). The official
  NVIDIA model is licensed under **OpenMDW-1.1** (check its terms before
  shipping). For English-only use cases the model card recommends NVIDIA's
  English-only `nemotron-speech-streaming-en-0.6b` instead.
