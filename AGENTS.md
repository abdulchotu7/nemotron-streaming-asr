# AGENTS.md

Real-time offline streaming ASR for NVIDIA Nemotron 3.5 (RNNT) on Apple Silicon, built on MLX. Stateful, cache-aware pipeline that is O(n) and token-identical to the stateless reference implementation.

## Project

- Python ≥3.10, src-layout package (`src/nemotron_streaming_asr/`), built with setuptools. Dependency manager in use: `uv`.
- Heavy deps: `mlx`, `mlx-audio` (pinned git dep), `mlx-lm`, `transformers`, `numpy`, `sounddevice`, `pynput`, `pyobjc-framework-cocoa/quarts`.
- CLI entry points (declared in `pyproject.toml` `[project.scripts]`):
  - `nemotron-dictation` — desktop dictation app (default: tap right Option ⌥ to start/stop)
  - `nemotron-mic` — console live demo
  - `nemotron-benchmark` — benchmark harness

## Commands

- Install: `uv pip install -e .` (deps are also frozen in `requirements.txt`; don't install the project from there — only its third-party deps)
- Tests: `python -m pytest tests/ -q` — unit + equivalence suite (uses a tiny random model; no weights download). Integration/slow excluded by default:
  - `-m integration` — needs the real HuggingFace model + `data/*.wav` (gitignored)
  - `-m slow` — multi-second benchmark runs
  - Unit-only fast pass: `python -m pytest tests/ -m "not integration and not slow"` (54 tests)
- No linter/formatter is configured — match surrounding style.

## Architecture

- `pipeline/audio_buffer.py` — `StreamingAudioBuffer`: bounded PCM history, trims processed audio aggressively (~1.2 s STFT window).
- `pipeline/encoder.py` — `StreamingEncoder`: stateful cache-aware FastConformer (attn/conv/mel caches). The stateless `stream_encode_chunks()` / `stream_encode()` reference impls are kept for regression tests — do not delete.
- `pipeline/decoder.py` — `StreamingDecoder`: stateful greedy RNN-T.
- `pipeline/session.py` — `NemotronStreamingSession`: orchestrator; `feed(pcm)` → `step()` per chunk (cumulative `AlignedResult`), `finish()` flushes the trailing partial, `reset()` starts a new utterance. Latency = `att_context_size=[56, lookahead]`, lookahead ∈ {0,1,3,6,13} (80 ms frames).
- `benchmark/` — `PerformanceStats` (optional, zero-impact), `StreamingBenchmark` runner, system/memory metrics.
- `apps/dictation/` — layered: `hotkey.py` (pynput, hold or tap-to-toggle) → `app.py` (fresh session per recording) → `microphone.py` (20 ms blocks) → `transcript.py` (cumulative) → `text_insertion.py` (clipboard-safe ⌘V via `CGEventPost`).

## Conventions

- `session.feed(block)` then `session.step()` per chunk; always `session.finish()` before reading final text.
- Dictation layers are constructor-injected (`hotkey`/`recorder`/`ui`) — swap without touching the app.
- Hotkey: `PynputGlobalHotkey(modifiers=..., key=..., toggle=...)`. Toggle mode requires a trigger key; right Option = `alt_r` (left Option is `alt`, indistinguishable from the family). Release alone never stops in toggle mode — only a fresh trigger press does.
- macOS permissions: **Input Monitoring** for pynput, **Accessibility** for the synthetic ⌘V paste. App warns at startup if paste permission is missing.
- Tests use `tiny_model` + `seeded_audio` session fixtures from `tests/conftest.py` (deterministic RNG) and prove frame/token equivalence to the reference impl.
- `data/` and the sample WAV are gitignored — never commit them.

## Notes

<!-- quick-add scratchpad: (empty) -->

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature>`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

