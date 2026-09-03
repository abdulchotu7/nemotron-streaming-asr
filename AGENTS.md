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
- `pipeline/session.py` — `NemotronStreamingSession`: orchestrator; `feed(pcm)` → `step()` per chunk (cumulative `AlignedResult`), `finish()` flushes the trailing partial, `reset()` starts a new utterance. Latency = `att_context_size=[56, lookahead]`, lookahead ∈ {0,1,3,6,13} (80 ms frames). `language="auto"` enables live language detection: the decoder latches the first emitted `<xx-XX>` tag (`decoder.detected_language` / `session.detected_language`) and the session switches the encoder prompt for later chunks.
- `benchmark/` — `PerformanceStats` (optional, zero-impact), `StreamingBenchmark` runner, system/memory metrics.
- `apps/dictation/` — layered: `hotkey.py` (pynput, right-Option tap-to-toggle) → `app.py` (fresh session per recording) → `microphone.py` (20 ms blocks) → `transcript.py` (cumulative) → `text_insertion.py` (clipboard-safe ⌘V via `CGEventPost`). Displays sit behind the `display.py` `RecordingDisplay` seam (`status/on_partial/set_level/show/hide/tick`; overlay with console fallback). Caret sensing lives in `caret.py` (`place_panel` is pure and headless-testable). One recording is an `utterance.py` `Utterance` (worker + drain + finish + paste behind `start/stop/join`); the app keeps hotkey, session construction, and display. The recording only stops on a hotkey tap — pauses for thinking are fine, there is no auto-stop.

## Conventions

- `session.feed(block)` then `session.step()` per chunk; always `session.finish()` before reading final text.
- Dictation layers are constructor-injected (`hotkey`/`recorder`/`insertion`/`ui`) — swap without touching the app. Production wiring lives in `DictationApp.build_default()`; `build_display()` picks overlay-or-console.
- Hotkey: `PynputGlobalHotkey(key="alt_r")` — tap-to-toggle only, right Option = `alt_r` (left Option is `alt` and never triggers). Release alone never stops — only a fresh trigger press does. No modifier combos or hold mode.
- macOS permissions: **Input Monitoring** for pynput, **Accessibility** for the synthetic ⌘V paste. App warns at startup if paste permission is missing.
- Tests use `tiny_model` + `seeded_audio` session fixtures from `tests/conftest.py` (deterministic RNG) and prove frame/token equivalence to the reference impl.
- `data/` and the sample WAV are gitignored — never commit them.

## Notes

<!-- quick-add scratchpad: -->
<!-- review/optimisations round: fixed dictation bugs (toggle requires modifiers; modifier-only hotkey CLI error; swallowed rapid stop->start tap), trim view pinning of fed buffers, lazy mel waveform conversion on no-op steps, LatencyProbe float64 copy; 4 regression tests. Pipeline math untouched (reference-identical). -->
<!-- simplification round: hotkey is now toggle-only right Option (PynputGlobalHotkey(key="alt_r")); removed --hotkey/--toggle CLI flags, hold mode and modifier machinery; hotkey tests rewritten. -->
<!-- mic demo: session.finish() flush on exit so Ctrl+C keeps the trailing partial chunk. -->
<!-- gap-closing round: real-model integration + 6-min benchmark validated (RTF ~0.05-0.15, memory bounded; data/ WAV restored locally, gitignored); auto language detection (language="auto" -> decoder latches <xx-XX> tag, session re-prompts); EnergyVAD auto-stop (--no-auto-stop, --stop-silence); floating OverlayUI (--overlay); removed dead duplicate utils/tokenizer.py (upstream tok has the same helpers). -->
<!-- benchmark note: 'GROWTH DETECTED: python_heap_bytes' is the stats collector's own token_latency bucket (1 sample/token, capped by max_samples), not a pipeline leak; all pipeline signals flat. -->
<!-- scope round: removed overlay UI and VAD auto-stop at user request -- dictation is tap-to-toggle only, recording never stops on silence, final text is pasted at the cursor; pipeline auto-language detection (language="auto") kept but dictation default stays en-US. -->

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature>`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

