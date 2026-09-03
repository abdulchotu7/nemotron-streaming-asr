"""NemotronStreamingSession: long-session identity, reset, bounded memory,
and zero-impact benchmark instrumentation."""

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.nemotron_asr import streaming as ref_streaming
from mlx_audio.stt.models.nemotron_asr.audio import iter_log_mel_spectrogram

from nemotron_streaming_asr import PerformanceStats, NemotronStreamingSession

from .conftest import BLOCK, run_live, token_ids


def _run_with_stats(model, audio, stats):
    session = NemotronStreamingSession(model, language="en-US", stats=stats)
    return [token_ids(r) for r in run_live(session, audio)]


def test_long_session_token_identical(tiny_model):
    """A ~40 s live session (feed 20 ms blocks, step each, finish) must be
    token-identical to the offline reference over all mel frames."""
    np.random.seed(1)
    audio = (np.random.randn(int(40 * 16000)) * 0.05).astype(np.float32)

    session = NemotronStreamingSession(tiny_model, language="en-US")
    results = run_live(session, audio)

    # Every fed sample was consumed.
    assert session.audio.total_samples == audio.shape[0]

    # Reference over the full offline mel sequence (incl. final flush).
    iter_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(audio), tiny_model.preprocessor_config, chunk_frames=112
        )
    )
    ref = list(
        tiny_model._decode_prompted_chunks(
            ref_streaming.stream_encode_chunks(tiny_model, iter(iter_chunks), "en-US")
        )
    )

    assert len(results) == len(ref), (len(results), len(ref))
    for r, expected in zip(results, ref):
        assert token_ids(r) == token_ids(expected)


def test_memory_bounded_during_long_session(tiny_model):
    np.random.seed(1)
    audio = (np.random.randn(int(40 * 16000)) * 0.05).astype(np.float32)

    session = NemotronStreamingSession(tiny_model, language="en-US")
    max_held = 0
    for i in range(0, audio.shape[0], BLOCK):
        session.feed(audio[i : i + BLOCK])
        list(session.step())
        max_held = max(max_held, session.audio._length)

    assert max_held < 20000, f"retained {max_held} samples"
    assert max_held < audio.shape[0] / 10


def test_memory_footprint_reports_session_state(tiny_model, seeded_audio):
    """memory_footprint() is the single seam for sampling session state:
    flat dict, identical keys before/after warmup, non-negative ints."""
    session = NemotronStreamingSession(tiny_model, language="en-US")

    cold = session.memory_footprint()
    assert list(cold) == [
        "waveform_samples",
        "pending_mel_frames",
        "mel_cache_frames",
        "attn_cache_elems",
        "conv_cache_elems",
    ]
    assert all(isinstance(v, int) and v >= 0 for v in cold.values())
    assert cold["waveform_samples"] == 0

    session.feed(seeded_audio[:BLOCK])
    fed = session.memory_footprint()
    assert fed["waveform_samples"] == BLOCK

    list(session.step())
    list(session.finish())
    warm = session.memory_footprint()
    assert list(warm) == list(cold)  # keys stable across warmup
    assert all(isinstance(v, int) and v >= 0 for v in warm.values())


def test_reset_starts_fresh_session(tiny_model, seeded_audio):
    np.random.seed(2)
    audio_b = (np.random.randn(int(3 * 16000)) * 0.08).astype(np.float32)

    session = NemotronStreamingSession(tiny_model, language="en-US")
    r1 = [token_ids(r) for r in run_live(session, seeded_audio)]

    # Same session, same audio again after reset == fresh run.
    session.reset()
    r1b = [token_ids(r) for r in run_live(session, seeded_audio)]
    assert r1 == r1b

    # Same session, new audio after reset == a brand-new session.
    session.reset()
    r2 = [token_ids(r) for r in run_live(session, audio_b)]
    fresh = NemotronStreamingSession(tiny_model, language="en-US")
    rf = [token_ids(r) for r in run_live(fresh, audio_b)]
    assert r2 == rf


def test_component_reset(tiny_model, seeded_audio):
    from nemotron_streaming_asr.pipeline.decoder import StreamingDecoder
    from nemotron_streaming_asr.pipeline.encoder import StreamingEncoder

    enc = StreamingEncoder(tiny_model)
    dec = StreamingDecoder(tiny_model)
    mel_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(seeded_audio), tiny_model.preprocessor_config, chunk_frames=112
        )
    )

    enc.reset()
    a = [mx.array(x) for x in _prompted(enc, mel_chunks[:2])]
    enc.reset()
    b = [mx.array(x) for x in _prompted(enc, mel_chunks[:2])]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert tuple(x.shape) == tuple(y.shape)
        assert float(mx.max(mx.abs(x - y))) == 0.0

    dec.reset()
    da = [token_ids(dec.feed(p)) for p in a]
    dec.reset()
    db = [token_ids(dec.feed(p)) for p in a]
    assert da == db


def _prompted(enc, mel_chunks):
    out = []
    for m in mel_chunks:
        out.extend(list(enc.feed(m, "en-US")))
    out.extend(list(enc.finish("en-US")))
    return out


def test_benchmark_instrumentation_zero_impact(tiny_model, seeded_audio):
    """stats=None, stats(enabled=False) and stats(enabled=True) must all
    produce identical recognition results; enabled stats must collect samples."""
    base = _run_with_stats(tiny_model, seeded_audio, None)
    assert len(base) > 0

    assert _run_with_stats(tiny_model, seeded_audio, PerformanceStats(enabled=False)) == base

    enabled = PerformanceStats(enabled=True)
    assert _run_with_stats(tiny_model, seeded_audio, enabled) == base

    for key in ("audio_feed", "mel", "encoder", "decoder", "step",
                "end_to_end", "token_latency", "eval", "clear_cache"):
        assert enabled.count(key) > 0, key
    assert all(v > 0 for v in enabled.samples("end_to_end"))
    assert enabled.count("eval") > 0 and enabled.count("clear_cache") > 0
    assert enabled.count("tokens") > 0 and enabled.count("words") >= 0


# ------------------------------------------------------------------- language
def test_auto_language_switches_prompt(tiny_model, seeded_audio):
    """In 'auto' prompt mode a detected language tag re-prompts the encoder
    for the following chunks (first detection wins)."""
    session = NemotronStreamingSession(tiny_model, language="auto")
    assert session.detect_language is True

    class StubDecoder:
        """Feeds back a fake result and latches a language tag."""

        def __init__(self):
            self.detected_language = "en-US"
            self.calls = 0

        def feed(self, prompted):
            self.calls += 1
            return type("R", (), {"text": "hi", "sentences": []})()

        def reset(self):
            self.detected_language = None

    session.decoder = StubDecoder()

    # Feed enough audio for at least one full native chunk (112 mel frames).
    for i in range(0, 60 * BLOCK, BLOCK):
        session.feed(seeded_audio[i : i + BLOCK])
    results = list(session.step())

    assert results, "expected at least one prompted chunk"
    assert session.decoder.calls == len(results)
    assert session.language == "en-US", "prompt language must switch"

    # Reset restores the configured prompt language.
    session.reset()
    assert session.language == "auto"
    assert session.decoder.detected_language is None


def test_detection_disabled_for_fixed_language(tiny_model):
    """A fixed prompt language never switches, even if the decoder latches a
    tag (the tag is still recorded for callers)."""
    session = NemotronStreamingSession(tiny_model, language="en-US")
    assert session.detect_language is False

    session.decoder.detected_language = "fr-FR"
    session._apply_detected_language()
    assert session.language == "en-US"

    session.reset()
    assert session.language == "en-US"
