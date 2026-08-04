"""Latency operating-point validation (NVIDIA model card).

The model is trained for chunk latencies of 80/160/320/560/1120 ms via
``att_context_size = [left, right]`` in 80 ms frames (right = lookahead).
Each operating point must remain frame/token-identical to the reference
implementation at that same configuration.
"""

import mlx.core as mx
import pytest

from mlx_audio.stt.models.nemotron_asr import streaming as ref_streaming
from mlx_audio.stt.models.nemotron_asr.audio import iter_log_mel_spectrogram

from nemotron_streaming_asr import NemotronStreamingSession
from nemotron_streaming_asr.pipeline.audio_buffer import StreamingAudioBuffer
from nemotron_streaming_asr.pipeline.encoder import StreamingEncoder

from .conftest import run_live, token_ids

# (lookahead, chunk latency ms, mel frames per chunk)
OPERATING_POINTS = [(0, 80, 8), (1, 160, 16), (3, 320, 32), (6, 560, 56), (13, 1120, 112)]


@pytest.mark.parametrize("lookahead,latency_ms,chunk_mel", OPERATING_POINTS)
def test_buffer_chunk_size(tiny_model, seeded_audio, lookahead, latency_ms, chunk_mel):
    buf = StreamingAudioBuffer(tiny_model, att_context_size=[56, lookahead])
    assert buf.chunk_mel == chunk_mel

    emitted = []
    for i in range(0, seeded_audio.shape[0], 320):
        buf.feed(seeded_audio[i : i + 320])
        emitted.extend(list(buf.get_ready_mel_chunks()))
    assert emitted
    assert all(c.shape[1] == chunk_mel for c in emitted)


@pytest.mark.parametrize("lookahead,latency_ms,chunk_mel", OPERATING_POINTS)
def test_encoder_identical_at_operating_point(tiny_model, seeded_audio, lookahead, latency_ms, chunk_mel):
    """feed()+finish() == stream_encode_chunks() at the same att_context_size."""
    acs = [56, lookahead]
    mel_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(seeded_audio), tiny_model.preprocessor_config,
            chunk_frames=chunk_mel,
        )
    )
    ref = list(
        ref_streaming.stream_encode_chunks(tiny_model, iter(mel_chunks), "en-US",
                                           att_context_size=acs)
    )
    enc = StreamingEncoder(tiny_model, att_context_size=acs)
    stateful = []
    for m in mel_chunks:
        stateful.extend(list(enc.feed(m, "en-US")))
    stateful.extend(list(enc.finish("en-US")))

    assert len(stateful) == len(ref), (lookahead, len(stateful), len(ref))
    for a, b in zip(ref, stateful):
        a, b = mx.array(a), mx.array(b)
        assert tuple(a.shape) == tuple(b.shape)
        assert float(mx.max(mx.abs(a - b))) == 0.0


@pytest.mark.parametrize("lookahead,latency_ms,chunk_mel", OPERATING_POINTS)
def test_session_matches_offline_at_operating_point(tiny_model, seeded_audio, lookahead, latency_ms, chunk_mel):
    """The live session (buffer + encoder + decoder, incl. finish) must decode
    the same tokens as the offline reference at the same att_context_size."""
    acs = [56, lookahead]
    session = NemotronStreamingSession(tiny_model, language="en-US", att_context_size=acs)
    results = run_live(session, seeded_audio)

    from mlx_audio.stt.models.nemotron_asr.audio import log_mel_spectrogram

    mel = log_mel_spectrogram(mx.array(seeded_audio), tiny_model.preprocessor_config)
    offline = tiny_model.decode(mel, language="en-US", att_context_size=acs)

    final = results[-1]
    assert token_ids(final) == token_ids(offline), (
        f"lookahead={lookahead}: live={final.text!r} offline={offline.text!r}"
    )


def test_unsupported_lookahead_still_coherent(tiny_model):
    """An untrained operating point still works (cache math is generic); the
    session warns via logging but stays coherent."""
    session = NemotronStreamingSession(tiny_model, language="en-US", att_context_size=[56, 5])
    assert session.att_context_size == [56, 5]
    assert session.audio.chunk_mel == (5 + 1) * 8  # still coherent
    assert session.encoder.chunk_mel == (5 + 1) * 8
