"""StreamingAudioBuffer: bounded storage, bit-identical mel, offsets, reset."""

import mlx.core as mx
import numpy as np

from mlx_audio.stt.models.nemotron_asr.audio import iter_log_mel_spectrogram

from nemotron_streaming_asr.pipeline.audio_buffer import StreamingAudioBuffer

from .conftest import BLOCK


def _collect(buf, audio):
    emitted = []
    for i in range(0, audio.shape[0], BLOCK):
        buf.feed(audio[i : i + BLOCK])
        emitted.extend(list(buf.get_ready_mel_chunks()))
    return emitted


def test_mel_chunks_bit_identical_to_offline(tiny_model, seeded_audio):
    """Incrementally emitted mel chunks must equal offline extraction of the
    full waveform (no incremental zero-padding, no trim artifacts)."""
    ref_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(seeded_audio), tiny_model.preprocessor_config, chunk_frames=112
        )
    )
    ref_full = [c for c in ref_chunks if c.shape[1] == 112]

    buf = StreamingAudioBuffer(tiny_model)
    emitted = _collect(buf, seeded_audio)

    assert len(emitted) == len(ref_full)
    for a, b in zip(ref_full, emitted):
        diff = float(mx.max(mx.abs(mx.array(a) - mx.array(b))))
        assert diff == 0.0  # bit-identical


def test_trimming_is_transparent(tiny_model, seeded_audio):
    """The waveform trim must not change mel output at all."""
    trimmed = StreamingAudioBuffer(tiny_model)
    untrimmed = StreamingAudioBuffer(tiny_model)
    untrimmed._can_trim = False  # keep the full waveform (original behavior)

    a = _collect(trimmed, seeded_audio)
    b = _collect(untrimmed, seeded_audio)

    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert float(mx.max(mx.abs(mx.array(x) - mx.array(y)))) == 0.0


def test_memory_bounded(tiny_model, seeded_audio):
    """Retained waveform stays bounded (~1.2 s) no matter how much audio flows."""
    long_audio = (np.random.randn(int(40 * 16000)) * 0.05).astype(np.float32)
    buf = StreamingAudioBuffer(tiny_model)
    max_held = 0
    for i in range(0, long_audio.shape[0], BLOCK):
        buf.feed(long_audio[i : i + BLOCK])
        list(buf.get_ready_mel_chunks())
        max_held = max(max_held, buf._length)

    assert max_held < 20000, f"retained {max_held} samples"
    assert buf.total_samples == long_audio.shape[0]


def test_reset(tiny_model, seeded_audio):
    buf = StreamingAudioBuffer(tiny_model)
    a = _collect(buf, seeded_audio)
    assert buf.next_mel_frame > 0 and buf.total_samples > 0

    buf.reset()
    assert buf._length == 0
    assert buf.trim_sample == 0 and buf.total_samples == 0
    assert buf.next_mel_frame == 0

    b = _collect(buf, seeded_audio)  # fresh session == first run
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert float(mx.max(mx.abs(mx.array(x) - mx.array(y)))) == 0.0


def test_tail_flush(tiny_model, seeded_audio):
    """get_tail_mel_chunks emits the sub-chunk remainder and matches the
    offline tail chunk exactly."""
    ref_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(seeded_audio), tiny_model.preprocessor_config, chunk_frames=112
        )
    )
    tail = ref_chunks[-1]  # 27 frames
    assert tail.shape[1] < 112

    buf = StreamingAudioBuffer(tiny_model)
    for i in range(0, seeded_audio.shape[0], BLOCK):
        buf.feed(seeded_audio[i : i + BLOCK])
        list(buf.get_ready_mel_chunks())

    emitted_tail = list(buf.get_tail_mel_chunks())
    assert len(emitted_tail) == 1
    assert emitted_tail[0].shape == tail.shape
    assert float(mx.max(mx.abs(mx.array(emitted_tail[0]) - mx.array(tail)))) == 0.0
