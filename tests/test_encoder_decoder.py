"""Encoder/decoder equivalence against the reference implementation.

The reference (ground truth) is the original stateless implementation shipped
in ``mlx_audio.stt.models.nemotron_asr.streaming`` (``stream_encode_chunks``)
and ``Model._decode_prompted_chunks``.
"""

import mlx.core as mx
import pytest

from mlx_audio.stt.models.nemotron_asr import streaming as ref_streaming
from mlx_audio.stt.models.nemotron_asr.audio import iter_log_mel_spectrogram

from nemotron_streaming_asr.pipeline.decoder import StreamingDecoder
from nemotron_streaming_asr.pipeline.encoder import StreamingEncoder

from .conftest import token_ids


@pytest.fixture(scope="module")
def mel_chunks(tiny_model, seeded_audio):
    # 251 mel frames -> [112, 112, 27]: full native chunks plus a partial tail.
    return list(
        iter_log_mel_spectrogram(
            mx.array(seeded_audio), tiny_model.preprocessor_config, chunk_frames=112
        )
    )


@pytest.fixture(scope="module")
def reference_prompted(tiny_model, mel_chunks):
    return list(
        ref_streaming.stream_encode_chunks(tiny_model, iter(mel_chunks), "en-US")
    )


def test_encoder_matches_reference(tiny_model, mel_chunks, reference_prompted):
    """feed()+finish() must be frame-identical to stream_encode_chunks(),
    including the final is_final=True flush of the partial tail."""
    enc = StreamingEncoder(tiny_model)
    stateful = []
    for m in mel_chunks:
        stateful.extend(list(enc.feed(m, "en-US")))
    stateful.extend(list(enc.finish("en-US")))

    assert len(stateful) == len(reference_prompted)
    for a, b in zip(reference_prompted, stateful):
        a, b = mx.array(a), mx.array(b)
        assert a.shape == b.shape
        assert float(mx.max(mx.abs(a - b))) == 0.0  # bit-identical


def test_encoder_open_stream_matches_reference(tiny_model, mel_chunks):
    """Live feed() semantics: full chunks processed non-final match a reference
    stream that ends with an empty sentinel chunk (open-ended stream)."""
    native = [m for m in mel_chunks if m.shape[1] == 112]
    empty = mx.zeros((1, 0, native[0].shape[2]), dtype=native[0].dtype)
    ref = list(
        ref_streaming.stream_encode_chunks(tiny_model, iter(native + [empty]), "en-US")
    )

    enc = StreamingEncoder(tiny_model)
    stateful = []
    for m in native:
        stateful.extend(list(enc.feed(m, "en-US")))
    stateful.extend(list(enc.finish("en-US")))

    assert len(stateful) == len(ref)
    for a, b in zip(ref, stateful):
        a, b = mx.array(a), mx.array(b)
        assert tuple(a.shape) == tuple(b.shape)
        assert float(mx.max(mx.abs(a - b))) == 0.0


def test_decoder_matches_reference(tiny_model, reference_prompted):
    """feed() must be token-identical to _decode_prompted_chunks()."""
    ref = list(tiny_model._decode_prompted_chunks(iter(reference_prompted)))

    dec = StreamingDecoder(tiny_model)
    stateful = [dec.feed(p) for p in reference_prompted]

    assert len(stateful) == len(ref)
    for a, b in zip(ref, stateful):
        assert a.text == b.text
        assert token_ids(a) == token_ids(b)


def test_finish_matches_final_flush(tiny_model, mel_chunks):
    """finish() reproduces the reference's final is_final=True flush."""
    enc = StreamingEncoder(tiny_model)
    stateful = []
    for m in mel_chunks:
        stateful.extend(list(enc.feed(m, "en-US")))
    stateful.extend(list(enc.finish("en-US")))

    ref = list(
        ref_streaming.stream_encode_chunks(tiny_model, iter(mel_chunks), "en-US")
    )
    assert len(stateful) == len(ref)
    for a, b in zip(ref, stateful):
        a, b = mx.array(a), mx.array(b)
        assert tuple(a.shape) == tuple(b.shape)
        assert float(mx.max(mx.abs(a - b))) == 0.0
