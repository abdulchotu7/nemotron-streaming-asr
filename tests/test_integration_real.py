"""Integration test against the real HuggingFace model and the sample WAV.

Skipped automatically when the model cannot be loaded or ``data/`` has no WAV.
"""

import numpy as np
import pytest

from nemotron_streaming_asr import NemotronStreamingSession

from .conftest import BLOCK, DATA_DIR, WAV_PATH


def _load_real_model():
    from mlx_audio.stt import load

    return load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")


@pytest.mark.integration
def test_real_model_matches_stream_generate():
    if not WAV_PATH.exists():
        pytest.skip(f"sample WAV not found at {WAV_PATH}")
    try:
        model = _load_real_model()
    except Exception as e:  # offline / missing cache
        pytest.skip(f"real model unavailable: {e!r}")
    model.eval()

    from mlx_audio.stt.utils import load_audio
    import mlx.core as mx

    wav = load_audio(str(WAV_PATH), model.preprocessor_config.sample_rate, dtype=mx.float32)
    audio = np.array(wav, dtype=np.float32)

    ref_results = list(model.stream_generate(wav, language="en-US"))

    session = NemotronStreamingSession(model, language="en-US")
    results = []
    for i in range(0, audio.shape[0], BLOCK):
        session.feed(audio[i : i + BLOCK])
        for r in session.step():
            results.append(r)
    results.extend(session.finish())

    assert len(results) == len(ref_results), (len(results), len(ref_results))
    for r, expected in zip(results, ref_results):
        ids = [t.id for s in r.sentences for t in s.tokens]
        ref_ids = [t.id for s in expected.sentences for t in s.tokens]
        assert ids == ref_ids
    assert results[-1].text == ref_results[-1].text


@pytest.mark.integration
def test_sample_audio_present():
    """The benchmark defaults to this WAV; guard against silent fallback."""
    assert DATA_DIR.exists()
    assert WAV_PATH.exists(), "run: cp linus-original-demo_4bucvKgI.wav data/"
