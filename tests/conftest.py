"""Shared fixtures and helpers for the nemotron_streaming_asr test suite.

The suite uses a tiny randomly-initialized model (no weights download) for the
algorithmic equivalence tests, plus an optional integration test against the
real HuggingFace model and the sample WAV in ``data/``.
"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WAV_PATH = DATA_DIR / "linus-original-demo_4bucvKgI.wav"

BLOCK = 320  # 20 ms at 16 kHz, matching apps/mic.py


def tiny_config() -> dict:
    """Config for a tiny randomly-initialized streaming Nemotron model."""
    vocab = ["<unk>", "<en-US>", "▁hello", "▁world", "!", "a", "b", "c"]
    return {
        "model_type": "nemotron_asr",
        "preprocessor": {"features": 80, "n_fft": 512, "normalize": "NA"},
        "encoder": {
            "feat_in": 80,
            "n_layers": 2,
            "d_model": 32,
            "n_heads": 2,
            "ff_expansion_factor": 2,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 8,
            "conv_kernel_size": 9,
            "causal_downsampling": True,
            "conv_context_size": "causal",
            "conv_norm_type": "layer_norm",
            "att_context_style": "chunked_limited",
            "att_context_size": [[56, 13]],
            "pos_emb_max_len": 500,
            "use_bias": False,
        },
        "prompt": {
            "num_prompts": 4,
            "prompt_hidden": 16,
            "prompt_dictionary": {"en-US": 0, "auto": 1},
        },
        "decoder": {
            "pred_hidden": 16,
            "pred_rnn_layers": 2,
            "vocab_size": len(vocab),
            "blank_as_pad": True,
        },
        "joint": {
            "joint_hidden": 16,
            "activation": "relu",
            "encoder_hidden": 32,
            "pred_hidden": 16,
            "num_classes": len(vocab),
        },
        "vocabulary": vocab,
        "default_language": "auto",
        "default_att_context_size": [56, 13],
        "max_symbols": 5,
    }


@pytest.fixture(scope="session")
def tiny_model():
    """A tiny randomly-initialized model; enough to exercise the exact
    encoder/decoder math without downloading real weights.

    The RNG is seeded before construction so the model (and therefore the
    decoded tokens it produces on ``seeded_audio``) is deterministic.
    """
    from mlx_audio.stt.models.nemotron_asr import Model, ModelConfig

    mx.random.seed(0)
    np.random.seed(0)
    model = Model(ModelConfig.from_dict(tiny_config()))
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture(scope="session")
def seeded_audio():
    """Deterministic pseudo-speech waveform (float32, 1-D)."""
    np.random.seed(0)
    sr = 16000
    return (np.random.randn(int(2.5 * sr)) * 0.1).astype(np.float32)


def token_ids(result):
    """Flat list of decoded token ids for an AlignedResult."""
    return [t.id for s in result.sentences for t in s.tokens]


def run_live(session, audio, block=BLOCK):
    """Feed ``audio`` in blocks, step after every feed, then flush the tail.

    Returns the list of cumulative AlignedResult objects.
    """
    results = []
    for i in range(0, audio.shape[0], block):
        session.feed(audio[i : i + block])
        for r in session.step():
            results.append(r)
    results.extend(session.finish())
    return results
