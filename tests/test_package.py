"""Public package API surface."""

import importlib


def test_package_imports():
    import nemotron_streaming_asr as pkg

    for name in ("NemotronStreamingSession", "StreamingAudioBuffer",
                 "StreamingEncoder", "StreamingDecoder",
                 "PerformanceStats", "StreamingBenchmark"):
        assert hasattr(pkg, name), name
    assert pkg.__version__


def test_submodules_import_cleanly():
    for module in (
        "nemotron_streaming_asr.pipeline.session",
        "nemotron_streaming_asr.pipeline.audio_buffer",
        "nemotron_streaming_asr.pipeline.encoder",
        "nemotron_streaming_asr.pipeline.decoder",
        "nemotron_streaming_asr.benchmark.stats",
        "nemotron_streaming_asr.benchmark.system",
        "nemotron_streaming_asr.benchmark.runner",
        "nemotron_streaming_asr.apps.mic",
    ):
        assert importlib.import_module(module) is not None, module


def test_reference_impl_preserved():
    """The stateless reference encoder must remain importable unchanged for
    regression testing (stream_encode_chunks / stream_encode)."""
    from nemotron_streaming_asr.pipeline import encoder as enc_mod

    assert callable(enc_mod.stream_encode_chunks)
    assert callable(enc_mod.stream_encode)
    assert callable(enc_mod._stream_block)
