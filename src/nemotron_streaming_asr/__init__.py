"""nemotron_streaming_asr: real-time offline streaming ASR on Apple Silicon.

A stateful, cache-aware streaming implementation of NVIDIA's Nemotron 3.5
Streaming ASR (MLX) for live microphone input, with a production benchmarking
framework. The recognition math is identical to the reference implementation;
the pipeline stores streaming state across microphone chunks instead.

Typical use::

    from mlx_audio.stt import load
    from nemotron_streaming_asr import NemotronStreamingSession

    model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
    session = NemotronStreamingSession(model, language="en-US")
    session.feed(pcm)                       # raw mono float32 PCM
    for result in session.step():           # one cumulative AlignedResult per chunk
        print(result.text)
"""

from .pipeline.audio_buffer import StreamingAudioBuffer
from .pipeline.decoder import StreamingDecoder
from .pipeline.encoder import StreamingEncoder
from .pipeline.session import NemotronStreamingSession
from .benchmark.stats import PerformanceStats
from .benchmark.runner import StreamingBenchmark

__version__ = "0.1.0"

__all__ = [
    "NemotronStreamingSession",
    "StreamingAudioBuffer",
    "StreamingEncoder",
    "StreamingDecoder",
    "PerformanceStats",
    "StreamingBenchmark",
]
