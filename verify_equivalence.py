"""Regression suite for the stateful Nemotron streaming pipeline.

Reference (ground truth) is the original stateless implementation:
  * ``mlx_audio...nemotron_asr.streaming.stream_encode_chunks``
  * ``Model._decode_prompted_chunks``

Coverage:
  A. encoder.feed()/finish()  == stream_encode_chunks()         (frame-identical)
  B. decoder.feed()           == _decode_prompted_chunks()      (token-identical)
  C. buffer mel chunks        == offline log_mel over full wav  (bit-identical)
  D. long live session        == reference over same mels       (token-identical)
  E. finish() final flush     == reference final flush
  F. reset()                  == completely fresh session
  G. memory bounded during a long microphone-style session
  H. real 0.6B model on a WAV == stream_generate()
"""

import sys
import numpy as np
import mlx.core as mx

sys.path.insert(0, "/Users/abdulrahim/Downloads/nemotron")

from mlx_audio.stt.models.nemotron_asr import Model, ModelConfig
from mlx_audio.stt.models.nemotron_asr import streaming as ref_streaming
from mlx_audio.stt.models.nemotron_asr.audio import (
    iter_log_mel_spectrogram,
)

from streaming_encoder import StreamingEncoder
from streaming_decoder import StreamingDecoder
from streaming_session import NemotronStreamingSession, StreamingAudioBuffer

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def _tiny_config() -> dict:
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


def _build_tiny() -> Model:
    model = Model(ModelConfig.from_dict(_tiny_config()))
    mx.eval(model.parameters())
    model.eval()
    return model


def _prompted(enc: StreamingEncoder, mel_chunks, language="en-US"):
    out = []
    for m in mel_chunks:
        out.extend(list(enc.feed(m, language)))
    out.extend(list(enc.finish(language)))
    return out


def _token_ids(result):
    return [t.id for s in result.sentences for t in s.tokens]


# --------------------------------------------------------------------------
def test_encoder_equivalence(model, mel_chunks):
    # Reference: stateless iterator over the whole sequence (includes the
    # final is_final=True flush of the partial trailing chunk).
    ref = list(
        ref_streaming.stream_encode_chunks(model, iter(mel_chunks), "en-US")
    )

    # Stateful: one feed() per chunk, then finish() for the final flush.
    enc = StreamingEncoder(model)
    st = _prompted(enc, mel_chunks, "en-US")

    check(
        "A: encoder chunk count matches reference",
        len(st) == len(ref),
        f"ref={len(ref)} stateful={len(st)}",
    )
    for i, (a, b) in enumerate(zip(ref, st)):
        a = mx.array(a)
        b = mx.array(b)
        check(
            f"A: prompted chunk {i} frame-identical",
            a.shape == b.shape and float(mx.max(mx.abs(a - b))) == 0.0,
            f"shape {tuple(a.shape)} vs {tuple(b.shape)}",
        )

    # Open-ended stream: only non-final full chunks (live mic semantics).
    native = [m for m in mel_chunks if m.shape[1] == 112]
    empty = mx.zeros((1, 0, native[0].shape[2]), dtype=native[0].dtype)
    ref_open = list(
        ref_streaming.stream_encode_chunks(model, iter(native + [empty]), "en-US")
    )
    enc2 = StreamingEncoder(model)
    st_open = []
    for m in native:
        st_open.extend(list(enc2.feed(m, "en-US")))
    st_open.extend(list(enc2.finish("en-US")))
    check(
        "A: open-stream feed() matches reference",
        len(st_open) == len(ref_open)
        and all(
            tuple(a.shape) == tuple(b.shape)
            and float(mx.max(mx.abs(mx.array(a) - mx.array(b)))) == 0.0
            for a, b in zip(st_open, ref_open)
        ),
        f"ref={len(ref_open)} stateful={len(st_open)}",
    )


def test_decoder_equivalence(model, prompted_chunks):
    ref = list(model._decode_prompted_chunks(iter(prompted_chunks)))
    dec = StreamingDecoder(model)
    st = [dec.feed(p) for p in prompted_chunks]
    check(
        "B: decoder result count matches reference",
        len(st) == len(ref),
        f"ref={len(ref)} stateful={len(st)}",
    )
    for i, (a, b) in enumerate(zip(ref, st)):
        check(
            f"B: decode result {i} identical",
            a.text == b.text and _token_ids(a) == _token_ids(b),
            f"tokens={_token_ids(a)}",
        )


def test_buffer_mel_bit_identical(model, audio_np):
    # Reference: offline chunking over the full waveform.
    ref_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(audio_np), model.preprocessor_config, chunk_frames=112
        )
    )
    ref_full = [c for c in ref_chunks if c.shape[1] == 112]

    # Live: feed 20 ms blocks, step after every feed.
    buf = StreamingAudioBuffer(model)
    emitted = []
    for i in range(0, audio_np.shape[0], 320):
        buf.feed(audio_np[i : i + 320])
        emitted.extend(list(buf.get_ready_mel_chunks()))

    check(
        "C: buffer emitted the same number of full chunks",
        len(emitted) == len(ref_full),
        f"ref={len(ref_full)} buffer={len(emitted)}",
    )
    for k, (a, b) in enumerate(zip(emitted, ref_full)):
        diff = float(mx.max(mx.abs(mx.array(a) - mx.array(b))))
        check(f"C: mel chunk {k} bit-identical to offline", diff == 0.0, f"max|diff|={diff}")

    # Trimming must be transparent: same chunks with trimming disabled.
    buf2 = StreamingAudioBuffer(model)
    buf2._can_trim = False
    emitted2 = []
    for i in range(0, audio_np.shape[0], 320):
        buf2.feed(audio_np[i : i + 320])
        emitted2.extend(list(buf2.get_ready_mel_chunks()))
    check(
        "C: trimming is transparent (no-trim == trim)",
        len(emitted2) == len(emitted)
        and all(
            float(mx.max(mx.abs(mx.array(a) - mx.array(b)))) == 0.0
            for a, b in zip(emitted2, emitted)
        ),
        f"n={len(emitted2)}",
    )


def _run_live(model, audio_np, block=320):
    """Feed in blocks, step after each; return (results, buffer, max_held)."""
    session = NemotronStreamingSession(model, language="en-US")
    results = []
    max_held = 0
    for i in range(0, audio_np.shape[0], block):
        session.feed(audio_np[i : i + block])
        for r in session.step():
            results.append(r)
        max_held = max(max_held, session.audio._length)
    return session, results, max_held


def test_long_session_token_identity(model, audio_np):
    # Long (~40 s) live microphone-style session.
    session, results, max_held = _run_live(model, audio_np)
    tail_results = list(session.finish())

    check(
        "D: long session processed all audio",
        session.audio.total_samples == audio_np.shape[0],
        f"fed={audio_np.shape[0]} processed={session.audio.total_samples}",
    )

    # Reference over the full offline mel sequence (includes the final
    # is_final=True flush of the trailing partial chunk).
    iter_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(audio_np), model.preprocessor_config, chunk_frames=112
        )
    )
    ref = list(
        model._decode_prompted_chunks(
            ref_streaming.stream_encode_chunks(model, iter(iter_chunks), "en-US")
        )
    )

    all_results = results + tail_results
    check(
        "D: long session (step+finish) result count matches reference",
        len(all_results) == len(ref),
        f"ref={len(ref)} session={len(all_results)} (step {len(results)}, finish {len(tail_results)})",
    )
    for i, (a, b) in enumerate(zip(ref, all_results)):
        check(
            f"D: result {i} token-identical",
            _token_ids(a) == _token_ids(b),
            f"ref={a.text!r} session={b.text!r}",
        )
    return max_held


def test_finish_final_flush(model, mel_chunks):
    # finish() must reproduce the reference's final is_final=True flush of the
    # trailing partial chunk: feed everything non-final, then finish().
    enc = StreamingEncoder(model)
    st = _prompted(enc, mel_chunks, "en-US")
    ref = list(
        ref_streaming.stream_encode_chunks(model, iter(mel_chunks), "en-US")
    )
    check(
        "E: finish() final flush matches reference",
        len(st) == len(ref)
        and all(
            tuple(a.shape) == tuple(b.shape)
            and float(mx.max(mx.abs(mx.array(a) - mx.array(b)))) == 0.0
            for a, b in zip(st, ref)
        ),
        f"ref={len(ref)} stateful={len(st)}",
    )


def test_reset(model, audio_a, audio_b):
    session = NemotronStreamingSession(model, language="en-US")

    _, r1, _ = _run_live(model, audio_a)          # baseline on audio_a
    session.reset()
    _, r1b, _ = _run_live_on(session, audio_a)    # same session, audio_a again
    check(
        "F: reset() then same audio == fresh session (a)",
        [ _token_ids(r) for r in r1 ] == [ _token_ids(r) for r in r1b ],
        f"n={len(r1)}/{len(r1b)}",
    )

    session.reset()
    _, r2, _ = _run_live_on(session, audio_b)     # switch utterance after reset
    fresh = NemotronStreamingSession(model, language="en-US")
    _, rf, _ = _run_live_on(fresh, audio_b)
    check(
        "F: reset() then new audio == brand-new session",
        [ _token_ids(r) for r in r2 ] == [ _token_ids(r) for r in rf ],
        f"n={len(r2)}/{len(rf)}",
    )

    # Direct component resets.
    enc = StreamingEncoder(model)
    dec = StreamingDecoder(model)
    mel_chunks = list(
        iter_log_mel_spectrogram(mx.array(audio_a), model.preprocessor_config, chunk_frames=112)
    )
    enc.reset(); dec.reset()
    pa = _prompted(enc, mel_chunks[:2], "en-US")
    enc.reset()
    pb = _prompted(enc, mel_chunks[:2], "en-US")
    check(
        "F: encoder reset() restarts cleanly",
        all(
            tuple(a.shape) == tuple(b.shape)
            and float(mx.max(mx.abs(mx.array(a) - mx.array(b)))) == 0.0
            for a, b in zip(pa, pb)
        ),
    )
    dec.reset()
    da = [dec.feed(p) for p in pa]
    dec.reset()
    db = [dec.feed(p) for p in pa]
    check(
        "F: decoder reset() restarts cleanly",
        [_token_ids(r) for r in da] == [_token_ids(r) for r in db],
    )


def _run_live_on(session, audio_np, block=320):
    results = []
    for i in range(0, audio_np.shape[0], block):
        session.feed(audio_np[i : i + block])
        for r in session.step():
            results.append(r)
    return session, results, 0


def test_stats_zero_impact(model, audio_np):
    """Benchmark instrumentation must not change recognition results.

    stats=None, stats(enabled=False) and stats(enabled=True) must all produce
    identical token sequences; enabled stats must collect sensible samples.
    """
    from benchmark import PerformanceStats

    def run(stats):
        session = NemotronStreamingSession(model, language="en-US", stats=stats)
        res = []
        for i in range(0, audio_np.shape[0], 320):
            session.feed(audio_np[i : i + 320])
            for r in session.step():
                res.append(r)
        res.extend(session.finish())
        return [_token_ids(r) for r in res]

    base = run(None)
    check("I: stats=None baseline produced results", len(base) > 0, f"n={len(base)}")

    check(
        "I: stats(enabled=False) identical to stats=None",
        run(PerformanceStats(enabled=False)) == base,
    )

    enabled = PerformanceStats(enabled=True)
    check(
        "I: stats(enabled=True) identical to stats=None",
        run(enabled) == base,
    )

    for key in ("audio_feed", "mel", "encoder", "decoder", "step",
                "end_to_end", "token_latency", "eval", "clear_cache"):
        check(
            f"I: stats collected {key} samples",
            enabled.count(key) > 0,
            f"n={enabled.count(key)}",
        )
    check(
        "I: end-to-end latencies positive",
        all(v > 0 for v in enabled.samples("end_to_end")),
    )
    check(
        "I: eval and clear_cache measured",
        enabled.count("eval") > 0 and enabled.count("clear_cache") > 0,
    )
    check(
        "I: tokens/words counted",
        enabled.count("tokens") > 0 and enabled.count("words") >= 0,
        f"tokens={enabled.count('tokens')} words={enabled.count('words')}",
    )


def test_real_model():
    from mlx_audio.stt import load
    from mlx_audio.stt.utils import load_audio

    real = load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
    real.eval()

    wav = load_audio(
        "/Users/abdulrahim/Downloads/nemotron/linus-original-demo_4bucvKgI.wav",
        real.preprocessor_config.sample_rate,
        dtype=mx.float32,
    )
    audio_np = np.array(wav, dtype=np.float32)

    ref_results = list(real.stream_generate(wav, language="en-US"))
    ref_final = ref_results[-1]

    session = NemotronStreamingSession(real, language="en-US")
    st = []
    for i in range(0, audio_np.shape[0], 320):
        session.feed(audio_np[i : i + 320])
        for r in session.step():
            st.append(r)
    for r in session.finish():
        st.append(r)

    check(
        "H: real model final text identical",
        st[-1].text == ref_final.text,
        f"ref={ref_final.text!r} stateful={st[-1].text!r}",
    )
    check(
        "H: real model per-chunk token sequences identical",
        [_token_ids(r) for r in ref_results] == [_token_ids(r) for r in st],
        f"ref_chunks={len(ref_results)} stateful_chunks={len(st)}",
    )


def main():
    np.random.seed(0)
    mx.random.seed(0)

    model = _build_tiny()
    sr = model.preprocessor_config.sample_rate

    # ~2.5 s: 251 mel frames -> [112, 112, 27] (partial tail).
    audio_short = (np.random.randn(int(2.5 * sr)) * 0.1).astype(np.float32)
    mel_chunks = list(
        iter_log_mel_spectrogram(
            mx.array(audio_short), model.preprocessor_config, chunk_frames=112
        )
    )
    print("mel chunks:", [tuple(c.shape) for c in mel_chunks])

    test_encoder_equivalence(model, mel_chunks)

    prompted = list(
        ref_streaming.stream_encode_chunks(model, iter(mel_chunks), "en-US")
    )
    test_decoder_equivalence(model, prompted)

    test_buffer_mel_bit_identical(model, audio_short)

    # ~40 s live session (tiny model).
    audio_long = (np.random.randn(int(40 * sr)) * 0.05).astype(np.float32)
    max_held = test_long_session_token_identity(model, audio_long)

    bound = 20000  # ~1.25 s of audio at 16 kHz
    check(
        "G: memory bounded during long session",
        max_held < bound,
        f"max held samples={max_held} (bound {bound})",
    )
    check(
        "G: retained history << total audio",
        max_held < audio_long.shape[0] / 10,
        f"held={max_held} total={audio_long.shape[0]}",
    )

    test_finish_final_flush(model, mel_chunks)

    audio_b = (np.random.randn(int(3 * sr)) * 0.08).astype(np.float32)
    test_reset(model, audio_short, audio_b)

    test_stats_zero_impact(model, audio_short)

    test_real_model()

    print()
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {FAILED}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
