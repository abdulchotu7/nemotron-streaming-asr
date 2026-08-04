"""Cache-aware streaming for the Nemotron FastConformer encoder.

Each conformer layer keeps an attention cache (last ``left_context`` attention-input
frames) and a causal-conv cache (last ``conv_kernel-1`` GLU-output frames);
subsampling is incremental with a small mel cache. With the window sized to the
allowed left context, no attention mask is needed, so the streamed encoder output is
frame-identical to the offline ``chunked_limited`` encoder at the native chunk size
(``right_context + 1``). This yields the model's native O(n), no-recompute streaming.
"""

import mlx.core as mx
import mlx.nn as nn
from time import perf_counter_ns

_PRE_ENCODE_MEL_CACHE = 16  # >= causal receptive field of the 8x dw-striding stack


def _stream_block(block, x, pos_enc, attn_cache, conv_cache, left_cache, conv_left):
    # half-step FFN 1
    residual = x + 0.5 * block.feed_forward1(block.norm_feed_forward1(x))

    # cache-aware self-attention: Q = chunk, K/V = [cache ++ chunk]
    xn = block.norm_self_att(residual)
    kv = xn if attn_cache is None else mx.concatenate([attn_cache, xn], axis=1)
    pos_emb = pos_enc.pos_emb_for(kv.shape[1], x.dtype)
    residual = residual + block.self_attn.stream(xn, kv, pos_emb)
    attn_next = kv[:, -left_cache:] if left_cache > 0 else kv[:, :0]

    # cache-aware causal conv: prepend conv cache instead of zero-padding
    xc = block.norm_conv(residual)
    g = nn.glu(block.conv.pointwise_conv1(xc), axis=-1)  # (B, c, d)
    if conv_cache is None:
        conv_cache = mx.zeros((g.shape[0], conv_left, g.shape[2]), dtype=g.dtype)
    din = mx.concatenate([conv_cache, g], axis=1)
    dw = block.conv.depthwise_conv(din)  # valid conv -> (B, c, d)
    conv_next = din[:, -conv_left:]
    y = block.conv.batch_norm(dw)
    y = block.conv.activation(y)
    residual = residual + block.conv.pointwise_conv2(y)

    # half-step FFN 2 + final norm
    residual = residual + 0.5 * block.feed_forward2(block.norm_feed_forward2(residual))
    return block.norm_out(residual), attn_next, conv_next


def stream_encode_chunks(
    model, mel_chunks, language, chunk_frames=None, att_context_size=None
):
    """Yield post-prompt encoder frames from one or more mel chunks.

    The encoder/conv/subsampling caches persist across input mel chunks, so callers
    can keep STFT memory bounded without resetting model context at chunk boundaries.
    """
    enc = model.encoder
    acs = att_context_size or model.default_att_context_size
    left_cache = int(acs[0])
    right = int(acs[1])
    cf = chunk_frames or (right + 1)
    sf = enc.args.subsampling_factor
    chunk_mel = cf * sf
    conv_left = enc.args.conv_kernel_size - 1

    n = len(enc.layers)
    attn_cache = [None] * n
    conv_cache = [None] * n
    mel_cache = None
    emitted = 0
    consumed = 0
    pending = None

    def append_pending(chunk):
        nonlocal pending
        if chunk.ndim == 2:
            chunk = mx.expand_dims(chunk, 0)
        if chunk.shape[1] == 0:
            return
        pending = chunk if pending is None else mx.concatenate([pending, chunk], axis=1)

    def encode_mel_chunk(m, is_final):
        nonlocal mel_cache, emitted, consumed
        cache_len = 0 if mel_cache is None else mel_cache.shape[1]
        win = m if mel_cache is None else mx.concatenate([mel_cache, m], axis=1)
        win_len = win.shape[1]
        sub = enc.pre_encode(win, mx.array([win_len], dtype=mx.int32))[0]  # (1, k, d)

        end = consumed + m.shape[1]
        base = (consumed - cache_len) // sf
        lo = emitted - base
        hi = sub.shape[1] if is_final else (end // sf - base)
        consumed = end
        mel_cache = win[:, -_PRE_ENCODE_MEL_CACHE:]

        if hi <= lo:
            emitted = base + max(lo, hi)
            return
        emitted = base + hi
        h = sub[:, lo:hi]
        for li, block in enumerate(enc.layers):
            h, attn_cache[li], conv_cache[li] = _stream_block(
                block,
                h,
                enc.pos_enc,
                attn_cache[li],
                conv_cache[li],
                left_cache,
                conv_left,
            )
        yield model.apply_prompt(h, language)

    def encode_ready(is_final):
        nonlocal pending
        while pending is not None and pending.shape[1] > 0:
            if pending.shape[1] < chunk_mel and not is_final:
                break

            take = min(chunk_mel, pending.shape[1])
            if is_final and pending.shape[1] <= chunk_mel:
                take = pending.shape[1]

            m = pending[:, :take]
            pending = pending[:, take:]
            is_final_chunk = is_final and pending.shape[1] == 0
            yield from encode_mel_chunk(m, is_final_chunk)

    iterator = iter(mel_chunks)
    try:
        current = next(iterator)
    except StopIteration:
        return

    for next_chunk in iterator:
        append_pending(current)
        yield from encode_ready(is_final=False)
        current = next_chunk

    append_pending(current)
    yield from encode_ready(is_final=True)


def stream_encode(model, mel, language, chunk_frames=None, att_context_size=None):
    """Yield post-prompt encoder frames (1, c, d) per chunk, cache-aware.

    Frame-identical to ``encoder(...)`` + ``apply_prompt(...)`` at the native chunk
    size (right_context + 1).
    """
    yield from stream_encode_chunks(
        model,
        [mel],
        language,
        chunk_frames=chunk_frames,
        att_context_size=att_context_size,
    )


class StreamingEncoder:
    """Stateful cache-aware streaming encoder for live microphone input.

    Persistent version of :func:`stream_encode_chunks`: the per-layer attention
    caches, conv caches, mel cache and the emitted/consumed/pending bookkeeping
    live on the instance, so streaming state survives across ``feed()`` calls
    instead of being recreated per call.

    The encoder math, attention, convolution, subsampling and prompting are
    identical to the stateless reference above; only the storage location of the
    persistent state changed. ``feed()`` replaces the mel-chunk iterator and
    ``finish()`` performs the final flush of the trailing partial chunk.
    ``reset()`` clears all streaming state while keeping the loaded model.

    ``language`` is deliberately not stored here: it is owned by the session and
    passed to every ``feed()``/``finish()`` call, matching the reference
    ``stream_encode_chunks(model, mel_chunks, language, ...)`` signature.
    """

    def __init__(self, model, chunk_frames=None, att_context_size=None, stats=None):
        self.model = model
        # Optional benchmark instrumentation. None or disabled -> zero impact.
        self._stats = stats
        self._bench = stats is not None and stats.enabled

        enc = model.encoder
        acs = att_context_size or model.default_att_context_size
        self.left_cache = int(acs[0])
        self.right = int(acs[1])
        self.cf = chunk_frames or (self.right + 1)
        self.sf = enc.args.subsampling_factor
        self.chunk_mel = self.cf * self.sf
        self.conv_left = enc.args.conv_kernel_size - 1

        self.reset()

    def reset(self):
        """Clear all streaming state while keeping the loaded model."""
        n = len(self.model.encoder.layers)
        self.attn_cache = [None] * n
        self.conv_cache = [None] * n
        self.mel_cache = None
        self.emitted = 0
        self.consumed = 0
        self.pending = None

    def _append_pending(self, chunk):
        if chunk.ndim == 2:
            chunk = mx.expand_dims(chunk, 0)
        if chunk.shape[1] == 0:
            return
        self.pending = chunk if self.pending is None else mx.concatenate([self.pending, chunk], axis=1)

    def _encode_mel_chunk(self, m, is_final, language):
        cache_len = 0 if self.mel_cache is None else self.mel_cache.shape[1]
        win = m if self.mel_cache is None else mx.concatenate([self.mel_cache, m], axis=1)
        win_len = win.shape[1]
        sub = self.model.encoder.pre_encode(win, mx.array([win_len], dtype=mx.int32))[0]  # (1, k, d)

        end = self.consumed + m.shape[1]
        base = (self.consumed - cache_len) // self.sf
        lo = self.emitted - base
        hi = sub.shape[1] if is_final else (end // self.sf - base)
        self.consumed = end
        self.mel_cache = win[:, -_PRE_ENCODE_MEL_CACHE:]

        if hi <= lo:
            self.emitted = base + max(lo, hi)
            return
        self.emitted = base + hi
        h = sub[:, lo:hi]
        for li, block in enumerate(self.model.encoder.layers):
            h, self.attn_cache[li], self.conv_cache[li] = _stream_block(
                block,
                h,
                self.model.encoder.pos_enc,
                self.attn_cache[li],
                self.conv_cache[li],
                self.left_cache,
                self.conv_left,
            )
        yield self.model.apply_prompt(h, language)

    def _encode_ready(self, is_final, language):
        while self.pending is not None and self.pending.shape[1] > 0:
            if self.pending.shape[1] < self.chunk_mel and not is_final:
                break

            take = min(self.chunk_mel, self.pending.shape[1])
            if is_final and self.pending.shape[1] <= self.chunk_mel:
                take = self.pending.shape[1]

            m = self.pending[:, :take]
            self.pending = self.pending[:, take:]
            is_final_chunk = is_final and self.pending.shape[1] == 0
            yield from self._encode_mel_chunk(m, is_final_chunk, language)

    def feed(self, mel_chunk, language):
        """Encode one incoming mel chunk (1, chunk_mel, features).

        Appends the chunk to ``pending`` and emits every now-complete native
        encoder chunk, yielding zero or more prompted encoder frames (1, c, d) --
        the same output the stateless :func:`stream_encode_chunks` yields for the
        same chunk sequence. ``language`` is the prompt key (e.g. ``"en-US"``).

        When a benchmark ``stats`` object is attached and enabled, each yielded
        chunk is synchronized with ``mx.eval`` so the recorded time reflects
        actual encoder execution (not lazy dispatch); the eval duration is
        recorded independently.
        """
        self._append_pending(mel_chunk)
        t0 = perf_counter_ns() if self._bench else None
        for prompted in self._encode_ready(is_final=False, language=language):
            if t0 is not None:
                t_sync = perf_counter_ns()
                mx.eval(prompted)
                t1 = perf_counter_ns()
                self._stats.record_eval(t1 - t_sync)
                self._stats.record_encoder(t1 - t0)
                t0 = perf_counter_ns()
            yield prompted

    def finish(self, language):
        """Flush the trailing partial chunk (end-of-utterance final flush)."""
        t0 = perf_counter_ns() if self._bench else None
        for prompted in self._encode_ready(is_final=True, language=language):
            if t0 is not None:
                t_sync = perf_counter_ns()
                mx.eval(prompted)
                t1 = perf_counter_ns()
                self._stats.record_eval(t1 - t_sync)
                self._stats.record_encoder(t1 - t0)
                t0 = perf_counter_ns()
            yield prompted
