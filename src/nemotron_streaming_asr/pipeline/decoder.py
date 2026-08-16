"""Stateful greedy RNN-T decoder for prompted encoder chunks.

Persistent version of ``Model._decode_prompted_chunks`` for live microphone input:
the decoder RNN hidden state, last predicted token, growing hypothesis and global
frame counter live on the instance, so decoding state survives across ``feed()``
calls instead of being recreated per invocation.

The decoding math is identical to the original; only the storage location of the
persistent state changed. ``reset()`` clears all streaming state while keeping
the loaded model.
"""

import mlx.core as mx
from time import perf_counter_ns

from mlx_audio.stt.models.nemotron_asr import tokenizer as tok
from mlx_audio.stt.models.nemo.alignment import (
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)


class StreamingDecoder:
    """Stateful greedy RNN-T decoder for prompted encoder chunks.

    Owns the persistent decoding state (decoder RNN hidden state, last predicted
    token, growing hypothesis, global frame counter) and consumes prompted
    encoder chunks one at a time via ``feed()``, returning the cumulative
    :class:`AlignedResult` after each chunk.
    """

    def __init__(self, model, stats=None):
        self.model = model
        # Optional benchmark instrumentation. None or disabled -> zero impact.
        self._stats = stats
        self._bench = stats is not None and stats.enabled

        self.blank_id = model.blank_id
        self.decoder = model.decoder
        self.joint = model.joint
        self.max_symbols = model.max_symbols
        self.vocabulary = model.vocabulary

        self.frame_sec = (
            self.model.encoder_config.subsampling_factor
            * self.model.preprocessor_config.hop_length
            / self.model.preprocessor_config.sample_rate
        )

        self.reset()

    def reset(self):
        """Clear all streaming state while keeping the loaded model."""
        self.last_token = self.blank_id
        self.decoder_hidden = None
        self.hypothesis: list[AlignedToken] = []
        self.global_time = 0
        # First emitted language tag (e.g. "en-US") while streaming in "auto"
        # prompt mode; used by the session to switch the encoder prompt.
        self.detected_language = None

    def feed(self, prompted):
        """Greedy-decode one prompted encoder chunk (1, c, d).

        Returns the cumulative :class:`AlignedResult` after this chunk -- exactly
        what one yield of ``_decode_prompted_chunks`` would produce.

        When a benchmark ``stats`` object is attached and enabled, the whole
        call is timed (the per-frame ``int(mx.argmax(...))`` already
        synchronizes with the GPU) and ``mx.clear_cache()`` is timed
        independently.
        """
        t0 = perf_counter_ns() if self._bench else None
        chunk_len = prompted.shape[1]
        time = 0
        new_symbols = 0
        while time < chunk_len:
            feature = prompted[:, time : time + 1]
            current_token = (
                mx.array([[self.last_token]], dtype=mx.int32)
                if self.last_token != self.blank_id
                else None
            )
            decoder_output, (h, c) = self.decoder(current_token, self.decoder_hidden)
            decoder_output = decoder_output.astype(feature.dtype)
            proposed_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
            joint_output = self.joint(feature, decoder_output)
            pred_token = int(mx.argmax(joint_output))
            if pred_token != self.blank_id:
                self.last_token = pred_token
                self.decoder_hidden = proposed_hidden
                if self.detected_language is None and (
                    0 <= pred_token < len(self.vocabulary)
                    and tok.is_lang_tag(self.vocabulary[pred_token])
                ):
                    # Language-ID token (e.g. "<en-US>"), emitted in "auto"
                    # prompt mode; latch the first one for the session.
                    self.detected_language = self.vocabulary[pred_token][1:-1]
                if not tok.is_special_token(self.last_token, self.vocabulary):
                    self.hypothesis.append(
                        AlignedToken(
                            self.last_token,
                            start=(self.global_time + time) * self.frame_sec,
                            duration=self.frame_sec,
                            text=tok.decode([self.last_token], self.vocabulary),
                        )
                    )
                new_symbols += 1
                if self.max_symbols is not None and new_symbols >= self.max_symbols:
                    time += 1
                    new_symbols = 0
            else:
                time += 1
                new_symbols = 0
        self.global_time += chunk_len
        result = sentences_to_result(tokens_to_sentences(self.hypothesis))
        if t0 is not None:
            self._stats.record_decoder(perf_counter_ns() - t0)
            t_sync = perf_counter_ns()
            mx.clear_cache()
            self._stats.record_clear_cache(perf_counter_ns() - t_sync)
        else:
            mx.clear_cache()
        return result
