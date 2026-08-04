import numpy as np
import mlx.core as mx

from mlx_audio.stt.models.nemotron_asr.audio import (
    log_mel_spectrogram_frames,
)


class NemotronStreamingSession:
    def __init__(self, model, language="en-US"):
        self.model = model
        self.language = language

        self.audio = StreamingAudioBuffer(model)

    def feed(self, pcm):
        self.audio.feed(pcm)

    def step(self):
        """
        Temporary implementation.

        For now we only verify that our audio frontend emits
        correctly-sized mel chunks.

        Next step:
            mel -> stream_encode_chunks() -> _decode_prompted_chunks()
        """
        for mel in self.audio.get_ready_mel_chunks():
            print(f"Mel chunk emitted: {mel.shape}")


class StreamingAudioBuffer:
    def __init__(self, model):
        self.config = model.preprocessor_config

        right_context = model.default_att_context_size[1]
        subsampling = model.encoder.args.subsampling_factor

        # Native chunk size expected by stream_encode_chunks()
        self.chunk_mel = (right_context + 1) * subsampling

        print(f"Chunk mel frames : {self.chunk_mel}")

        # Raw PCM waveform history
        self.waveform = np.empty(0, dtype=np.float32)

        # First mel frame that has not yet been emitted
        self.next_mel_frame = 0

    def feed(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)

        self.waveform = np.concatenate(
            [self.waveform, pcm]
        )

    @property
    def available_frames(self):
        return self.waveform.shape[0] // self.config.hop_length + 1

    def get_ready_mel_chunks(self):

        print("--------------------------------------")
        print(f"Waveform samples : {self.waveform.shape[0]}")
        print(f"Available frames : {self.available_frames}")
        print(f"Next mel frame   : {self.next_mel_frame}")

        waveform = mx.array(self.waveform)

        while (
            self.available_frames - self.next_mel_frame
            >= self.chunk_mel
        ):

            start = self.next_mel_frame
            end = start + self.chunk_mel

            mel = log_mel_spectrogram_frames(
                waveform,
                self.config,
                start,
                end,
            )

            print(f"Emit mel frames {start} -> {end}")
            print(f"Mel shape: {mel.shape}")

            self.next_mel_frame = end

            yield mel


class StreamingEncoder:
    def __init__(self, model):
        self.model = model
        self.attn_cache = None
        self.conv_cache = None
        self.mel_cache = None
        self.pending = None
        self.emitted = None
        self.consumed = None

    def encode(self, mel):
        """
        Encode a chunk of mel frames into a chunk of encoder states.

        This is the first step in the streaming ASR pipeline.
        """
        pass