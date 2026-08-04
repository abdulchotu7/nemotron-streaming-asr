import mlx.core as mx
import mlx_audio.stt as stt
import sounddevice as sd
import numpy as np
from queue import Queue
import time

model = stt.load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
print(model.preprocessor_config.hop_length)
print(model.preprocessor_config.n_fft)
print(model.preprocessor_config.sample_rate)
print(model.default_att_context_size)
print(model.encoder.args.subsampling_factor)

RATE = 16000
BLOCK = 320          # 20 ms

audio_queue = Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


buffer = []

last_run = time.time()

last_text = ""

with sd.InputStream(
    samplerate=RATE,
    channels=1,
    dtype="float32",
    blocksize=BLOCK,
    callback=callback,
):
    print("Listening... Ctrl+C to stop")

    while True:

        chunk = audio_queue.get()

        buffer.append(chunk)

        if time.time() - last_run > 0.5:

            audio = np.concatenate(buffer, axis=0)
            start = time.perf_counter()


            for result in model.stream_generate(
                mx.array(audio.squeeze()),
                language="en-US",
            ):
                pass
            elapsed = time.perf_counter() - start

            print(f"{elapsed:.3f}s")

            if result.text != last_text:
                print(result.text)
                last_text = result.text
            last_run = time.time()