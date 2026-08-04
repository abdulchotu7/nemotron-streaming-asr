import mlx.core as mx
import mlx_audio.stt as stt
import sounddevice as sd
import numpy as np
from queue import Queue

model = stt.load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")

RATE = 16000
BLOCK = 320          # 20 ms

audio_queue = Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

buffer = []

with sd.InputStream(
    samplerate=RATE,
    channels=1,
    dtype="float32",
    blocksize=BLOCK,
    callback=callback,
):

    print("Speak for 5 seconds...")

    while len(buffer) < 250:
        buffer.append(audio_queue.get())

print("Finished recording")

audio = np.concatenate(buffer, axis=0)

audio = mx.array(audio.squeeze())

result = model.generate(audio, language="en-US")

print(result.text)