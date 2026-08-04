from mlx_audio.stt import load
import sounddevice as sd

model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
samplerate = 16000
channels = 1

# auto language detection (default)
# print(model.generate("/Users/abdulrahim/Downloads/nemotron/linus-original-demo_4bucvKgI.wav").text)
# result = model.generate("/Users/abdulrahim/Downloads/nemotron/linus-original-demo_4bucvKgI.mp3").text


for r in model.stream_generate("/Users/abdulrahim/Downloads/nemotron/linus-original-demo_4bucvKgI.wav", language="en-US"):
    print(r.text)

stream = sd.InputStream(
    samplerate=samplerate,
    channels=channels,
    dtype="float32",
    blocksize=320,   # 20 ms
)

stream.start()

print(type(model))
print(model.__class__)