import sounddevice as sd
from queue import Queue

from mlx_audio.stt import load
from streaming_session import NemotronStreamingSession


RATE = 16000
BLOCK = 320  # 20 ms

audio_queue = Queue()


def callback(indata, frames, time, status):
    if status:
        print(status)

    audio_queue.put(indata.copy())


def main():
    model = load("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")

    session = NemotronStreamingSession(
        model,
        language="en-US",
    )

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
            pcm = audio_queue.get()

            session.feed(pcm)

            for result in session.step():
                if result.text != last_text:
                    print(result.text)
                    last_text = result.text


if __name__ == "__main__":
    main()