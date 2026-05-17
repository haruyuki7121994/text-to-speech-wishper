# This is a sample Python script.
from lightning_whisper_mlx import LightningWhisperMLX


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi():
    # Use a breakpoint in the code line below to debug your script.
    whisper = LightningWhisperMLX(model="distil-medium.en", batch_size=12, quant=None)

    text = whisper.transcribe(audio_path="/audio.mp3")['text']

    print(text)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
