import wave
import sounddevice as sd
import soundfile as sf

import matplotlib.pyplot as plt


with open("Untitled1.bin", "rb") as f:
    data = f.read()
    with wave.open("sound.wav", "wb") as out_f:
        out_f.setnchannels(1)
        out_f.setsampwidth(2) # number of bytes
        out_f.setframerate(2000)
        out_f.writeframesraw(data)


# filename = 'Untitled.bin'
# data, fs = sf.read(filename, samplerate=10000, format='RAW', channels=1, subtype='PCM_16')
# sd.play(data, fs)

# import simpleaudio as sa
# play_obj = sa.play_buffer(data, 1, 2, 10000)

# plt.plot(data)
# plt.show()
