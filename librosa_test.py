import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


y, sr = librosa.load('bachWTC.mp3', sr=44100, duration=0.3)

print(f'sr:{sr}')

fig, ax = plt.subplots()

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)),
                            ref=np.max)

img = librosa.display.specshow(D, y_axis='log', sr=sr,
                         x_axis='time')

fig.colorbar(img)

plt.show()

