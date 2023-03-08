from main import *
import pandas as pd
import csv


data = pd.read_csv("freqs.csv")
l = 150
times = data.columns.to_numpy()[:l]
frequency = data.to_numpy()[0][:l]

# print(frequency)


notes_frequecies = [440,659.25, 739.9,  783.9, 830.6, 880, 932.3, 987.8]
notes_names = ["A4", "F5", "Gb5", "G5", "E5", "Ab5", "A5", "Bb5", "B5"]

notes_names = ["A4", "Bb4", "B4", "C5", "Db5", "D5", "Eb5", "E5", "F5", "Gb5", "G5", "Ab5", "A5", "Bb5", "B5", "C5"]
notes_frequecies = [440, 466.16, 493.9, 523.3, 554.4, 587.3, 622.3, 659.3, 698.5, 740, 784, 830.6, 880, 932.3, 987.8, 1046.5]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

predicted_notes = []
for f in frequency:
    if f <= 440:
        predicted_notes.append(0)
    else:
        pred = find_nearest(notes_frequecies, f)
        predicted_notes.append(pred)

# print(predicted_notes)


tnrfont = {'fontname':'Times New Roman',
           'size': 13}
fig= plt.figure()
ax = fig.add_subplot(2, 1, 1)
# print(frequency, len(frequency))
ax.plot(times, frequency, 's-', color = "black")
ax.plot(times, predicted_notes, 's-', color = "blue")
ax.set_yscale('log')
# fig.(f"GRW-MH, Preconditioned with max(FFT), sd = 5", **tnrfont)
ax.grid(lw = 0.5)
ax.set_xlim(0,l)
ax.set_ylim(400,1200)
ax.set_yticks(notes_frequecies, notes_names)
# ax.xlabel("Time", **tnrfont)
# plt.set_
# ax.ylabel("Frequency / Hz", **tnrfont)
# plt.hlines(659.25,[0],[len(x)],colors="black", linestyles='dotted')
# plt.xticks(range(0,len(x)+1,2))
plt.show()