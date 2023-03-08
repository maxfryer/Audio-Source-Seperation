from main import *
import csv

e5, sr = librosa.load("mp3 Notes\e5.mp3", sr =44100)
c5, sr = librosa.load("mp3 Notes\c5.mp3", sr =44100)
a4, sr = librosa.load("mp3 Notes\\a4.mp3", sr =44100)

mix = np.append(e5, c5)
mix = np.append(mix, a4)


flu = AudioSourceSeperation("recordings\c-4.mp3", 4000)
cyrinx = librosa.load("recordings\cyrinx.mp3", sr=44100)[0]
flu.import_numpy(cyrinx) #cyrinx

window = 0.08
print(flu.time)
times = np.arange(0,flu.time, window)
frequency = np.zeros(len(times)-1)

for i in range(len(times)-1):
    print(times[i], times[i+1])


for i in range(len(times)-1):
    print(times[i], times[i+1])

    flu.set_duration(times[i], times[i+1])

    # Number of samplepoints
    N = len(flu.samples)
    # sample spacing
    T = 1.0 / flu.rate
    x = np.linspace(0.0, N * T, N)
    yf = scipy.fftpack.fft(flu.samples)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    m = max(2.0 / N * np.abs(yf[:N // 2]))
    # # pos = (2.0 / N * np.abs(yf[:N // 2])).index(m)
    itemindex = np.where((2.0 / N * np.abs(yf[:N // 2])) == m)
    max_freq = xf[itemindex]

    w_init = max_freq



    # w_init = [random.uniform(400,720)] 
    # w_init = [600]#659.25
    M = [11]

    flu.M = [11]
    print(w_init)

    x, acc = flu.singleMCMC(w_init, 20, sd=5)
    # flu.fft_plot()

    # MCMC Convergance
    tnrfont = {'fontname':'Times New Roman',
            'size': 13}

    # print(x[-1], acc)
    # plt.plot(x, 's-', color = "black")
    # plt.title(f"Simulated Annealing MCMC\n beta={20}, w0 ={w_init}Hz, wN ={np.around(x[-1][0],2)}Hz", **tnrfont)
    # plt.grid(lw = 0.5)
    # plt.xlabel("Iteration", **tnrfont)
    # plt.ylabel("Frequency / Hz", **tnrfont)
    # plt.hlines(659.25,[0],[len(x)],colors="black", linestyles='dotted')
    # plt.xticks(range(0,len(x)+1,2))
    # plt.show()
    print(flu.final_w)
    print(w_init)

    # flu.generate_M_K_w(flu.final_w, M)
    # flu.compare_residuals(flu.final_w)

    # Calculate inital fit
    # flu.generate_M_K_w(w_init, M)
    # flu.compare_residuals(w_init)

    frequency[i] = flu.final_w


with open('freqs.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(times[:-1])
    writer.writerow(frequency)

# print(frequency, len(frequency))
plt.plot(times[:-1], frequency, 's-', color = "black")
plt.title(f"Syrinx, Debussy, Opening Flute Solo", **tnrfont)
plt.grid(lw = 0.5)
plt.xlabel("Time / s", **tnrfont)
plt.ylabel("Frequency / Hz", **tnrfont)
# plt.hlines(659.25,[0],[len(x)],colors="black", linestyles='dotted')
# plt.xticks(range(0,len(x)+1,2))
plt.show()