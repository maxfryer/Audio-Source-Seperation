
from main import *

e5, sr = librosa.load("mp3 Notes/e5.mp3", sr=44100)
d4, sr = librosa.load("mp3 Notes/d4.mp3", sr=44100)
mix = e5[:len(d4)]+ d4
print(mix.shape)

eg = AudioSourceSeperation("recordings/download.wav")
eg.import_numpy(mix)
print(eg.samples.shape)
eg.set_duration(0,0.2)
eg.generate_window_functions(5000)


for b in [3,10,30]:
    for w in [[290, 670],[250,710],[220,740]]:
        x, acc = eg.MCMC(w, beta=b, n_iters=20)
        plt.plot(x, '.-')
        plt.title(f"MCMC Frequency, acc={acc}, beta={b}\n w0 ={w}Hz\n wN ={np.around(x[-1],2)}Hz")
        plt.xlabel("Iterations")
        plt.hlines([293.66, 659.25],[0,0],[len(x),len(x)],colors="black", linestyles='dotted')
        plt.xticks(range(0,len(x)+1,2))
        plt.ylabel("Frequency")
        plt.show()

