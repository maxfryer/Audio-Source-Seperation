import string
import random
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import scipy.io.wavfile as si
import scipy.stats as stats
import scipy.integrate as scint
import scipy.fftpack

from tqdm import tqdm

# We'll need IPython.display's Audio widget
from IPython.display import Audio


# y, sr = librosa.load("recordings/download.wav", sr=44100, offset=0.6, duration = 1)

class ProcessAudio:
    def __init__(self, recording, root_freq) -> None:
        self.recording = recording
        self.root_frequency = root_freq
        self.rate = librosa.load(recording, sr=44100)[1]
        self.samples = librosa.load(recording, sr=44100)[0]
        self.N = len(self.samples)
        self.start = 0
        self.stop= self.N
        self.sample_counter = np.array(range(self.N))
        self.window_length = 5000
        self.windows = self.generate_window_functions(self.window_length)
        self.I = self.windows.shape[0]
        self._raw_samples = librosa.load(recording, sr=44100)[0]
        self.K = None
        self.M = None
        self.w = None

    def __str__(self) -> str:
        return f"Recording File: {self.recording} \n" \
        + f"Sampling Frequency: {self.rate} \n" \
        + f"Samples: {self.start} to {self.stop} ({self.N}) \n" \
        + f"Time: {(self.start/self.rate):.2f}s to {(self.stop/self.rate):.2f}s " \
            + f"({((self.stop- self.start)/self.rate):.2f}s) \n" \

    def set_duration(self, start, stop):
        self.start = int(start * self.rate)
        self.stop= int(stop * self.rate)
        self.N = self.stop-self.start
        self.sample_counter = np.array(range(self.N))
        self.windows = self.generate_window_functions(self.window_length)
        self.samples = self._raw_samples[self.start:self.stop]
        self.I = len(self.windows)
        print(self.__str__()) # show updated info
    
    def generate_M_K_w(self, Notes, Orders):
        self.K = len(Notes)
        self.M = Orders
        w = []
        for i, root in enumerate(Notes):
            w.append([root*i for i in range(1, Orders[i]+1)])
        self.w = w

    def generate_window_functions(self, window_length):
        window_starts = np.arange(0, self.N, window_length // 2)
        I = len(window_starts)+1
        p = window_length // 2
        hanning = np.hanning(2*p)
        windows = np.zeros((I,self.N))

        # First window
        first = np.zeros(self.N)
        first[:p] = hanning[-p:]
        windows[0] = first

        # Middle windows
        for i, w in enumerate(window_starts[1:-1],1):
            middle = np.zeros(self.N)
            middle[w-p:w+p] = hanning
            windows[i] = middle
        
        # Penultimate
        remainder = self.N % p
        pen = np.zeros(self.N)
        pen[-(p+remainder):] = hanning[:p+remainder]
        windows[I-2] = pen

        # Last window
        last = np.zeros(self.N)
        last[-remainder:] = hanning[:remainder]
        windows[I-1] = last

        window_starts = np.concatenate(([0], window_starts, [self.N, self.N]))

        self.windows = windows
        self._window_starts = window_starts
        print(f"Windows: {windows.shape}")

        return windows

    def _generate_note_D(self, w, t):
        n = 2 * len(w)
        note = np.zeros(n)
        for i, m in enumerate(w):
            note[2*i] = np.cos(2 * np.pi * m * t / self.rate)
            note[2*i + 1] = np.sin(2 * np.pi * m * t / self.rate)
        return note

    def generate_D(self):
        width = 2 * sum(self.M)
        D = np.zeros((self.N, width * self.I))
        
        for i in tqdm(range(self.I)):
            note = np.zeros((self.N, width))
            for sample in range(self._window_starts[i],self._window_starts[i+2]):
                counter = 0
                for k in range(self.K): #iterate over notes
                    note[sample][counter:counter+2*self.M[k]] = self._generate_note_D(self.w[k], sample) * self.windows[i][sample]
                    counter += 2*self.M[k]

            D[:, i*width:(i+1)*width] = note
        self.D = D
        return D

    def import_numpy(self, samples):
        self._raw_samples = samples
        self.samples = samples
        self.N = len(samples)
        self.start = 0
        self.stop = self.N
        self.windows = self.generate_window_functions(self.window_length)
        self.I = self.windows.shape[0]
        print(self.__str__()) # show updated info
        return

    def fft_plot(self):
        colours = ["blue","orange", "green", "red", "pink"]

        colours_arg = []
        for i, order in enumerate(self.M):
            # i=0, note=15
            for a in range(order): 
                colours_arg.append(colours[i])

        # Number of samplepoints
        N = len(self.samples)
        # sample spacing
        T = 1.0 / self.rate
        x = np.linspace(0.0, N * T, N)
        yf = scipy.fftpack.fft(self.samples)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

        fig, ax = plt.subplots()
        mag = max(2.0 / N * np.abs(yf[:N // 2]))

        ax.vlines(self.w, -0.1*mag, 0.1*mag, colors=colours_arg)
        ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        fig.suptitle("FFT with Degree Labels (c4 261.6Hz)")
        plt.xlabel("Frequency/Hz")
        plt.ylabel("Magnitude")
        plt.xlim([0,5000])
        plt.show()
    
class AnalyseAudio:
    def __init__(self, D, x, M, K, w) -> None:
        self.D = D
        self.x = x
        self.B = None
        self.M = M
        self.K = K
        self.w = w
        pass

    def MLE(self):
        B = np.linalg.inv(np.transpose(self.D) @ self.D) @ np.transpose(self.D) @ self.x
        self.B = B
        return B

    def resynth_sound(self):
        y = self.D @ self.B
        return y

class VisualiseAudio:
    def __init__(self, data) -> None:
        self.data = data
        self.N = len(self.data)
        self.sample_counter = np.array(range(self.N))
        self.sample_rate = 44100

    def __str__(self) -> str:
        return f"Length: {self.N}"

    def visualise(self):
        self.constant_q_spectrogram(self)
        self.spectrogram(self)

    def constant_q_spectrogram(self):
        C = np.abs(librosa.iirt(self.data, sr=self.sample_rate))
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                    sr=self.sample_rate, x_axis='time', y_axis='cqt_note', ax=ax)
        ax.set_title('Constant-Q Power Spectrum')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.show()

    def spectrogram(self):
        D = librosa.stft(self.data)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
        ax.set(title='Spectogram')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.show()

    def frequency_error(self, estimated):
        error = self.data - estimated
        
        N = len(error)
        # sample spacing
        T = 1.0 / 44100
        x = np.linspace(0.0, N*T, N)
        y = error
        yf = scipy.fftpack.fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

        fig, ax = plt.subplots()
        ax.plot(xf, (2.0/N * np.abs(yf[:N//2])))
        plt.xlim([0,5000])
        plt.show()



    


