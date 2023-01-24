import string
import random
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

import scipy.io.wavfile as si
import scipy.stats as stats
import scipy.integrate as scint

from tqdm import tqdm

# We'll need IPython.display's Audio widget
from IPython.display import Audio


y, sr = librosa.load("recordings/download.wav", sr=44100, offset=0.6, duration = 1)

class ProcessAudio:
    def __init__(self, recording, root_freq) -> None:
        self.recording = recording
        self.root_frequency = root_freq
        self.sampling_frequency = librosa.load(recording, sr=44100)[1]
        self.samples = librosa.load(recording, sr=44100)[0]
        self.N = len(self.samples)
        self.start = 0
        self.stop= self.N
        self.sample_counter = np.array(range(self.N))
        self.windows = self.generate_window_functions(5000)
        self._raw_samples = librosa.load(recording, sr=44100)[0]

    def __str__(self) -> str:
        return f"Recording File: {self.recording} \n" \
        + f"Sampling Frequency: {self.sampling_frequency} \n" \
        + f"Samples: {self.start} to {self.stop} ({self.N}) \n" \
        + f"Time: {(self.start/self.sampling_frequency):.2f}s to {(self.stop/self.sampling_frequency):.2f}s " \
            + f"({((self.stop- self.start)/self.sampling_frequency):.2f}s) \n" \

    def set_duration(self, start, stop):
        self.start = int(start * self.sampling_frequency)
        self.stop= int(stop * self.sampling_frequency)
        self.N = self.stop-self.start
        self.sample_counter = np.array(range(self.N))
        self.windows = self.generate_window_functions(5000)
        self.samples = self._raw_samples[self.start:self.stop]
        print(self.__str__()) # show updated info

    def generate_window_functions(self, window_length):
        """Generate the Hanning window matrix based on number of samples and length of windows.
        Inputs:
            samples - Array of samples
            window_length - Desired length of windows
        Outputs:
            window_function - matrix of n x l window values, where n is number of windows and l in number of samples"""
        rem = self.N % (window_length // 2)
        hann = np.hanning(window_length)

        windows = range(0, self.N, window_length // 2)

        x = np.zeros(self.N)

        y = np.zeros(self.N)
        y[:window_length // 2] = hann[-window_length // 2:]
        window_function = y

        for window in windows[1:-1]:
            y = np.zeros(self.N)
            y[window - window_length // 2:window + window_length // 2] = hann
            window_function = np.concatenate((window_function, y))

        y = np.zeros(self.N)
        y[windows[-1] - window_length // 2:windows[-1] + rem] = hann[:window_length // 2 + rem]
        window_function = np.concatenate((window_function, y))

        y = np.zeros(self.N)
        y[-rem:] = hann[:rem]
        window_function = np.concatenate((window_function, y))

        window_function = np.reshape(window_function, (len(window_function) // (self.N), self.N))
        print(f"Windows: {window_function.shape}")
        return window_function

    def generate_d_matrix(self, windows, model_order):
        # Will have to alter this to add more than one note
        d_matrix = np.empty((self.N , 2 * len(windows) * model_order))

        for t in tqdm(self.sample_counter):
            # collects array of basis weights for each time point
            windows_vec = np.empty((0, 2 * model_order))
            for num, window in enumerate(windows):
                # collects array of basis cosines and sines for each window
                weights = window[t] * np.array(
                    [(np.cos(2 * np.pi * m * self.root_frequency * t / 44100), np.sin(2 * np.pi * m * self.root_frequency * t / 44100)) for m in
                    range(1, model_order + 1)])
                windows_vec = np.append(windows_vec, weights)
            d_matrix[t] = np.array([windows_vec])
        print(f"D:{d_matrix.shape}")
        return d_matrix


class VisualiseData:
    def __init__(self, data) -> None:
        self.data = data
        self.N = len(self.data)
        self.sample_counter = np.array(range(self.N))


    def __str__(self) -> str:
        return f"Length: {self.N}"

    def visualise(self):
        self.constant_q_spectrogram(self)
        self.spectrogram(self)

    def constant_q_spectrogram(self):
        C = np.abs(librosa.iirt(self.data, sr=sr))
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                    sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
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



    


