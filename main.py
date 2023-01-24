import string
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa

import scipy.io.wavfile as si
import scipy.stats as stats
import scipy.integrate as scint

from tqdm import tqdm

y, sr = librosa.load("recordings/download.wav", sr=44100, offset=0.6, duration = 1)

class ProcessAudio:
    def __init__(self, recording, root_freq) -> None:
        self.recording = recording
        self.root_frequency = root_freq
        self.sampling_frequency = librosa.load(recording, sr=44100)[1]
        self.samples = librosa.load(recording, sr=44100)[0]
        self.N = len(self.samples)
        self.start = 0
        self.end = self.N/self.sampling_frequency
        self.sample_counter = np.array(range(self.N))

    def __str__(self) -> str:
        return f"Recording File: {self.recording} \n" \
        + f"Sampling Frequency: {self.sampling_frequency} \n" \
        + f"Number of Samples: {self.N} ({(self.N/self.sampling_frequency):.2f}s) \n" \
        + f"{self.start}s to {self.end:.2f}s ({self.end - self.start:.2f}s) \n" \

    def set_duration(self, start, stop):
        self.start = start
        self.end = stop
        self.N = int((self.end-self.start) * self.sampling_frequency)
        self.sample_counter = np.array(range(self.N))

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
        print(f"Windows:{window_function.shape}")
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


