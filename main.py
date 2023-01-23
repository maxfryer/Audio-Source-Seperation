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
    def __init__(self, recording) -> None:
        self.recording = recording
        self.sampling_frequency = librosa.load(recording)[1]
        self.samples = librosa.load(recording)[0]
        self.N = len(self.samples)

    def __str__(self) -> str:
        info = f"Sampling Frequency: {self.sampling_frequency} \n", f"Num. Samples:{self.N}"
        return info

    def windowfunctions(samples, window_length):
        """Generate the Hanning window matrix based on number of samples and length of windows.
        Inputs:
            samples - Array of samples
            window_length - Desired length of windows
        Outputs:
            window_function - matrix of n x l window values, where n is number of windows and l in number of samples"""
        num_samples = len(samples)
        rem = len(sample_no) % (window_length // 2)
        hann = np.hanning(window_length)

        windows = range(0, num_samples, window_length // 2)

        x = np.zeros(num_samples)

        y = np.zeros(num_samples)
        y[:window_length // 2] = hann[-window_length // 2:]
        window_function = y

        for window in windows[1:-1]:
            y = np.zeros(num_samples)
            y[window - window_length // 2:window + window_length // 2] = hann
            window_function = np.concatenate((window_function, y))

        y = np.zeros(num_samples)
        y[windows[-1] - window_length // 2:windows[-1] + rem] = hann[:window_length // 2 + rem]
        window_function = np.concatenate((window_function, y))

        y = np.zeros(num_samples)
        y[-rem:] = hann[:rem]
        window_function = np.concatenate((window_function, y))

        window_function = np.reshape(window_function, (len(window_function) // (num_samples), num_samples))

        return window_function


