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

from priors import *

from tqdm import tqdm

# We'll need IPython.display's Audio widget
from IPython.display import Audio


# y, sr = librosa.load("recordings/download.wav", sr=44100, offset=0.6, duration = 1)

class AudioSourceSeperation:
    def __init__(self, recording) -> None:
        self.recording = recording
        self.rate = librosa.load(recording, sr=44100)[1]
        self.samples = librosa.load(recording, sr=44100)[0]
        self._raw_samples = librosa.load(recording, sr=44100)[0]
        self.N = len(self.samples)
        self.start = 0
        self.stop= self.N
        self.sample_counter = np.array(range(self.N))
        self.window_length = 5000
        self.windows = self.generate_window_functions(self.window_length)
        self.I = self.windows.shape[0]
        self.K = None
        self.sigma = None
        
    def __str__(self) -> str:
        return f"Recording File: {self.recording} \n" \
        + f"Sampling Frequency: {self.rate} \n" \
        + f"Samples: {self.start} to {self.stop} ({self.N}) \n" \
        + f"Time: {(self.start/self.rate):.2f}s to {(self.stop/self.rate):.2f}s " \
            + f"({((self.stop- self.start)/self.rate):.2f}s) \n"   
    
    def set_duration(self, start, stop):
        self.start = int(start * self.rate)
        self.stop= int(stop * self.rate)
        self.N = self.stop-self.start
        self.sample_counter = np.array(range(self.N))
        self.windows = self.generate_window_functions(self.window_length)
        self.samples = self._raw_samples[self.start:self.stop]
        self.I = len(self.windows)
        print(self.__str__()) # show updated info
    

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
        self.window_starts = window_starts
        print(f"Windows: {windows.shape}")

        return windows
    
    def generate_M_K_w(self, w, M):
        self.w = w
        self.K = len(w)
        self.M = M
        ws = []
        for i, root in enumerate(w):
            ws.append([root*i for i in range(1, M[i]+1)])
        self.ws = np.array(ws)

    def _generate_note_D(self, ws, t):
        n = 2 * len(ws)
        note = np.zeros(n)
        for i, m in enumerate(ws):
            note[2*i] = np.cos(2 * np.pi * m * t / self.rate)
            note[2*i + 1] = np.sin(2 * np.pi * m * t / self.rate)
        return note

    def generate_D(self):
        self.generate_M_K_w(self.w, self.M)
        width = 2 * np.sum(self.M)
        D = np.zeros((self.N, width * self.I))
        
        for i in range(self.I):
            note = np.zeros((self.N, width))
            for sample in range(self.window_starts[i],self.window_starts[i+2]):
                counter = 0
                for k in range(self.K): #iterate over notes
                    note[sample][counter:counter+2*self.M[k]] = self._generate_note_D(self.ws[k], sample) * self.windows[i][sample]
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

    def log_likelihood(self, w1,w2):
        self.generate_M_K_w([w1,w2], [15,15])
        self.generate_D()
        self.MLE()
        resynthed = self.resynth_sound()
        error = np.linalg.norm(self.samples - resynthed)
        return error
    
    def log_sigma(self, sigma):
        self.sigma = sigma
        # self.w = w
        # self.M = M
        # self.K = K
        w_prior = [293.66, 659.25]
        SNR = 15
        gamma,nu = 1e-4,1e-4
        # self.generate_D()
        # self.generate_S(SNR)
        # self.generate_P()

        ytPy = self.samples.T @ self.P @ self.samples
        log_p = (-(gamma + ytPy)/(2*self.sigma**2)) -(self.N+nu+2)*np.log(self.sigma)
        return log_p

    def log_beta(self, beta):
        self.beta = beta
        mean = self.S @ self.D.T @ self.samples
        cov = self.sigma **2 *self.S
        error = self.beta - mean
        log_p = multivariate_normal(mean=mean, cov=cov).logpdf(error)
        return log_p

    def log_posterior(self, w):
        self.w = w
        self.M = [15,15]
        self.K = 2
        w_prior = [293.66, 659.25]
        SNR = 15
        gamma,nu = 1e-4,1e-4
        self.generate_D()
        self.generate_S(SNR)
        self.generate_P()

        ytPy = self.samples.T @ self.P @ self.samples
        detS = np.linalg.det(self.S)

        log_p = -0.5*(self.N)*np.log(gamma + ytPy) # + 0.5*np.log(np.linalg.det(self.S))
        return(log_p)

    def generate_S(self, SNR):
        R = np.sum(self.M)
        a = self.D.T @ self.D + (1/SNR) * np.eye(2*R*(self.I))
        # print("Inverting S...")
        S = np.linalg.inv(a)
        # print("S Inverted")
        self.S = S
        return S

    def generate_P(self):
        # print("Generating P...")
        P = np.eye(self.N) - self.D @ self.S @ self.D.T
        # print("P Generated")
        self.P = P
        return P

    def MCMC(self, w_init, n_iters=30, beta=40):
        # theta = [w1,w2]
        # theta = theta_init

        # while l < L:

            # theta =  theta_last
            # form candidate root frequencies
                # 85% normal distribution around previous theta[0]
                # 15% uniform distribution
                # theta[0] = new theta
            # MH step
            # new root_freq = (either old or new)

            # theta = theta_last
            # form candidate sigma
                # theta[1] = new theta
            # MH step
            # new sigma = (either old or new)

            # theta = [new_root, new_sigma]
             
        X = []
        acc = 0
        w_prev = w_init

        ll_prev = self.log_posterior(w_prev)


        rand = [np.exp(-2*i/n_iters) for i in np.arange(n_iters)]

        for i in tqdm(range(n_iters)):

            w_new = w_prev + beta * np.random.normal(0, 5,(1,2))[0]
            print(f"proposal:{w_new}")
            print(f"current:{w_prev}")

            ll_new = self.log_posterior(w_new)

            log_alpha = np.minimum(ll_new - ll_prev, 0) #: Calculate pCN acceptance probability
            log_u = np.log(np.random.random())

            # Accept/Reject
            accept = log_alpha >= log_u #: Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
            if accept:
                acc += 1
                X.append(w_new)
                w_prev = w_new
                ll_prev = ll_new
            else:
                X.append(w_prev)

        return X, acc / n_iters

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

        ax.vlines(self.ws, -0.1*mag, 0.1*mag, colors=colours_arg)
        ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        fig.suptitle("FFT of d4 + e5 with harmonics \n w = [293.66Hz, 659.25Hz]")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.xlim([0,5000])
        plt.show()

    def MLE(self):
        B = np.linalg.inv(self.D.T @ self.D) @ self.D.T @ self.samples
        self.B = B
        return B

    def resynth_sound(self):
        y = self.D @ self.B
        return y
    
class AnalyseAudio:
    def __init__(self, y, windows, window_starts) -> None:
        self.y = y
        self.rate = 44100
        self.N = len(self.y)
        self.sample_counter = np.array(range(self.N))
        self.window_length = 5000
        self.windows = windows
        self.window_starts = window_starts
        self.I = self.windows.shape[0]
        self.w = None
        self.K = None
        self.M = None
        self.D = None
        self.B = None
        self.S = None

    def __str__(self) -> str:
        return f"w={self.w}\nK={self.K}\nM={self.M}\nI={self.I}"

    def MLE(self): # Maybe don't use this for a bit
        B = np.linalg.inv(np.transpose(self.D) @ self.D) @ np.transpose(self.D) @ self.y
        self.B = B
        return B

    def resynth_sound(self):
        y = self.D @ self.B
        return y

    def generate_note_D(self, w, t):
        n = 2 * len(w)
        note = np.zeros(n)
        for i, m in enumerate(w):
            note[2*i] = np.cos(2 * np.pi * m * t / self.rate)
            note[2*i + 1] = np.sin(2 * np.pi * m * t / self.rate)
        return note

    def generate_D(self):
        width = 2 * np.sum(self.M)
        N = self.y.shape[0]
        D = np.zeros((N, width * self.I))

        flat =  self.w.flatten()

        for i in tqdm(range(self.I)):
            note = np.zeros((N,width))
            for sample in range(self.window_starts[i], self.window_starts[i+2]):
                note[sample][:] = self.generate_note_D(flat, sample) * self.windows[i][sample]
            D[:, i*width:(i+1)*width] = note
        self.D = D
        return D
    
    def generate_S(self, SNR):
        R = np.sum(self.M)
        a = self.D.T @ self.D + (1/SNR) * np.eye(2*R*(self.I))
        print("inverting S")
        S = np.linalg.inv(a)
        print("S inverted")
        self.S = S
        return S

    def generate_P(self):
        P = np.eye(self.N) - self.D @ self.S @ self.D.T
        self.P = P
        return P

    def posterior_frequecies(self, w, M, K, SNR =10, sigma = 40, mean_notes = 2, mean_partials = 10, gamma=2, nu=0.001):
        # p(w, M, K |y)
        self.w = w
        self.M = M
        self.K = K
        self.generate_D()
        self.generate_S(SNR)
        self.generate_P()
        print((gamma + self.y.T @ self.P @ self.y))
        print(0.5*(-self.N+nu))
        print(np.linalg.det(self.S)**0.5)
        print(prior_w(w, K, M, sigma))
        probability =  (gamma + self.y.T @ self.P @ self.y)**(0.5*(-self.N+nu)) * np.linalg.det(self.S)**0.5 * prior_w(w, K, M, sigma)

        return probability

class VisualiseAudio:
    def __init__(self, data) -> None:
        self.data = data
        self.N = len(self.data)
        self.sample_counter = np.array(range(self.N))
        self.sample_rate = 44100

    def __str__(self) -> str:
        return f"Length: {self.N}"

    def visualise(self):
        self.constant_q_spectrogram()
        self.spectrogram()

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