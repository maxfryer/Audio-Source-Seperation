import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import librosa
import librosa.display
import logging
from mpl_toolkits.mplot3d import axes3d

logging.basicConfig(format='%(asctime)s\t%(levelname)s:%(message)s', level=logging.INFO)

# y, sr = librosa.load('mp3 Notes/c5.mp3', sr=None, offset=0.5, duration=0.4)
# y, sr = librosa.load('bachWTC.mp3', sr=None, duration=1)
y, sr = librosa.load('download.wav', sr=None)

sr = 44100
M = 10 * 2048
N = 1 * 2048


def constant_q_spectrogram(y):
    C = np.abs(librosa.iirt(y, sr=sr))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                   sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


# constant_q_spectrogram(y)


def reassigned_spectrogram(y, n_fft=2048):
    freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=sr,
                                                        n_fft=n_fft)
    mags_db = librosa.amplitude_to_db(mags, ref=np.max)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,
                                   hop_length=n_fft // 4, ax=ax[0])
    ax[0].set(title="Spectrogram", xlabel=None)
    ax[0].label_outer()
    ax[1].scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
    ax[1].set_title("Reassigned spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


# reassigned_spectrogram(y)


def stft_plot(y, start_sample, samples):
    S = np.abs(librosa.stft(y[start_sample:start_sample + samples], n_fft=2048))
    logging.info(f"Shape of input:{y.shape}")
    f = librosa.fft_frequencies(sr=44100, n_fft=2048)
    fig, ax = plt.subplots()
    ax.set_title('Semitone spectrogram (iirt)')
    logging.info(f"Shape of plot:{S.shape}")
    plt.plot(S)
    plt.show()

# stft_plot(y,M,N)

def vqt_plot(y):
    C = np.abs(librosa.cqt(y, sr=sr))
    V = np.abs(librosa.vqt(y, sr=sr))
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                             sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[0])
    ax[0].set(title='Constant-Q power spectrum', xlabel=None)
    ax[0].label_outer()
    img = librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),
                                   sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[1])
    ax[1].set_title('Variable-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


# vqt_plot(y)


def iirt_plot(y):
    D = np.abs(librosa.iirt(y))
    C = np.abs(librosa.cqt(y=y, sr=sr))
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                   y_axis='cqt_hz', x_axis='time', ax=ax[0])
    ax[0].set(title='Constant-Q transform')
    ax[0].label_outer()
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                   y_axis='cqt_hz', x_axis='time', ax=ax[1])
    ax[1].set_title('Semitone spectrogram (iirt)')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


# iirt_plot(y)

def three_d_wireplot():
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    # Get the test data
    X, Y, Z = axes3d.get_test_data(0.05)

    # Give the first plot only wireframes of the type y = c
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=0, style='ggplot')
    ax.set_title("Column (x) stride set to 0")

    plt.tight_layout()
    plt.show()

# three_d_wireplot()