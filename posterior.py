import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as si
from tqdm import tqdm

root_freq = 261.6

y, sr = librosa.load("download.wav", sr=44100, offset=1, duration=0.01)
sample_no = list(range(len(y)))


def example_tone(t, root, sigma):
    return 3 * np.sin(2 * np.pi * root * t / 44100) + 2 * np.cos(2 * np.pi * root * t / 44100) \
           + 1 * np.sin(2 * np.pi * root * t / 44100 * 2) + 0.7 * np.cos(2 * np.pi * root * t / 44100 * 2) \
           + np.random.normal(0, sigma)


def basis_function(t, root, degree):
    basis = [(np.cos(2 * np.pi * m * root * t / 44100), np.sin(2 * np.pi * m * root * t / 44100)) for m in
             range(1, degree + 1)]
    return [1] + [item for t in basis for item in t]


def MAP(time_since_note_beginning, y_points, degree, error=0.05, variance=0.1, initial=0.4, decay_const=2.61e-5,
        degree_const =0.231):
    # decay const means weights half every .6s
    # degree const means weights half every 3 degrees

    matrix = np.array([basis_function(t, root_freq, degree) for t in time_since_note_beginning])
    print(" mat", matrix)
    covariance = np.diag([variance for i in range(2 * degree + 1)])
    print(covariance)

    mean_prior = np.exp(0) * initial * \
                 np.transpose(np.array([np.exp(-degree_const * (d // 2)) for d in range(2 * degree)]))
    mean_prior = np.reshape(np.concatenate(([0], mean_prior)), (2 * degree + 1, 1))

    # print(mean_prior, covariance)
    print(matrix.shape)


    w = np.linalg.inv(np.transpose(matrix) @ matrix + error * np.linalg.inv(covariance)) \
                @ (error ** -1 * np.transpose(matrix) @ y_points)

    return w


w = MAP(sample_no, y, degree=7)

print(w)

# plt.plot(w)
# plt.show()

# x = np.linspace(0,15,1500)
# y = [0.4 * np.exp(-i*2.31e-1) for i in x]
# plt.plot(x,y)
# plt.ylabel("Mean Prior Magnitude")
# plt.xlabel("Root Degree")
# plt.suptitle("Exponential Mean Prior on Weights $t_{half} = 3$ ")
# plt.show()