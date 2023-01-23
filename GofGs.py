import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as si
import scipy.stats as stats
import scipy.integrate as scint
from tqdm import tqdm

root_freq = 261.6
# root_freq = 440

y, sr = librosa.load("download.wav", sr=44100, offset=0.6, duration = 1)


vio1, sr = librosa.load("356181__mtg__violin-d5.wav", sr=44100, offset=0, duration = 2.5)
vio2, sr = librosa.load("356176__mtg__violin-gsharp4.wav", sr=44100, offset=0, duration = 2.5)
pia1, sr = librosa.load("download.wav", sr=44100, offset=0.6, duration = 1.5)
pia2, sr = librosa.load("download.wav", sr=44100, offset=0.6, duration = 2)


piano = np.concatenate((pia1, pia1, pia2,[0,0,0,0]))
Violin = np.concatenate((vio1,vio2))


scaled1 = np.int16(piano / np.max(np.abs(piano)) * 32767)
scaled2 = np.int16(Violin / np.max(np.abs(Violin)) * 32767)

mixed = piano + Violin
mixed = mixed[:100000]
mix_scale = np.int16(mixed / np.max(np.abs(mixed)) * 32767)

# si.write('mixed.wav', 44100, mix_scale)
sample_no = list(range(len(y)))


def windowfunctions(samples, window_length):
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

    print(window_function.shape)
    return window_function


def dmatrix(samples, windows, model_order):
    # Will have to alter this to add more than one note
    d_matrix = np.empty((len(samples), 2 * len(windows) * model_order))

    for t in tqdm(samples):
        # collects array of basis weights for each time point
        windows_vec = np.empty((0, 2 * model_order))
        for num, window in enumerate(windows):
            # collects array of basis cosines and sines for each window
            weights = window[t] * np.array(
                [(np.cos(2 * np.pi * m * root_freq * t / 44100), np.sin(2 * np.pi * m * root_freq * t / 44100)) for m in
                 range(1, model_order + 1)])
            windows_vec = np.append(windows_vec, weights)
        d_matrix[t] = np.array([windows_vec])
    print(d_matrix.shape)
    return d_matrix

def dmatrixofdmatrix(samples, windows, model_order):
    

    d_matrix1 = np.empty((len(samples), 3* 2 * len(windows) * model_order))
    d_matrix2 = np.empty((len(samples), 3* 2 * len(windows) * model_order))
    d_matrix3 = np.empty((len(samples), 3* 2 * len(windows) * model_order))

    # for f in [587.33, 415.3, 261.6]:

    for t in tqdm(samples):
        # collects array of basis weights for each time point
        windows_vec = np.empty((0, 2 * model_order))
        for num, window in enumerate(windows):
            # collects array of basis cosines and sines for each window
            weights = window[t] * np.array(
                [(np.cos(2 * np.pi * m * 261.6 * t / 44100), np.sin(2 * np.pi * m * 261.6 * t / 44100)) for m in
                range(1, model_order + 1)])
            windows_vec = np.append(windows_vec, weights)
        d_matrix1[t] = np.array([windows_vec])

    for t in tqdm(samples):
        # collects array of basis weights for each time point
        windows_vec = np.empty((0, 2 * model_order))
        for num, window in enumerate(windows):
            # collects array of basis cosines and sines for each window
            weights = window[t] * np.array(
                [(np.cos(2 * np.pi * m * 415.3 * t / 44100), np.sin(2 * np.pi * m * 415.3 * t / 44100)) for m in
                range(1, model_order + 1)])
            windows_vec = np.append(windows_vec, weights)
        d_matrix2[t] = np.array([windows_vec])
    
    for t in tqdm(samples):
        # collects array of basis weights for each time point
        windows_vec = np.empty((0, 2 * model_order))
        for num, window in enumerate(windows):
            # collects array of basis cosines and sines for each window
            weights = window[t] * np.array(
                [(np.cos(2 * np.pi * m * 587.33 * t / 44100), np.sin(2 * np.pi * m * 587.33 * t / 44100)) for m in
                range(1, model_order + 1)])
            windows_vec = np.append(windows_vec, weights)
        d_matrix3[t] = np.array([windows_vec])

    DofD = np.append(d_matrix1,d_matrix2, axis=1)
    DofD = np.append(DofD,d_matrix3, axis=1)

    print(d_matrix1.shape)
    print(DofD.shape)

    return DofD

windows = windowfunctions(sample_no, 5000)
dmat = dmatrixofdmatrix(sample_no, windows, 10)


def weights_priors(R, I, sigma_v, alpha=1e-4, beta=1e-4):
    mean = np.zeros((2*R*(I+1), 1))
    x = np.linspace(1e-6,5e-4,1000)
    zeta_prob = lambda x: np.exp(-beta/x)/(x ** (alpha+1)) #runtime warning since x is very small
    weights_cov = [zeta_prob(i) for i in np.linspace(0,5e-4,1000)] # trying to do this with scipy func 
    zeta = 3e-6 #this is just a reasonable sample
    # zeta = stats.gamma.pdf(x, a=alpha, loc=1/beta)
    # plt.plot(x,weights_cov)
    # plt.show()
    diag_val = sigma_v/zeta
    cov = np.diag(diag_val * np.ones(2*R*(I+1)))
    return mean, cov

# weights_priors(3,5,0.1)

def frequency_priors():
    '''set around equal tempered notes'''

    frequencies = np.zeros(89)
    for f in range(89):
        frequencies[f] = 440*(2**(1/12))**(f-49)
        frequencies[f] = 440*2**((np.round_(12*np.log2(frequencies[f]/440)))/12)
    print(frequencies)

    '''Gaussian of width 3sd'''
    x = np.linspace(0,4300,10000)
    for note in frequencies[:-1]:
        next_note = note*(2**(1/12))**(1)
        diff = next_note-note
        # stats.norm.pdf(x,loc = note, scale = diff/6) for x in range(x-3*diff, x+3*diff)
        # x[]

    # num_partials = np.zeros(88)
    # for f in range(88):
    #     num_partials[f] = min(22000//frequencies[f], 30)
    # print(num_partials)
    
def partials_prior(Bv = 0.05):
    m_prob = lambda m: (Bv+1)**(-m)
    x = range(0,35,1)
    ms = [m_prob(i) for i in x]
    plt.plot(x,ms)
    plt.suptitle("Number of Partials Prior $(Bv+1)^{-m}, Bv=0.05$")
    plt.xlabel("Number of Partials")
    plt.ylabel("Prior (unnormalized)")
    plt.show()

# frequency_priors()
# partials_prior()


def maximumlikelihood(y_points, dmat):
    matrix = dmat
    # print(matrix.shape)
    w = np.linalg.inv(np.transpose(matrix) @ matrix) @ np.transpose(matrix) @ y_points
    # print(w)
    return w


def resynth_sound(w, dmat):
    y = dmat @ w
    return y

# priors(y,1)

'''
windows = windowfunctions(sample_no, 4000)
dmat = dmatrix(sample_no, windows, 10)
w = maximumlikelihood(y, dmat)
y_sim = resynth_sound(w, dmat)
plt.plot(sample_no,y)
plt.plot(sample_no,y_sim)
plt.show()

scaled = np.int16(y_sim / np.max(np.abs(y_sim)) * 32767)
si.write('gofgsound.wav', 44100, scaled)
'''

# print(w.shape)
# print(w)

