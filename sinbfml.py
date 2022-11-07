import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as si

root_freq = 261.6
# root_freq = 440

# y, sr = librosa.load("download.wav", sr=44100)
y, sr = librosa.load("download.wav", sr=44100, offset=2, duration=0.5)
sample_no = list(range(len(y)))


# write_time = np.linspace(0, 1, 44100)
# write_y = [resynth_sound(w, i) for i in write_time]
# si.write('resynth_sound.wav', 44100, y)


def example_tone(t, root, sigma):
    return 3 * np.sin(2 * np.pi * root * t / 44100) + 2 * np.cos(2 * np.pi * root * t / 44100) \
           + 1 * np.sin(2 * np.pi * root * t / 44100 * 2) + 0.7 * np.cos(2 * np.pi * root * t / 44100 * 2) \
           + np.random.normal(0, sigma)


def basis_function(t, root, degree=5):
    basis = [(np.cos(2 * np.pi * m * root * t / 44100), np.sin(2 * np.pi * m * root * t / 44100)) for m in
             range(1, degree + 1)]
    return [1] + [item for t in basis for item in t]


def maximum_likelihood(times, y_points, degree=5):
    matrix = np.array([basis_function(t, root_freq, degree) for t in times])
    # print(matrix.shape)
    w = np.linalg.inv(np.transpose(matrix) @ matrix) @ np.transpose(matrix) @ y_points
    # print(w)
    return w


def resynth_sound(w, t, degree):
    basis = basis_function(t, root_freq, degree)
    return np.dot(w, basis)


def fft_plot(y):
    S = np.fft.fft(y)
    S = np.log(np.abs(S))
    # logging.info(f"Shape of input:{y.shape}")
    fig, ax = plt.subplots()
    ax.set_title('fft')
    plt.plot(S[:1000])
    plt.show()


def time_varying_weights(window_length, y_points, samples, ml_length=300, degree=5):
    weights_array = []
    window_times = samples[::window_length] + [samples[-1]]
    if len(samples) < window_length:
        print(f'Too short (length:{len(samples)})')
        return

    weights = maximum_likelihood(range(ml_length // 2), y_points[:ml_length // 2], degree)
    # windowing_data.append(np.array(y_points[:ml_length//2]))

    for s in window_times[1:-2]:
        points = np.concatenate(
            (np.array([y_points[s - ml_length // 2:s]]), np.array([y_points[s:s + ml_length // 2]])), axis=1)
        w = maximum_likelihood(range(s - ml_length // 2, s + ml_length // 2), points[0], degree)
        weights = np.concatenate((weights, w))

    if (window_times[-1]-window_times[-2]< ml_length//2): #deals with case that last window is really short
        s = window_times[-2]
        points = np.array([y_points[s - ml_length // 2:s]])
        w = maximum_likelihood(range(s - ml_length // 2, s), points[0], degree)
        weights = np.concatenate((weights, w))
    else:
        s = window_times[-2]
        points = np.concatenate(
            (np.array([y_points[s - ml_length // 2:s]]), np.array([y_points[s:s + ml_length // 2]])), axis=1)
        w = maximum_likelihood(range(s - ml_length // 2, s + ml_length // 2), points[0], degree)
        weights = np.concatenate((weights, w))

    w = maximum_likelihood(range(samples[-ml_length // 2], samples[-ml_length // 2] + ml_length // 2),
                           y_points[-ml_length // 2:], degree)
    weights = np.concatenate((weights, w))
    weights = np.reshape(weights, (len(weights) // (2 * degree + 1), 2 * degree + 1))
    return window_times, weights


wt, w = time_varying_weights(1000, y, sample_no, degree=8)

def interpolate_weights(wt, w, degree=5):
    """
    :param wt: sample times that weights are centred on
    :param w: weights corresponding to sample times
    :return: interpolated weights for each sample between wt[0] and wt[-1]
    """
    interp_weights = w[0]
    for i in range(0, len(wt) - 1):
        print(f'this next corresponds to{wt[i]} to {wt[i + 1]}')
        for s in range(wt[i], wt[i + 1]):
            w1 = (wt[i + 1] - s) / (wt[i + 1] - wt[i])
            w2 = (s - wt[i]) / (wt[i + 1] - wt[i])
            weight = w1 * w[i] + w2 * w[i + 1]
            interp_weights = np.concatenate((interp_weights, weight))
    interp_weights = np.reshape(interp_weights, (len(interp_weights) // (2 * degree + 1), 2 * degree + 1))
    interp_weights = np.delete(interp_weights, 0, 0)
    return np.array(range(wt[0], wt[-1])), interp_weights


s, iws = interpolate_weights(wt, w, degree=8)

print(s.shape, iws.shape)

y_sim = [resynth_sound(iws[t],s[t], 8) for t in range(len(s))]
scaled = np.int16(y_sim / np.max(np.abs(y_sim)) * 32767)
si.write('synthisized.wav', 44100, scaled)
# si.write('original.wav', 44100, y)

# x = range(11)
# fig, axs = plt.subplots(2, 3)
# axs[0,0].bar(x, w[0])
# axs[0,0].set_title('Axis 0')
# axs[0,1].bar(x, w[1])
# axs[0,1].set_title('Axis 1')
# axs[0,2].bar(x, w[2])
# axs[0,2].set_title('Axis 2')
# axs[1,0].bar(x, w[3])
# axs[1,0].set_title('Axis 3')
# axs[1,1].bar(x, w[4])
# axs[1,1].set_title('Axis 4')
# axs[1,2].bar(x, w[5])
# axs[1,2].set_title('Axis 5')
# plt.show()


# print(wt, w)
# fft_plot(y)

# #

'''
times = range(0,440)
y = [example_tone(t, root_freq, 0) for t in times]
w = maximum_likelihood(times, y)
y_sim = [resynth_sound(w, i) for i in times]
'''

'''
writes 0.1 seconds of audio, just to check that a 261.1hz going in is 261.1 going out
y, sr = librosa.load("download.wav", sr=44100, offset=2, duration=0.1)
sample_no = list(range(len(y)))

w = maximum_likelihood(sample_no, y)
y_sim = np.array([resynth_sound(w, i) for i in sample_no])
scaled = np.int16(y_sim / np.max(np.abs(y_sim)) * 32767)    note the scaling
si.write('2.wav', 44100, scaled)
'''

# plt.plot(sample_no, y)
# plt.plot(sample_no, y_sim)
# plt.show()


#
#
# plt.plot(tim, y_sim)
# plt.plot(times, y_sim)
# plt.xlabel("time/s")
# plt.ylabel("amplitude (normalized)")
# plt.title("comparison of regression model with true recording")
plt.show()
