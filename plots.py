import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, invgamma



def plot_w_prior(K, M, sigma):
    # Plots frequency prior given K and M
    # Though not quite since actual distribution is multivariate normal
    # K = [293.66, 659.25]
    # M = [15,15]

    N = 1000
    x = np.linspace(10,(M[-1]+1)*K[-1],N)
    y = np.zeros(N)

    colours = ["blue","orange", "green", "red", "pink"]
    colours_arg = []
    ws = []

    for i, order in enumerate(M):
        for partial in range(order):
            colours_arg.append(colours[i])
            w = (partial+1) * K[i]
            ws.append(w)
            y += norm(loc=w, scale = sigma).pdf(x)
            
    mag = max(y)

    fig, ax = plt.subplots(1, 1)
    ax.vlines(ws, -0.1*mag, 0.1*mag, colors=colours_arg)
    ax.plot(x, y)
    fig.suptitle("Frequency Prior P(w|K,M) \n K = [293.66, 659.25], M = [15, 15]")
    plt.xlabel("Frequency/Hz")
    plt.ylabel("Magnitude")
    plt.show()


def M_prior_plot(rate=10):
    # Plots poisson Distribution
    fig, ax = plt.subplots(1, 1)
    mu = rate
    x = np.arange(0,23)
    ax.plot(x, poisson.pmf(x, mu), 'x', label='poisson pmf')
    fig.suptitle("Orders Prior P(M|K) (Poisson) \n rate = 10")
    plt.xlabel("M")
    plt.ylabel("Probability")
    plt.show()

def frequency_priors():
    # Returns array of equal tempered note frequencies
    
    frequencies = np.zeros(89)
    for f in range(89):
        frequencies[f] = 440*(2**(1/12))**(f-49)
        frequencies[f] = 440*2**((np.round_(12*np.log2(frequencies[f]/440)))/12)
    return frequencies

def plot_k_prior(frequencies,sigma):
    # Plots frequency prior given K and M
    # Though not quite since actual distribution is multivariate normal
    # K = [293.66, 659.25]
    # M = [15,15]

    N = 1000
    x = np.linspace(0.8*frequencies[0],1.2*frequencies[-1],N)
    y = np.zeros(N)

    for f in frequencies:
        y += norm(loc=f, scale = sigma).pdf(x)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    fig.suptitle("Note Prior P(K) \n K = c3(130.8hz) to c5(523.25hz)")
    plt.xlabel("Frequency/Hz")
    plt.ylabel("Probability")
    plt.show()

def plot_noise_prior(mean):

    fig, ax = plt.subplots(1, 1)
    a = 1/mean
    x = np.linspace(invgamma.ppf(0.01, a),
                    invgamma.ppf(0.99, a), 100)
    ax.plot(x, invgamma.pdf(x, a))

    fig.suptitle("Noise Prior P($\sigma^2$) (Inverse Gamma), mean=1e-2")
    plt.xlabel("$\sigma^2$")
    plt.ylabel("Probability")
    plt.show()

if __name__ == '__main__':
    # Plot the frequency prior
    K = [293.66, 659.25]
    M = [15,15]
    plot_w_prior(K,M,50)

    # Plot the orders prior
    M_prior_plot()

    # Plot K priors
    freqs = frequency_priors()
    # K = [293.66, 659.25]
    # K = freqs[28:52]
    plot_k_prior(K,2)

    # Plot noise prior
    plot_noise_prior(1e-2)


    
