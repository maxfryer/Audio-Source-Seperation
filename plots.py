import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

K = [293.66, 659.25]
M = [15,15]

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
        print(i)
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

plot_w_prior(K,M,50)