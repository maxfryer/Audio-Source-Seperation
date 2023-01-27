import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as si
import scipy.stats as stats
import scipy.integrate as scint
from tqdm import tqdm

root_freq = 261.6
# root_freq = 440

y, sr = librosa.load("recordings/download.wav", sr=44100, offset=0.6, duration = 1)

# si.write('mixed.wav', 44100, mix_scale)
sample_no = list(range(len(y)))

def grw(log_target, u0, data, K, G, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    N= u0.shape[0]

    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)
    K_inverse = Kc_inverse.T @ Kc_inverse # Compute the inverse of K using its Cholesky decomopsition

    lt_prev = log_target(u_prev, data, K_inverse, G)

    zs = np.random.randn(N, n_iters)

    for i in range(n_iters):

        z = zs[:, i]

        u_new = u_prev + beta*(Kc @ z) # Propose new sample - use prior covariance, scaled by beta

        lt_new = log_target(u_new, data, K_inverse, G)

        log_alpha = np.minimum(lt_new - lt_prev, 0)  # Calculate acceptance probability based on lt_prev, lt_new
    
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_alpha >= log_u # Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


def pcn(log_likelihood, u0, y, K, G, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0

    N = u0.shape[0]

    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    ll_prev = log_likelihood(u_prev, y, G)

    zs = np.random.randn(N, n_iters)

    for i in range(n_iters):
        z = zs[:, i]

        u_new = np.sqrt(1-beta**2)*u_prev + beta*(Kc @ z) #: Propose new sample using pCN proposal

        ll_new = log_likelihood(u_new, y, G)

        log_alpha = np.minimum(ll_new - ll_prev, 0) #: Calculate pCN acceptance probability
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = log_alpha >= log_u #: Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return X, acc / n_iters



def dmatrixofdmatrix(samples, windows, model_order):
    

    d_matrix = np.empty((len(samples), 3* 2 * len(windows) * model_order))
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
        d_matrix[t] = np.array([windows_vec])

    return d_matrix

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



def actual_function(x):
    return 1 + x - 1.4 * x ** 2 + 0.15 * x ** 3


def observation(x, sigma):
    return actual_function(x) + np.random.normal(0, sigma)


def lin_basis_function(x_points, degree=2):
    result = np.transpose(np.array([[x_points[0] ** i for i in range(degree)]]))
    for x in x_points[1:]:
        result = np.concatenate((result, np.transpose(np.array([[x ** i for i in range(degree)]]))), axis=1)
    return np.transpose(result)


def maximum_likelihood(x_points, y_points):
    matrix = lin_basis_function(x_points, degree=6)
    w = np.linalg.inv(np.transpose(matrix) @ matrix) @ np.transpose(matrix) @ y_points
    print(w)
    return w

def resynth(w, x):
    basis = [x ** i for i in range(len(w))]
    return np.dot(w,basis)


# priors(y,1)

windows = windowfunctions(sample_no, 5000)
dmat = dmatrix(sample_no, windows, 10)
print(dmat.shape)
# dmat = dmatrixofdmatrix(sample_no, windows, 10)
'''


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

