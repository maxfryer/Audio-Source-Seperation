import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal


def generate_S(D, SNR, R, I):
    a = D.T @ D + (1/SNR) * np.eye(2*R*(I+1))
    return np.linalg.inv(a)

def generate_P(D, S, N):
    return np.eye(N) - D @ S @ D.T

def generate_D(K, M, w):
    return



###--- Density functions ---###

def log_likelihood(N, noise_var, y, D, B):
    return -0.5*N*np.log(noise_var) - np.linalg.norm(y - D @ B)


def log_prior(B, M, K, w, noise_var):
    # structure of prior is
    # p(B|w,M,K,noise_var) x p(w|M,K) x p(M|K)
    # x p(K) x p(noise_var)
    return log_B(M, K, w, noise_var) + log_w(M, K) \
        + log_M(K) + log_K() + log_noise_var()

def log_B(B, M, K, w, noise_var):

    return 0

def log_w(w_est, K, M, sigma):
    # P(w_est|K, M, sigma)
    n  = np.sum(M)
    cov = sigma * np.eye(n)
    w = np.zeros(n)

    # Generate expected w given K and M (i.e perfect harmonics)
    sum = 0
    for i, order in enumerate(M):
        for partial in range(order):
            w[sum] = (partial+1) * K[i]
            sum+=1
    
    # Calculate Error and multivariate prob of error
    error = np.array(w - w_est)
    prob = multivariate_normal(mean=np.zeros(n), cov=cov).pdf(error)
    return prob

def log_M(M, K):
    return 0

def log_K(K):
    return 0

def log_noise_var(noise_var):
    return 0




if __name__ == '__main__':
    w_est = np.array([2,4,6,3,6,9,12])
    K = np.array([2,3])
    M = np.array([3,4])

    prob = log_w(w_est,K,M,0.00001)
    print(prob)
    