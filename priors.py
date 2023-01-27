import numpy as np
import matplotlib.pyplot as plt



def generate_S(D, SNR, R, I):
    a = D.T @ D + (1/SNR) * np.eye(2*R*(I+1))
    return np.linalg.inv(a)

def generate_P(D, S, N):
    return np.eye(N) - D @ S @ D.T

def generate_D(K, M, w)



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

def log_w(M, K, w):
    return 0

def log_M(M, K):
    return 0

def log_K(K):
    return 0

def log_noise_var(noise_var):
    return 0