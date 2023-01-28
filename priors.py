import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, poisson, invgamma


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

def log_B(B, noise_var):
    # p(B|w,M,K,noise_var) is a zero mean multivariate Gaussian
    n = len(B)
    cov = noise_var * np.eye(n)
    prob = multivariate_normal(mean=np.zeros(n), cov=cov).pdf(B)
    return prob

def prior_w(ws, K, M, sigma):
    # P(w_est|K, M, sigma)
    w_est = ws.flatten()
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

def prior_M(M, mu=10):
    # p(M|K)
    prob=1
    # multiplicative sum of poisson likelihoods
    p = lambda k: poisson.pmf(k,mu)
    for order in M:
        prob *= p(order)
    return prob

def prior_K(K, theorectic_K, sigma=2):
    # P(K|K_theoretic, sigma)
    n  = K.shape[0]
    cov = sigma * np.eye(n)
    w = np.zeros(n)
    
    # Calculate Error and multivariate prob of error
    error = np.array(theorectic_K - K)
    prob = multivariate_normal(mean=np.zeros(n), cov=cov).pdf(error)
    return prob

def prior_noise_var(noise_var,mean):
    return invgamma.pdf(noise_var, 1/mean)


if __name__ == '__main__':
    w_est = np.array([2,4,6,3,6,9,12])
    K = np.array([2,3])
    M = np.array([3,4])

    prob = prior_w(w_est,K,M,sigma=0.4)
    print(prob)
    
    prob = prior_noise_var(0.01,0.01)
    print(prob)