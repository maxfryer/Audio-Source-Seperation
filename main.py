import string
import random
import numpy as np
import matplotlib.pyplot as plt


# from manual_test import *

# Following on from the IIA project handout, we formulate both the ML estimate,
# the Bayesian posterior density and Bayesian estimates for parameters for simple
# model y_n = c + e_n where c is a constant, e_n is gaussian iid noise and y_n
# are observations


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


x = np.linspace(1, 8, 10)
y = [observation(i, 1) for i in x]

w = maximum_likelihood(x,y)
y_sim = [resynth(w, i) for i in x]


plt.plot(x, y)
plt.plot(x, y_sim)
plt.show()

# matrix = lin_basis_function(x)
# print(matrix)


# Let us assume that the sound
