import numpy as np
import copy
import math

NUM_FEATURES = 12

def load_data(file, n):
    data = np.genfromtxt(file, delimiter=';', skip_header=True)
    X = data[:,:n-1]
    y = data[:,n-1]
    return X, y

def zScoreNormalization(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma      

    return X_norm, mu, sigma

def calculateCost(X, y, w, b, lambda_ = 0):
    '''
    m,n = X.shape
    cost = 0

    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b
        cost += (f_wb_i - y[i]) ** 2

    cost = cost /(2*m)

    # Regularization
    reg_cost = 0
    if lambda_ > 0:
        for j in range(n):
            reg_cost += w[j] ** 2
        reg_cost = reg_cost * (lambda_ / (2 * m))

    return cost + reg_cost
    '''

    m = len(y)
    predictions = np.dot(X, w) + b
    error = predictions - y
    cost = np.sum(error ** 2) / (2 * m)
    reg_cost = lambda_ * np.sum(w ** 2) / (2 * m)
    return cost + reg_cost

def calculateGradient(X, y, w, b, lambda_ = 0):
    '''
    m,n = X.shape
    dj_dw = np.zeros_like(X[0])
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i],w)+b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i,j]
        dj_db += err

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    # Regularization
    if lambda_ > 0:
        for j in range(n):
            dj_dw[j] += w[j] * (lambda_/m)

    return dj_dw, dj_db
    '''

    m = len(y)
    predictions = np.dot(X, w) + b
    error = predictions - y
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m
    dj_dw = dj_dw + (lambda_ / m) * w
    return dj_dw, dj_db

def descentGradient(X, y, w_init, b_init, alpha, iter, lambda_):
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(iter):
        dj_dw, dj_db = calculateGradient(X, y, w, b, lambda_)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = calculateCost(X, y, w, b)
        if i% math.ceil(iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")

    return w, b

X_train, y_train = load_data('winequality-white.csv', NUM_FEATURES)

# Print X and y shape
print(f"X.shape: {X_train.shape} y.shape: {y_train.shape}")

print(f"Peak to Peak range by column in Raw X: {np.ptp(X_train,axis=0)}")
X_norm, x_mu, x_sigma = zScoreNormalization(X_train)
print(f"Peak to Peak range by column in Normalized X: {np.ptp(X_norm,axis=0)}")

w_init = np.zeros_like(X_train[0])
b_init = 0
alpha = 1.0e-1
iter = 100
lambda_ = 0

w, b = descentGradient(X_norm, y_train, w_init, b_init, alpha, iter, lambda_)