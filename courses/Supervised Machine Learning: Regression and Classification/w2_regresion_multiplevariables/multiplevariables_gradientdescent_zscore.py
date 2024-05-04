import copy, math
import numpy as np
import matplotlib.pyplot as plt

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return X_norm, mu, sigma

def compute_cost(X, y, w, b, lambda_ = 0): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m, n = X.shape
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b     #(n,)(n,) = scalar (see np.dot)
        cost += (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                #scalar    

    reg_cost = 0
    if lambda_ > 0:
      for j in range(n):
        reg_cost += (w[j]**2)
      reg_cost  = reg_cost * (lambda_/2*m)

    return cost + reg_cost

def compute_gradient(X, y, w, b, lambda_ = 0): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape        
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] += err * X[i, j]  

        dj_db = dj_db + err   

    dj_dw = dj_dw / m                                
    dj_db = dj_db / m    

    if lambda_ > 0:
      for j in range(n):
          dj_dw[j] = dj_dw[j] + ((lambda_/m) * w[j])                            
        
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(X, y, w, b)  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw              
        b = b - alpha * dj_db              
      
        cost = cost_function(X, y, w, b)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")
        
    return w, b

# reduced display precision on numpy arrays
np.set_printoptions(precision=2)

# Load our data set
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Normalize features with zscore
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.

# some gradient descent settings
iterations = 1000
alpha = 1.0e-1

# run gradient descent 
w_norm, b_norm = gradient_descent(X_norm, y_train, initial_w, initial_b,
                                    compute_cost, compute_gradient, 
                                    alpha, iterations)

print(f"b,w found by gradient descent: {b_norm:0.2f},{w_norm} ")

m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_norm[i], w_norm) + b_norm:0.2f}, target value: {y_train[i]}")

x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")