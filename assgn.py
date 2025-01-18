import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading data
X = pd.read_csv('linearX.csv').values
Y = pd.read_csv('linearY.csv').values

# normalised x
X = (X - np.mean(X)) / np.std(X)

# initialisation of parameters
theta = np.zeros((X.shape[1], 1))
theta0 = 0.0


# cost function defination
def compute_cost(X, Y, theta, theta0):
    m = len(Y)
    predictions = np.dot(X, theta) + theta0
    cost = np.sum((predictions - Y) ** 2) / (2 * m)
    return cost


# gradient descent function
def gradient_descent(X, Y, theta, theta0, lr, iterations):
    m = len(Y)
    cost_history = []

    for i in range(iterations):
        predictions = np.dot(X, theta) + theta0
        errors = predictions - Y
        theta_gradient = np.dot(X.T, errors) / m
        theta0_gradient = np.sum(errors) / m
        theta -= lr * theta_gradient
        theta0 -= lr * theta0_gradient
        cost = compute_cost(X, Y, theta, theta0)
        cost_history.append(cost)
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-8:
            break

    return theta, theta0, cost_history


lr = 0.0005 # learning rate
iterations = 5000  # Adjust as needed

theta_opt, theta0_opt, cost_history = gradient_descent(X, Y, theta, theta0, lr, iterations)

# Plot Cost Function
plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_history[:50])), cost_history[:50], marker='o')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function vs Iteration for lr = 0.5')
plt.grid(True)
plt.show()

# Regression Line
predictions = np.dot(X, theta_opt) + theta0_opt
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Normalized X')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
