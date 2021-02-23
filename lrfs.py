import numpy as np
import matplotlib.pyplot as plt

X = np.asarray([54, 67, 79, 97, 113, 146, 179, 200, 231, 250]) 
y = np.asarray([1, 2, 5, 7, 13, 23, 39, 46, 61, 80]) 

X_mean = np.mean(X)
y_mean = np.mean(y)

n = len(X)

num = 0
denom = 0

for i in range(n):
	num += (X[i] - X_mean)*(y[i] - y_mean)
	denom += (X[i] - X_mean)**2

m = num / denom
c = y_mean - (m*X_mean)

# --------------------

def gradientDescent(m , c, lr, y, x, n, epochs):
	for i in range(epochs):
		y_pred = m*x + c
		dm = (-2/n)* sum(X*(y-y_pred))
		dc = (-2/n)* sum(y-y_pred)

		m = m - lr*dm
		c = c - lr*dc

		loss = (1/n)*sum(y-y_pred)**2
		print(loss)

epochs = 9
L = 0.001

gradientDescent(m, c, L, y, X, n, epochs)
print("\n")

# ------------------------------------------------------------
x_max = np.max(X) + 100
x_min = np.min(X) - 100

x = np.linspace(x_min, x_max, 1000)
y_ = c + (m * x)

plt.plot(x, y_, color='#00ff00', label='Linear Regression')
plt.scatter(X, y, color='#ff0000', label='Data Point')
plt.show()
