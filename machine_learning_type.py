
############################## Gradiyent pastlash ##############################

import numpy as np
import matplotlib.pyplot as plt

# 1. Ma'lumotlarni yaratish
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 2. Normallashtirish
# Ma'lumotlarni normallashtirish, bu gradient pastlashni tezlashtiradi
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# 3. Parametrlarni boshlash
theta = np.random.randn(2, 1)  # Parametrlar (theta0 va theta1)
X_b = np.c_[np.ones((100, 1)), X_norm]  # Birlik vektori qo'shamiz

# 4. Qiymat funksiyasi (Loss Function)
def compute_loss(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    loss = (1/2*m) * np.sum((predictions - y) ** 2)
    return loss

# 5. Gradiyent pastlash (Gradient Descent)
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    loss_history = np.zeros(n_iterations)
    
    for iteration in range(n_iterations):
        gradients = 1/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
        loss_history[iteration] = compute_loss(X, y, theta)
    
    return theta, loss_history

# Parametrlar
learning_rate = 0.1
n_iterations = 1000

# Gradiyent pastlashni bajarish
theta_final, loss_history = gradient_descent(X_b, y, theta, learning_rate, n_iterations)

# Natijalarni chiqarish
print(f"Yakuniy theta: {theta_final.ravel()}")

# 6. O'qitish jarayonidagi qiymat funksiyasining o'zgarishi
plt.plot(loss_history)
plt.title('Loss Function over iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# 7. Chiziqli regressiya chizig'ini chizish
plt.scatter(X_norm, y)
plt.plot(X_norm, X_b.dot(theta_final), color='red')
plt.title('Data and Linear Regression Line')
plt.xlabel('Normalized X')
plt.ylabel('y')
plt.show()
