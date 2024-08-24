
########################### LinearRegression ###########################

import numpy as np

class LinearRegression:
    def __init__(self):
        self.theta = None
    
    def fit(self, X, y):
        # X to'plamiga birlik vektori qo'shamiz
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal tenglama orqali theta ni hisoblaymiz
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

# Ma'lumotlarni yaratish
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Model yaratish va o'qitish
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Bashoratlar qilish
y_pred = lin_reg.predict(X)

# Natijalarni chizish
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title('Chiziqli Regressiya (Scratch)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

############################# Polinomial Regressiya ##########################

class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        super().fit(X_poly, y)

    def predict(self, X):
        X_poly = self._polynomial_features(X)
        return super().predict(X_poly)

    def _polynomial_features(self, X):
        X_poly = X
        for i in range(2, self.degree + 1):
            X_poly = np.c_[X_poly, X**i]
        return X_poly

# Ma'lumotlarni yaratish
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X**2 + np.random.randn(100, 1)

# Model yaratish va o'qitish
poly_reg = PolynomialRegression(degree=2)
poly_reg.fit(X, y)

# Bashoratlar qilish
y_poly_pred = poly_reg.predict(X)

# Natijalarni chizish
plt.scatter(X, y, color='blue')
plt.plot(X, y_poly_pred, color='red', linewidth=2)
plt.title('Polinomial Regressiya (Scratch)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


############################## Logistik Regressiya #########################

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.random.randn(X_b.shape[1], 1)
        
        for _ in range(self.n_iterations):
            gradients = X_b.T.dot(self._sigmoid(X_b.dot(self.theta)) - y) / len(y)
            self.theta -= self.learning_rate * gradients
    
    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X_b.dot(self.theta))
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# Ma'lumotlarni yaratish
np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = (4 + 3 * X + np.random.randn(100, 1) > 4).astype(int)

# Model yaratish va o'qitish
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Bashoratlar qilish
y_pred = log_reg.predict(X)

# Natijalarni chizish
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.title('Logistik Regressiya (Scratch)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

################################ K-Nearest Neighbors ###############################

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict_one(x) for x in X]
        return np.array(y_pred)
    
    def _predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.argmax(np.bincount(k_nearest_labels))

# Ma'lumotlarni yaratish
np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Model yaratish va o'qitish
knn = KNearestNeighbors(k=3)
knn.fit(X, y)

# Bashoratlar qilish
y_pred = knn.predict(X)

# Natijalarni chizish
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('K-NN (Scratch)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

################################ Support Vektor Mashinasi ###############################

class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

# Ma'lumotlarni yaratish
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Model yaratish va o'qitish
svm = SupportVectorMachineScratch()
svm.fit(X, y)

# Bashoratlar qilish
y_pred = svm.predict(X)

# Natijalarni chizish
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('SVM (Scratch)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
