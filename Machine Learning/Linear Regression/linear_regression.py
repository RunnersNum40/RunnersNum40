import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    """Linear Regression Model"""
    def __init__(self):
        pass

    def fit(self, x, y):
        # Add a 1 to each feature
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        b = np.ones((x.shape[0], 1))
        x = np.concatenate((b, x), axis=1)
        # Find the minimum of a least squares error
        self.w = np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))

    def predict(self, x):
        x = np.insert(x, 0, 1)
        return np.dot(self.w, x)

class PolynomialRegression:
    def __init__(self):
        pass

    def fit(self, x, y, max_power=10):
        # Creatate a train/test split to find the optimal power
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.66666)
        # Find the regression with best results on the training data
        trials = [(i, *self._test_power(x_train, x_test, y_train, y_test, i)) for i in range(1, max_power+1)]
        optimal = min(trials, key=lambda l: l[-1])
        #Store the best fitted model
        self.power = optimal[0]
        self.LRM = optimal[1]
        self.w = tuple(self.LRM.w)

    def _test_power(self, x_train, x_test, y_train, y_test, power):
        # Create an array with feature extraction
        x_train = np.array([x_train**n for n in range(1, power+1)]).T
        LRM = LinearRegression()
        LRM.fit(x_train, y_train)
        # Create the test results
        x_test = np.array([x_test**n for n in range(1, power+1)]).T
        y_pred = np.array([LRM.predict(item) for item in x_test])

        return LRM, np.sum(np.square(y_pred-y_test))

    def predict(self, x):
        return sum(w_i*x**n for n, w_i in enumerate(self.w))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    power = np.random.randint(5)+1
    n = 100
    # Training Data
    w = tuple(np.random.randn(power+1))
    x = np.random.rand(n).reshape((n))*100-50
    y = np.dot(w, np.array([x**n for n in range(power+1)]))+np.random.normal(0, 1, n)
    # Preform the fit
    PRM = PolynomialRegression()
    PRM.fit(x, y)
    # Visualization of the fit
    x_sample = np.linspace(np.min(x), np.max(x), 2*n)
    y_sample = np.array([PRM.predict(x_i) for x_i in x_sample])
    print(power, PRM.power)
    print(w, PRM.w)
    plt.scatter(x, y, label="Data")
    plt.plot(x_sample, y_sample, c="red", label="Fit")
    plt.legend(loc="upper right")
    plt.show()
