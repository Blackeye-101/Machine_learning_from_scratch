#implementing linear regression from scratch
import numpy as np

class BaseRegression:
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr=lr
        self.n_iterations=n_iterations
        self.weights=None
        self.bias=None

    def fit(self, X,y):
        # init our parameters
        n_samples, n_features=X.shape

        self.weights=np.zeros(n_features)
        self.bias=0

        # gradient descent to find out the values of w and b
        for _ in range (self.n_iterations):
            y_predicted= self._approximation(X, self.weights, self.bias)
            dw=(2/n_samples)*np.dot(X.T,(y_predicted - y))
            db=(2/n_samples)*np.sum(y_predicted-y)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db

    def _approximation(self, X, w,b):
        raise NotImplementedError()
    
    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X,w,b):
        raise NotImplementedError()

class LinearRegression(BaseRegression):

    def _approximation(self, X, w, b):
        return np.dot(X,w)+b

    def _predict(self,X,w,b):
        return np.dot(X,w)+b


class LogisticRegression(BaseRegression):
   
    def _approximation(self, X,w,b):
        linear_model=np.dot(X,w)+b
        return self._sigmoid(linear_model)

    def _predict(self,X,w,b):
        linear_model=np.dot(X, w)+b
        y_predicted=self._sigmoid(linear_model)

        y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]

        return y_predicted_cls


    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))


