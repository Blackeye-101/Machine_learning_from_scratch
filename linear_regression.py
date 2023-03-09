#implementing linear regression from scratch
import numpy as np

class LinearRegression:

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
            y_predicted=np.dot(X,self.weights)+self.bias
            dw=(2/n_samples)*np.dot(X.T,(y_predicted - y))
            db=(2/n_samples)*np.sum(y_predicted-y)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db

    def predict(self,X):
        y_predicted=np.dot(X,self.weights)+self.bias
        return y_predicted
        

