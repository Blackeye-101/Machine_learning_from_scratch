import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from Regression import LogisticRegression

bc=datasets.load_breast_cancer()
X,y=bc.data, bc.target

X_train, X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=1234)


regressor=LogisticRegression(lr=0.001, n_iterations=1000)
regressor.fit(X_train, y_train)
predictions=regressor.predict(X_test)

accuracy=np.sum(predictions==y_test)/len(y_test)
print(accuracy)
