
import numpy as np
import csv
import pandas as pd
FILE_NAME="spambase.data"

#loading using only python
# with open(FILE_NAME, 'r') as f:
#     data= list(csv.reader(f, delimiter=','))
# data=np.array(data)

#using numpy
# method-1
# data=np.loadtxt(FILE_NAME, delimiter=',')

# method-2
# data=np.genfromtxt(FILE_NAME, delimiter=",", dtype=np.float32, skip_header=1, missing_values="hello", filling_values=0)
# print(data.shape)

# n_samples, n_features=data.shape
# n_features-=1

# X=data[:, 0:n_features]
# y=data[:, n_features]

# print(X.shape, y.shape)


#using pandas
#pandas tries to read a heaeder
df=pd.read_csv(FILE_NAME, header=None,delimiter=",",dtype=np.float32, skiprows=1, na_values=["hello"])
df=df.fillna(0)
#skip headers by using "skiprows"
data=df.to_numpy()
print(type(data[0][0]))

n_samples, n_features=data.shape
n_features-=1

X=data[:,0:n_features]
y=data[:,n_features]

print(X.shape, y.shape)

print(X[0,0:5])

