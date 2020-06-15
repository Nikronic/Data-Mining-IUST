#%% import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% reading data
data = pd.read_csv('data/data.csv', header=None)
print('Number of samples: ',len(data))
data.plot.scatter(0, 1, figsize=(15, 10))
plt.show()

data = data.values

#%% normalization
mean = data.mean()
std = data.std()
data = (data - mean) / std

#%% model definition


#%% test
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.3)
y_dbscan = dbscan.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=y_dbscan, s=50, cmap='viridis', label=np.unique(y_dbscan))
centers = dbscan.components_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.1)
plt.legend()
plt.show()


