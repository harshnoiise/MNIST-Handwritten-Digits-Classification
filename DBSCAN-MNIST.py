from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import warnings
import itertools
import datetime as dt

import scipy
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import pylab
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
np.random.seed(42)

d = pd.read_csv("train.csv")
df = pd.DataFrame(data=d, )
df_elements = df.sample(n=10000)
df_elements_pix = df_elements.iloc[:, 0:785]
raw_data = scale(df_elements_pix)

n_samples, n_features = raw_data.shape
n_digits = len(np.unique(df_elements_pix.label))
labels = df_elements_pix.label

pca = PCA(n_components=n_digits).fit(raw_data)
reduced_data = PCA(n_components=2).fit_transform(raw_data)

# Step size of mesh
h = .02

# Plotting decision boundary
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

dbscan = DBSCAN()
clusters=dbscan.fit_predict(raw_data)
scaler=StandardScaler()
scaler.fit(reduced_data)
X_s=scaler.transform(reduced_data)
plt.figure()
clusters=dbscan.fit_predict(X_s)
plt.scatter(X_s[:,0], X_s[:,1], c=clusters, cmap="Paired", s=10)
plt.show()
lt.title('DBSCAN on MNIST Data set')
plt.savefig('DBSCAN-MNIST.png')
