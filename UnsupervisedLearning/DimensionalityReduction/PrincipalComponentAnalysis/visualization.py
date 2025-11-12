import pandas as pd
import numpy as np

data = pd.read_csv('/content/diabetes.csv')
data.head()
## descriptive statistics of a given data
data.describe()

## input features
data = data.drop(columns=['Outcome'], axis=1)
## convert the data into standard scaler form
## mean = 0 and standard deviation = 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataScaled = scaler.fit_transform(data)
dataScaled = pd.DataFrame(dataScaled)
dataScaled.head()

covarianceMatrix = dataScaled.T @ dataScaled / 767
covarianceMatrix

eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
print(eigenValues)

## PC1 data
PC1_data = dataScaled @ eigenVectors[:, 0]
## PC2 data
PC2_data = dataScaled @ eigenVectors[:, 1]
## PC3 data
PC3_data = dataScaled @ eigenVectors[:, 7]

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit_transform(dataScaled)

from sklearn.decomposition import PCA
pca = PCA()
principalComponent = pca.fit_transform(dataScaled)
pca.explained_variance_ratio_

import matplotlib.pyplot as plt
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance Ratio")
plt.savefig("ScreenPlot.png")
plt.show()