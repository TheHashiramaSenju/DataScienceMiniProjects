import numpy as np
import pandas as pd
data = np.array([[3, 7],
                 [-4, -6],
                 [1, -1],
                 [7, 8],
                 [-4, -1],
                 [-3, -7]])
dataframe = pd.DataFrame(data, columns=['x1', 'x2'])

dataframe.describe()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataScaled = scaler.fit_transform(dataframe)

## Approach 1
c1 = dataframe.x1
c2 = dataframe.x2
np.cov(c1, c2)

## Approach 2
covarianceMatrix = dataframe.T @ dataframe / 5
covarianceMatrix

## Approach 3
np.sum(c1 * c2)/5

eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)

## PC1 - Contains the maximum information of the original two features that you have
PC1 = dataframe @ eigenVectors[:, 0]
## PC2
PC2 = dataframe @ eigenVectors[:, 1]

