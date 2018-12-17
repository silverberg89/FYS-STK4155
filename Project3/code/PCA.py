import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_table('US_Election.csv', sep = ',', header = 0, encoding = 'latin1')

X = np.array(data.iloc[:, 12:61]) 
#X = X[:, list(range(3, 19)) + [22, 23, 29, 40]]
m, n = data.shape
scaler = StandardScaler()
X = scaler.fit_transform(X);
U, S, V = np.linalg.svd(X)
plt.plot(range(len(S)), S)


