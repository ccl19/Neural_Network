import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("housing.csv").dropna()
data = pd.get_dummies(data, columns=['ocean_proximity'])
cols = data.columns.tolist()
print(cols)
cols = cols[:8] + cols[9:] + [cols[8]]
print(cols)
data = data[cols]
print(data.columns)
data.reset_index(inplace=True, drop=True)
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data)
np.savetxt('housing.txt', data)
