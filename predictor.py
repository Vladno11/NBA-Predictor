import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

data = pd.read_excel('nba_rezultati.xlsx')

teams = data['Tim']
games_data = data.drop(columns=['Tim'])

x = np.arange(1, 42).reshape(-1, 1)
y = games_data.values


degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

predictions = []

for i in range(y.shape[0]):
    y_team = y[i, :]
    model.fit(x, y_team)
    x_future = np.arange(42, 83).reshape(-1, 1)
    y_future_pred = model.predict(x_future)
    predictions.append(y_future_pred)

predictions = np.array(predictions)
pred_pobede = np.sum(predictions > 0, axis=1)

stvarne_pobede = np.sum(y > 0, axis=1)
ukupno = stvarne_pobede + pred_pobede

data['stvarne_pobede'] = stvarne_pobede
data['pred_pobede'] = pred_pobede
data['ukupno'] = ukupno

print(data[['Tim', 'stvarne_pobede', 'pred_pobede', 'ukupno']])



