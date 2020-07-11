import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset/Admission_predict.csv")

class MLR(object):

    def __init__(self):
        self.coefficients = []

    def reshape_input(self, X):
        return X.reshape(-1, 1)

    def concatenate_ones(self, X):
        ones = np.ones(shape = X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)

    def fit(self, X, Y):
        if len(X.shape) == 1: X = self.reshape_input(X)
        X = self.concatenate_ones(X)
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)

    def predict(self, entry):
        prediction = b0 = self.coefficients[0]
        other = self.coefficients[1:]
        for xi, bi in zip(entry, other):
            prediction += (bi*xi)
        return prediction

X = dataset.drop('Chance of Admit', axis=1).values
Y = dataset['Chance of Admit'].values

model = MLR()
model.fit(X, Y)

y_preds = []
for row in X: y_preds.append(model.predict(row))

result = pd.DataFrame({
    'Actual' :  Y,
    'Predicted' : np.ravel(y_preds)
})

def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += (predicted[i] - actual[i])**2
    mean_square_error = sum_error / float(len(actual))
    return np.sqrt(mean_square_error)

rmse_metric(result['Actual'], result['Predicted'])
