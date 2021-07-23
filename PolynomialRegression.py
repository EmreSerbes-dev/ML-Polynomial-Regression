import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

linReg = LinearRegression()
linReg.fit(x, y)

polyReg = PolynomialFeatures(degree = 4)
xPoly = polyReg.fit_transform(x)
linReg2 = LinearRegression()
linReg2.fit(xPoly, y)

#For Linear model
plt.scatter(x, y, color = 'red')
plt.plot(x, linReg.predict(x), color = 'blue')
plt.title("True or false old salary")
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape((len(xGrid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(xGrid, linReg2.predict(polyReg.fit_transform(xGrid)), color = 'blue')
plt.title("True or false old salary")
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()