import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


y_data = np.array([8.828, 146.975, 4072.334, 18888])
x_data = np.array([2, 3, 4, 5])

y_data = np.log(y_data)

print(y_data)

X2 = sm.add_constant(x_data)
est = sm.OLS(y_data, X2)
results = est.fit()

print(results.summary())

a = results.params[0]
b = results.params[1]

min_x = np.floor(np.min(0)) + 0.001
max_x = np.ceil(np.max(7))+0.2
x = np.linspace(min_x, max_x, num=150)
# scatter-plot data
plt.plot(x_data, y_data, "bo")
# plot regression line on the same axes, set x-axis limits
plt.plot(x, a + b * x, color="r")
plt.grid()
plt.show()
