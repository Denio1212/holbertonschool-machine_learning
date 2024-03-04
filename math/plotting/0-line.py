# Placeholder
import matplotlib.pyplot as plt
import numpy as np

x_points = np.arange(0, 11)

y = np.arange(0, 11) ** 3
plt.plot(x_points, y,  color = 'r')
plt.title("A line plot")
plt.show()