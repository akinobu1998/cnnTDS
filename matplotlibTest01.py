import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

plt.plot(x, x, label='01')
plt.plot(x, x**2, label='02')
plt.plot(x, x**3, label='03')

plt.xlabel('x')
plt.ylabel('y')

plt.title('title')

plt.legend()

plt.show()
