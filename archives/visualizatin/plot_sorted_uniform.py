import matplotlib.pyplot as plt
import numpy as np

for _ in range(50):
    # p_true = np.random.rand(6)
    p_true = np.random.normal(0.5, 0.5, 6)
    p_true.sort()
    plt.plot(np.arange(1,7), p_true)
plt.show()
