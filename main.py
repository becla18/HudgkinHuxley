import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from hudgkin_huxley import hudgkin_huxley_euler_step


I_ext = 0.
alpha = 0.0001
time = np.arange(0, 100, alpha)
const_params = [55, 40, -77, 35, -65, 0.3, 1]
init_state = [-60, 0, 0, 0]

states = [init_state]
for _ in time[1:]:
    new_state = hudgkin_huxley_euler_step(states[-1], I_ext, const_params, alpha)
    states.append(new_state)

states = np.array(states)
V = states[:, 0]

plt.figure()
plt.plot(time, V)
plt.show()
