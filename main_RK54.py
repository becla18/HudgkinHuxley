import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from hudgkin_huxley import hudgkin_huxley_eqs


I_ext = 0.
const_params = [55, 40, -77, 35, -65, 0.3, 1]
init_state = [-60, 0, 0, 0]
t_bounds = (0, 20)
time = np.linspace(0, 20, 1000)

def hudgkin_huxley_scipy_format(time, state):
    return hudgkin_huxley_eqs(state, I_ext, const_params)

solution = solve_ivp(hudgkin_huxley_scipy_format, t_bounds, init_state, method='RK45', t_eval=time)

time, states = (solution.t, solution.y)
V = states[0]

plt.figure()
plt.plot(time, V)
plt.show()
