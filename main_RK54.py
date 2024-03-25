import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.integrate import solve_ivp
from hudgkin_huxley import hudgkin_huxley_eqs


I_ext_list = np.arange(-0.4, 1.4, 0.2)
const_params = [55, 40, -77, 35, -65, 0.3, 1]
init_state = [-60, 0, 0, 0]
t_bounds = (0, 100)
time = np.linspace(t_bounds[0], t_bounds[1], 1000)

V_I = []

for I_ext in I_ext_list:
    def hudgkin_huxley_scipy_format(time, state):
        return hudgkin_huxley_eqs(state, I_ext, const_params)

    solution = solve_ivp(hudgkin_huxley_scipy_format, t_bounds, init_state, method='RK45', t_eval=time)

    time, states = (solution.t, solution.y)
    V_I.append(states[0])

cmap = plt.cm.plasma
norm = colors.Normalize(vmin=I_ext_list[0], vmax=I_ext_list[-1] + 0.2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.figure()
for i, I_ext in enumerate(I_ext_list):
    plt.plot(time, V_I[i], label=f'I = {I_ext:.1f}', color=sm.to_rgba(I_ext))
plt.xlabel('t [ms]')
plt.ylabel('V [mV]')
plt.legend()
plt.show()
