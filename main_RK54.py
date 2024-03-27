import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.integrate import solve_ivp
from hudgkin_huxley import hudgkin_huxley_eqs


I_ext_list = np.arange(-0.4, 1.4, 0.2)
const_params = [55, 40, -77, 35, -65, 0.3, 1]
init_state = [-60, 0, 0, 0]
t_bounds = (0, 500)
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

plt.figure(figsize=(7, 4), dpi=300)
for i, I_ext in enumerate(I_ext_list[::-1]):
    plt.plot(time, V_I[-i-1], label= '$I_{ext} = $' + f'{I_ext:.1f}', color=sm.to_rgba(I_ext))
plt.xlabel('t [ms]')
plt.ylabel('V [mV]')
plt.xlim(t_bounds[0], t_bounds[1])
plt.ylim(-70, 40)
plt.subplots_adjust(right=0.7)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
plt.savefig('/Users/benja/Desktop/HH_V_pour_differents_I.png')
plt.show()
