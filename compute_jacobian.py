import matplotlib.pyplot as plt
import jax.numpy as np
from jax import jacobian
from find_fixed_point import find_fixed_point
from numpy.linalg import eigvals
from hudgkin_huxley import hudgkin_huxley_eqs


# find the fixed point for different values of external current
const_params = [55, 40, -77, 35, -65, 0.3, 1]
I_ext_list = np.linspace(-0.2, 0.5, 200)
fixed_points = find_fixed_point(I_ext_list, const_params)

# compute the jacobian
formatted_eqs = lambda state: hudgkin_huxley_eqs(state, I_ext_list[0], const_params)
jac_hudgkin_huxley = jacobian(formatted_eqs)

# evaluate the jacobian at the fixed points
jac_at_fps = []
for i, fp in enumerate(fixed_points):
    jac_at_fps.append(np.array(jac_hudgkin_huxley(fp)))

# compute the eigenvalues of the jacobian matrix
eigenvalues = []
for j in jac_at_fps:
    eigenvalues.append(eigvals(j))
eigenvalues = np.array(eigenvalues)

# show results
plt.figure()
plt.plot(I_ext_list, eigenvalues)
plt.axhline(0, color='black', linewidth=0.5)
plt.show()
