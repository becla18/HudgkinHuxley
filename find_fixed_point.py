from scipy.optimize import root
from hudgkin_huxley import hudgkin_huxley_eqs


def find_fixed_point(I_ext_list, const_params):
    # set I_ext values and initial optimization state
    init_optimization_state = [-50, 0, 0, 0]
    const_params = [55, 40, -77, 35, -65, 0.3, 1]

    # find fixed point for each I_ext
    fixed_points = []
    for I_ext in I_ext_list:
        args = (I_ext, const_params)
        solution = root(hudgkin_huxley_eqs, init_optimization_state, args)
        print('success:', solution.success)
        fixed_points.append(solution.x)

    return fixed_points

