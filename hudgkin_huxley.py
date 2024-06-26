import jax.numpy as np


# AUXILIARY EQUATIONS

def alpha_n(V):
    return 0.02 * (V - 25) / (1 - np.exp(- (V-25) / 9))

def alpha_m(V):
    return 0.182 * (V + 35) / (1 - np.exp(- (V+35) / 9))

def alpha_h(V):
    return 0.25 * np.exp(- (V+90) / 12)

def beta_n(V):
    return -0.002 * (V - 25) / (1 - np.exp((V-25) / 9))

def beta_m(V):
    return -0.124 * (V + 35) / (1 - np.exp((V+35) / 9))

def beta_h(V):
    return 0.25 * np.exp((V + 62)/6) / np.exp((V + 90) / 12)


# HUDGKIN-HUXLEY EQUATIONS

def dV_dt(V, n, m, h, I_ext, const_params):
    E_Na, g_Na, E_K, g_K, E_L, g_L, C = const_params
    return 1/C * (g_L * (E_L - V) + g_Na * m**3 * h * (E_Na - V) + g_K * n**4 * (E_K - V) + I_ext)

def dn_dt(V, n):
    return alpha_n(V) * (1 - n) - beta_n(V) * n

def dm_dt(V, m):
    return alpha_m(V) * (1 - m) - beta_m(V) * m

def dh_dt(V, h):
    return alpha_h(V) * (1 - h) - beta_h(V) * h

def hudgkin_huxley_eqs(state, I_ext, const_params):
    V, n, m, h = state
    V_dot = dV_dt(V, n, m, h, I_ext, const_params)
    n_dot = dn_dt(V, n)
    m_dot = dm_dt(V, m)
    h_dot = dh_dt(V, h)
    return [V_dot, n_dot, m_dot, h_dot]


# INTEGRATION FUNCTIONS

def hudgkin_huxley_euler_step(state, I_ext, const_params, alpha):
    V, n, m, h = state
    V_dot = dV_dt(V, n, m, h, I_ext, const_params)
    n_dot = dn_dt(V, n)
    m_dot = dm_dt(V, m)
    h_dot = dh_dt(V, h)
    return [V + alpha * V_dot, n + alpha * n_dot, m + alpha * m_dot, h + alpha * h_dot]

