import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ope.envs.graph import Graph

# initialize environment
env = Graph(make_pomdp=False,
            number_of_pomdp_states=2,
            transitions_deterministic=False,
            max_length=3,
            sparse_rewards=False,
            stochastic_rewards=False)


# Construct matrix P
def p_matrix(env: Graph, p):
    # Check if deterministic
    if env.transitions_deterministic:
        slippage = 0
    else:
        slippage = env.slippage
    
    # Get the numer of states and length of chains
    n = env.num_states()    
    L = int(n / 2)
    
    # Init matrix
    P = np.zeros((n,n))

    # Set values: sum[a] { T(s'|s,a) * pi(a|s) }
    sum_odd = slippage * (1-p) + (1-slippage) * p
    sum_even = slippage * p + (1-slippage) * (1-p)

    # Set values from x{0}
    P[0,1] = sum_odd
    P[0,2] = sum_even

    # Set values for all intermediate states x{t}
    for t in range(1, L-1):
        P[(2*t), (2*t + 1)] = sum_odd
        P[(2*t), (2*t + 2)] = sum_even
        
        P[(2*t)-1, (2*t + 1)] = sum_odd
        P[(2*t)-1, (2*t + 2)] = sum_even
    
    # Set values for states ending up in x{abs}
    P[n-3,n-1] = 1
    P[n-2,n-1] = 1

    return P

def d_state_visitation(P, gamma):
    s0 = np.zeros(len(P))
    s0[0] = 1
    print(s0)

    d = (1 - gamma) * np.linalg.inv(np.eye(len(P)) - gamma * P) @ s0
    return d

P = p_matrix(env, 1)
d = d_state_visitation(P, 0.8)
print(d)



# k = 0.23
# nu = 10.0
# Fao = 1.0

# def ssd(s, p):
#     if (s % 2) == 0:
#         return 0,75 * (1-p) + 0,25 * p
#     else:
#         return 0,75 * p + 0,25 * (1-p)

# def integrand(Fa):
#     return 

# def func(Fa):
#     integral,err = quad(integrand, Fao, Fa)
#     return 100.0 - integral

# vfunc = np.vectorize(func)

# f = np.linspace(0.01, 1)
# plt.plot(f, vfunc(f))
# plt.xlabel('Molar flow rate')
# plt.savefig('integral-eqn-guess.png')
# plt.show()

# Fa_guess = 0.1
# Fa_exit, = fsolve(vfunc, Fa_guess)
# print(Fa_exit)