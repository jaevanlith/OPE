import numpy as np
from scipy.optimize import fsolve
from ope.envs.graph import Graph


# Construct matrix P
def p_pi(env: Graph, p):
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

# Compute stationary state distribution
def d_pi(env: Graph, p, gamma=0.98):
    # Construct matrix P
    P = p_pi(env, p)

    # Set initial state distribution
    s0 = np.zeros(len(P))
    s0[0] = 1

    # Compute stationary state distr
    # NOTE: TRANSPOSE OF P TO TRY
    d = (1 - gamma) * np.linalg.inv(np.eye(len(P)) - gamma * np.transpose(P)) @ s0
    # Normalize
    d /= np.sum(d)

    return d

# Compute KL divergence between two policies
def kl_divergence(env: Graph, p, q, gamma=0.98):
    delta = 1e-15
    
    # Get stationary state distribution of both policies
    d_b = d_pi(env, q, gamma)
    d_e = d_pi(env, p, gamma)
    
    d_e += delta
    
    # Compute KL divergence
    kl = np.sum(d_b * np.log(np.divide(d_b, d_e) + delta))

    return kl

# Equation to solve
def func(p, q, target_kl):
    # Compute KL
    kl = kl_divergence(env, p, q)

    # Subtract from wanted KL
    return target_kl - kl

# Get policy corresponding to wanted KL divergence
def get_evaluation_policy(kl_target, q, p_guess):
    p = fsolve(func, p_guess, (q, kl_target))
    return p[0]

# TEST
# Init environment
env = Graph(make_pomdp=False,
            number_of_pomdp_states=2,
            transitions_deterministic=True,
            max_length=3,
            sparse_rewards=False,
            stochastic_rewards=False)

# Init params
kl_target = 0
q = 1
p_guess = 0

# Run final method
p = get_evaluation_policy(kl_target, q, p_guess)
print(round(p,3))

# Run state-visitation
print('P_pi for p=1:')
print(p_pi(env, 1))
print('\n')

print('d_pi for p=1:')
print(d_pi(env, 1))
print('\n')

print('P_pi for p=0:')
print(p_pi(env, 0))
print('\n')

print('d_pi for p=0:')
print(d_pi(env, 0))
print('\n')