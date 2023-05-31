import numpy as np
from scipy.optimize import fsolve
from ope.envs.graph import Graph
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D


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
def kl_divergence(env: Graph, p, q, gamma=0.8):
    delta = 1e-15
    
    # Get stationary state distribution of both policies
    d_b = d_pi(env, q, gamma)
    d_e = d_pi(env, p, gamma)
    
    d_e += delta
    
    # Compute KL divergence
    kl = np.sum(d_b * np.log(np.divide(d_b, d_e) + delta))

    return kl

# Equation to solve
def func(p, env: Graph, q, target_kl):
    # Compute KL
    kl = kl_divergence(env, p, q)

    # Subtract from wanted KL
    return target_kl - kl

# Get policy corresponding to wanted KL divergence
def get_evaluation_policy(env: Graph, kl_target, q, p_guess):
    # Solve to find the wanted p-value
    p = fsolve(func, p_guess, (env, q, kl_target))
    return p[0]

# Get max KL value
def get_kl_max(env: Graph, q_fixed, n):
    # Init all the possible values for p
    p = np.linspace(0, 1, n)
    q = np.linspace(0, 1, n)
    pv, qv = np.meshgrid(p,q)
    # Init KL's
    kl = np.zeros((n,n))

    # Loop over all (p,q)-pairs
    for i in range(n):
        for j in range(n):
            # Compute and store KL
            kl[i,j] = kl_divergence(env, pv[i,j], qv[i,j])

    # # Create and save plot
    fig, ax = plt.subplots( figsize=(10, 10), subplot_kw={'projection': '3d'})

    # Set different viewing angles for each subplot
    viewing_angles = [(30, 45), (60, 120), (90, 180)]

    # for i, ax in enumerate(axes):
    ax.plot_surface(pv,qv ,kl , cmap='viridis')
    ax.view_init(elev=viewing_angles[0][0], azim=viewing_angles[0][1])
    ax.set_xlabel('p', fontsize=17)
    ax.set_ylabel('q', fontsize=17)
    ax.set_zlabel('KL Divergence', fontsize=17)
    # ax.set_title('KL Divergence for different values of p and q')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    fig.savefig("KL_bounds_graph_plot.png")
    
    # Return maximum KL
    max_kl = max([kl_divergence(env, 0, q_fixed), kl_divergence(env, 1, q_fixed)])
    return max_kl
