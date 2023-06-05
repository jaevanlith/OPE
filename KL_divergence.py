import numpy as np
from scipy.optimize import fsolve
from ope.envs.graph import Graph
from ope.envs.gridworld import Gridworld
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import os
import tensorflow as tf

# env = Graph(make_pomdp=False,
#             number_of_pomdp_states=2,
#             transitions_deterministic=False,
#             max_length=3,
#             sparse_rewards=False,
#             stochastic_rewards=False)

#print(env.calculate_transition_matrix().shape)

# Construct matrix P
# def p_pi(env: Graph, p):
#     # Check if deterministic
#     if env.transitions_deterministic:
#         slippage = 0
#     else:
#         slippage = env.slippage
    
#     # Get the numer of states and length of chains
#     n = env.num_states()    
#     L = int(n / 2)
    
#     # Init matrix
#     P = np.zeros((n,n))

#     # Set values: sum[a] { T(s'|s,a) * pi(a|s) }
#     sum_odd = slippage * (1-p) + (1-slippage) * p
#     sum_even = slippage * p + (1-slippage) * (1-p)

#     # Set values from x{0}
#     P[0,1] = sum_odd
#     P[0,2] = sum_even

#     # Set values for all intermediate states x{t}
#     for t in range(1, L-1):
#         P[(2*t), (2*t + 1)] = sum_odd
#         P[(2*t), (2*t + 2)] = sum_even
        
#         P[(2*t)-1, (2*t + 1)] = sum_odd
#         P[(2*t)-1, (2*t + 2)] = sum_even
    
#     # Set values for states ending up in x{abs}
#     P[n-3,n-1] = 1
#     P[n-2,n-1] = 1

#     return P

# Compute stationary state distribution
def d_pi(env: Graph, p, gamma=0.98):
    # Construct matrix P
    #P = p_pi(env, p)
    state_size = env.n_dim
    action_size = env.n_actions

    transition_matrix = env.calculate_transition_matrix()
    print(transition_matrix)
    policy = np.zeros((state_size, action_size))
    
    policy[:,0] = p
    policy[:,1] = 1 - p
    

    P = np.sum(policy[:, :, None] * transition_matrix, axis=1)
    #print('P', P)
    # Set initial state distribution
    s0 = np.zeros(len(P))
    s0[0] = 1

    # Compute stationary state distr
    # NOTE: TRANSPOSE OF P TO TRY
    d = (1 - gamma) * np.linalg.inv(np.eye(state_size) - gamma * np.transpose(P)) @ s0
    #print(d)
    # Normalize
    #d /= np.sum(d)
    #print(d)

    return d

#print('state_dist', (d_pi(env, p = 0.1)))

# Compute KL divergence between two policies
def kl_divergence(env: Graph, p, q, gamma=0.8):
    delta = 1e-15
    
    # Get stationary state distribution of both policies
    d_b = d_pi(env, q, gamma)
    d_e = d_pi(env, p, gamma)
    
    d_e += delta
    
    # Compute KL divergence
    kl = np.sum(d_b * np.log(np.divide(d_b, d_e) + delta))
    #kl = np.sum(np.where(d_b !=0, d_b*np.log(d_b/d_e), 0))

    return kl

#print("printing kl divergence", kl_divergence(env, p = 0.1, q = 0.9))

# Equation to solve
def func(p, env: Graph, q, target_kl):
    # Compute KL
    kl = kl_divergence(env, p, q)

    # Subtract from wanted KL
    return target_kl - kl

# # Get policy corresponding to wanted KL divergence
# def get_evaluation_policy(env: Graph, kl_target, q, p_guess):
#     # Solve to find the wanted p-value
#     p = fsolve(func, p_guess, (env, q, kl_target))
#     return p[0]

def get_evaluation_policy(env: Graph, kl_target, p):

    num_actions = env.n_dim
    num_states = env.n_actions
    
    # Initialize q randomly
    q = tf.Variable(tf.random.uniform([num_states, num_actions]), dtype=tf.float32)

    # Define the loss function
    def loss():
        kl_divergence = tf.reduce_sum(p * tf.math.log(tf.maximum(p / q, 1e-12)))
        return tf.abs(kl_divergence - kl_target)

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam()

    # Perform optimization
    for _ in range(1000):  # You can change the number of iterations
        optimizer.minimize(loss, var_list=[q])

    # Normalize q to be a valid probability distribution
    q = q / tf.reduce_sum(q, axis=-1, keepdims=True)
    q = tf.clip_by_value(q, clip_value_min=1e-3, clip_value_max=1.0)
    q = q / tf.reduce_sum(q, axis=-1, keepdims=True)

    return q.numpy()

# Get max KL value
def get_kl_max(env: Graph, q_fixed, n, path):
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

    # Check if the directory exists
    if not os.path.exists(path):
        # If it doesn't exist, create it
        os.makedirs(path)

    # Save the plot
    fig.savefig(path + "KL_bounds_graph_plot.png")
    
    # Return maximum KL
    max_kl = max([kl_divergence(env, 0, q_fixed), kl_divergence(env, 1, q_fixed)])
    return max_kl

#print(get_kl_max(env, q_fixed= 0.1, n = 50, path="addyourpath"))
