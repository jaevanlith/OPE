import numpy as np
import json
from copy import deepcopy
from ope.envs.gridworld import Gridworld
from ope.policies.basics import BasicPolicy
from ope.policies.epsilon_greedy_policy import EGreedyPolicy
from ope.policies.tabular_model import TabularPolicy
from scipy.special import kl_div
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import os

#----------------------------------------------
#env = Gridworld(slippage=0.2)
#eval_policy = 0.2
#base_policy = 0.8
#gamma = 0.98
#----------------------------------------------
# to_grid and from_grid are particular to Gridworld
# These functions are special to convert an index in a grid to an 'image'
def to_grid(x, gridsize=[8, 8]):
    x = x.reshape(-1)
    x = x[0]
    out = np.zeros(gridsize)
    if x >= 64:
        return out
    else:
        out[x//gridsize[0], x%gridsize[1]] = 1.
    return out

# This function takes an 'image' and returns the position in the grid
def from_grid(x, gridsize=[8, 8]):
    if len(x.shape) == 3:
        if np.sum(x) == 0:
            x = np.array([gridsize[0] * gridsize[1]])
        else:
            x = np.array([np.argmax(x.reshape(-1))])
    return x

def state_distribution(env, prob_deviate, gamma):

    action_size = env.n_actions
    state_size = env.n_dim

    transition_matrix = np.zeros((state_size, action_size, state_size)) # T -> SxAxS

    for state in range(state_size):
        for action in range(action_size):
            for transition in env.T(state, action, use_slippage=True):
                prob, next_state = transition
                transition_matrix[state, action, next_state] += prob
    
    # processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
    processor = lambda x: x

    # Set up e-greedy policy using epsilon-optimal (epsilon is 0.001)
    policy = env.best_policy()

    # absorbing state for padding if episode ends before horizon is reached. This is environment dependent.
    absorbing_state = processor(np.array([len(policy)]))

    poli = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=prob_deviate, action_space_dim=env.n_actions)
    
    policy_vec = np.zeros((env.n_dim, env.n_actions))

    for i in range(state_size):
        policy_vec[i, :] = poli.predict(np.array([i]))

    P_policy = np.sum(policy_vec[:, :, None] * transition_matrix, axis=1)

    s0 = np.zeros(state_size)
    s0[0] = 1.0  # start from the first state

    d = (1 - gamma) * np.linalg.inv(np.eye(state_size) - gamma * np.transpose(P_policy)) @ s0

    d = d[0:64]

    return d.sum(), poli, policy_vec, P_policy, d

def kl_divergence(env, b, e, gamma):

    d_b = state_distribution(env, b, gamma)[4]
    d_e = state_distribution(env, e, gamma)[4]

    return kl_div(d_b,d_e).sum()

# Get max KL value
def get_kl_max(env, b_fixed, n, gamma):
    # Init all the possible values for p
    e = np.linspace(0, 1, n)
    b = np.linspace(0, 1, n)
    bv, ev = np.meshgrid(b,e)
    # Init KL's
    kl = np.zeros((n,n))

    # Loop over all (p,q)-pairs
    for i in range(n):
        for j in range(n):
            # Compute and store KL
            kl[i,j] = kl_divergence(env, bv[i,j], ev[i,j], gamma)

    # Return maximum KL
    max_kl = max([kl_divergence(env, b_fixed, 0, gamma), kl_divergence(env, b_fixed, 1, gamma)])
    return max_kl, bv, ev, kl

def get_evaluation_policy(env, b_fixed, kl_target, n, gamma):
    # Init all the possible values for e
    e_values = np.linspace(0, 1, n)
    b_values = np.full(n, b_fixed)
    
    # Compute KL divergences for all e
    kl_values = [kl_divergence(env, b_values[i], e_values[i], gamma) for i in range(n)]
    
    #print('kl_target', kl_target)
    # Find the index of the e value that results in KL divergence closest to kl_target
    closest_index = np.argmin(np.abs(np.array(kl_values) - kl_target))
    
    # Return the corresponding e value
    return e_values[closest_index]

def get_evaluation_policies(env, p, n):
    # Construct all possible eval policies
    tot_steps = n*100
    qs = np.linspace(0, 1, tot_steps)

    # Store corresponding KL divergence for each eval policy
    qs_with_kl = []
    for q in qs:
        kl = kl_divergence(env, p, q, 0.98)
        qs_with_kl.append((q,kl))

    # Sort list
    qs_with_kl = sorted(qs_with_kl, key=lambda x: x[1])

    # Get the max KL divergence (last element)
    max_kl = qs_with_kl[tot_steps-1][1]

    # Init the target KL divergences
    target_kls = np.linspace(0, max_kl, n)

    # Init list to store results
    eval_policies = []
    # Set first element 
    eval_policies.append(qs_with_kl[0][0])
    print(eval_policies)
    # Keep track of target KL's with var j
    j = 1
    # Loop over all possible evaluation policies (with ascending KL)
    for i in range(0,tot_steps-1):
        # Get two consecutive policy-KL tuples
        (q0,kl0) = qs_with_kl[i]
        (q1,kl1) = qs_with_kl[i+1]
        # Check if target KL falls in between
        if target_kls[j] >= kl0 and target_kls[j] <= kl1:
            # Compute distance to candidates
            dist0 = abs(target_kls[j] - kl0)
            dist1 = abs(target_kls[j] - kl1)
            # Pick the closest one
            if dist0 < dist1:
                eval_policies.append(q0)
            else:
                eval_policies.append(q1)
            # Move to next target KL
            j += 1
            # Stop when out of bounds
            if j >= n:
                break
    
    return list(zip(eval_policies, target_kls))

def plot_kl_bounds(env, n, path):
    # Init all the possible values for p and q
    p = np.linspace(0, 1, n)
    q = np.linspace(0, 1, n)
    pv, qv = np.meshgrid(p,q)
    # Init KL's
    kl = np.zeros((n,n))

    # Loop over all (p,q)-pairs
    for i in range(n):
        for j in range(n):
            # Compute and store KL
            kl[i,j] = kl_divergence(env, pv[i,j], qv[i,j], 0.98)

    # Create and save plot
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

def plot(pv, qv, kl):
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
    plt.show()

    return 

#---------------------------------Plots below-------------------------------------#
# from mpl_toolkits.mplot3d import Axes3D
# # Create X, Y coordinates
# x = np.arange(int(np.sqrt(state_size)))
# y = np.arange(int(np.sqrt(state_size)))
# X, Y = np.meshgrid(x, y)
# Z = d.reshape((int(np.sqrt(state_size)), int(np.sqrt(state_size))))
# J = de.reshape((int(np.sqrt(state_size)), int(np.sqrt(state_size))))

# # Plot a 3D surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z)
# ax.plot_surface(X, Y, J)
# ax.set_zlim([0, 0.05])
# plt.show()

# fig, ax = plt.subplots()

# # Display an image on the axes
# cax = ax.imshow(env.grid, cmap='viridis')

# # Add a colorbar to a plot
# fig.colorbar(cax)

# # Show the plot
# plt.show()



