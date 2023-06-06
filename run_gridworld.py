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

#from ope.experiment_tools.experiment import ExperimentRunner, analysis
#from ope.experiment_tools.config import Config
#from ope.experiment_tools.factory import setup_params

#----------------------------------------------
env = Gridworld(slippage=0.2)
eval_policy = 0.2
base_policy = 0.8
gamma = 0.98

#----------------------------------------------
action_size = env.n_actions
state_size = env.n_dim

transition_matrix = np.zeros((state_size, action_size, state_size)) # T -> SxAxS

for state in range(state_size):
    for action in range(action_size):
        for transition in env.T(state, action, use_slippage=True):
            prob, next_state = transition
            transition_matrix[state, action, next_state] += prob


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

# processor processes the state for storage,  {(processor(x), a, r, processor(x'), done)}
processor = lambda x: x

# Set up e-greedy policy using epsilon-optimal
policy = env.best_policy()

# absorbing state for padding if episode ends before horizon is reached. This is environment dependent.
absorbing_state = processor(np.array([len(policy)]))


def kl_divergence(b, e):

    pi_b = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=b, action_space_dim=env.n_actions)
    pi_e = EGreedyPolicy(model=TabularPolicy(policy, absorbing=absorbing_state), processor=from_grid, prob_deviation=e, action_space_dim=env.n_actions)

    policy_b = np.zeros((state_size, action_size))
    policy_e = np.zeros((state_size, action_size))

    for i in range(state_size):
        policy_b[i, :] = pi_b.predict(np.array([i]))

    for i in range(state_size):
        policy_e[i, :] = pi_e.predict(np.array([i]))

    P_policy_b = np.sum(policy_b[:, :, None] * transition_matrix, axis=1)
    P_policy_e = np.sum(policy_e[:, :, None] * transition_matrix, axis=1)

    # Define the initial state distribution (assuming always starting from the first state)
    s0 = np.zeros(state_size)
    s0[0] = 1.0  # start from the first state

    d = (1 - gamma) * np.linalg.inv(np.eye(state_size) - gamma * np.transpose(P_policy_b)) @ s0
    de = (1 - gamma) * np.linalg.inv(np.eye(state_size) - gamma * np.transpose(P_policy_e)) @ s0

    d = d[0:64]
    de = de[0:64]

    return kl_div(d,de).sum()

# Get max KL value
def get_kl_max(q_fixed, n, path):
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
            kl[i,j] = kl_divergence(pv[i,j], qv[i,j])

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
    plt.show()
    # Check if the directory exists
    if not os.path.exists(path):
        # If it doesn't exist, create it
        os.makedirs(path)

    # Save the plot
    fig.savefig(path + "KL_bounds_graph_plot.png")
    
    # Return maximum KL
    max_kl = max([kl_divergence(0, q_fixed), kl_divergence(1, q_fixed)])
    return max_kl

print(get_kl_max(q_fixed= 0.1, n = 50, path="home/mayank/Documents/IDM_project"))




# policy_b = policy_b[0:64]
# # Reshape it to a grid shape (assuming it is 8x8)
# policy_b = policy_b.reshape(8, 8, 4)

# # Prepare a grid
# X, Y = np.meshgrid(np.arange(0.5, 8, 1), np.arange(0.5, 8, 1))

# # Compute actions for each point of the grid
# actions = np.argmax(policy, axis=-1)

# # Create a mapping from actions to arrow directions (assuming the actions order is [up, right, down, left])
# arrow_mapping = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

# # Get arrow directions
# U, V = np.zeros_like(actions), np.zeros_like(actions)
# for action, (dx, dy) in arrow_mapping.items():
#     U[actions == action] = dy
#     V[actions == action] = dx  # Negate to make the plot origin at the top left, like a matrix
# print(U)
# print(V)
# # Create the plot
# plt.figure(figsize=(8, 8))
# plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
# plt.xlim(-1, 8)
# plt.ylim(-1, 8)
# plt.xticks(np.arange(0, 8, 1))
# plt.yticks(np.arange(0, 8, 1))
# plt.grid(visible=True)
# plt.gca().invert_yaxis()  # To align the (0,0) with top left
# plt.show()

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



