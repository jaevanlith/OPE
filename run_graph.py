import numpy as np
import pickle
from matplotlib import pyplot as plt
from datetime import date
from KL_divergence import get_evaluation_policy, get_kl_max

from ope.envs.graph import Graph
from ope.models.basics import BasicPolicy
from ope.experiment_tools.experiment import ExperimentRunner, analysis
from ope.experiment_tools.config import Config
from ope.utils import make_seed
from neurips_seeds import weighted_graph_args, unweighted_graph_args
from neurips_plotting import neurips_plot, neurips_plot_kl_3D, neurips_plot_kl_2D
import json
import argparse
import os

def get_configuration(experiment_args, seed):
    # basic configuration with varying number of trajectories
    configuration = {
        "gamma": experiment_args["gamma"],
        "horizon": experiment_args["horizon"],
        "base_policy": experiment_args["p0"],
        "eval_policy": experiment_args["p1"],
        "stochastic_env": True,
        "stochastic_rewards": False,
        "sparse_rewards": False,
        "num_traj": max(experiment_args["Nvals"]),
        "Nvals": experiment_args["Nvals"], # Compute value of estimate for these data sizes
        "is_pomdp": False,
        "pomdp_horizon": 2,
        "seed": seed,
        "experiment_number": 0,
        "access": 0,
        "secret": 0,
        "modeltype": "tabular",
        "to_regress_pi_b": False,
        "nstep_int": 1,
        "weighted": experiment_args["weighted"],
        "nstep_custom_ns": experiment_args["nstep_custom_ns"]
    }

    # store them
    cfg = Config(configuration)

    return cfg

def run(experiment_args, kl_target):

    runner = ExperimentRunner()

    seeds  = []
    all_N_vals = []
    for t in range(experiment_args["num_trials"]): # Number of trials, config file for each trial.

        # Set random seed for trial.
        if experiment_args["seeds"] is None: seed = make_seed()
        else: seed = experiment_args["seeds"][t]
        seeds.append(seed)

        # For each Nval, we will compute the estimate for that value on the same dataset.
        all_N_vals += experiment_args["Nvals"]

        # Retrieve config
        cfg = get_configuration(experiment_args, seed)

        # initialize environment with this configuration
        env = Graph(make_pomdp=cfg.is_pomdp,
                    number_of_pomdp_states=cfg.pomdp_horizon,
                    transitions_deterministic=not cfg.stochastic_env,
                    max_length=cfg.horizon,
                    sparse_rewards=cfg.sparse_rewards,
                    stochastic_rewards=cfg.stochastic_rewards)

        # set seed for the experiment
        np.random.seed(cfg.seed)

        # processor processes the state for storage
        processor = lambda x: x

        # absorbing state for padding if episode ends before horizon is reached
        absorbing_state = processor(np.array([env.n_dim - 1]))

        # Calculated evaluation policy for the given kl_divergence
        cfg.eval_policy = get_evaluation_policy(env, kl_target, cfg.base_policy, 0)

        # Setup policies
        actions = [0, 1]
        pi_e = BasicPolicy(
            actions, [max(.001, cfg.eval_policy), 1 - max(.001, cfg.eval_policy)])
        pi_b = BasicPolicy(
            actions, [max(.001, cfg.base_policy), 1 - max(.001, cfg.base_policy)])

        # add env, policies, absorbing state and processor
        cfg.add({
            'env': env,
            'pi_e': pi_e,
            'pi_b': pi_b,
            'processor': processor,
            'absorbing_state': absorbing_state
        })

        # Decide which OPE methods to run.
        # Currently only all is available
        cfg.add({'models': experiment_args["models"]})

        # Add the configuration
        runner.add(cfg)

    # Run the configurations
    results = runner.run()

    return results, {'seeds':seeds, 'args': experiment_args}


def run_kl_experiment(experiment_args):
    results_all_behaviors = []

    # Run for all different behavior policies
    for pi_b in experiment_args["behavior_policies"]:
        print("Running behavior policy ", pi_b)
        # Construct environment
        cfg = get_configuration(experiment_args, 0)
        env = Graph(make_pomdp=cfg.is_pomdp,
                    number_of_pomdp_states=cfg.pomdp_horizon,
                    transitions_deterministic=not cfg.stochastic_env,
                    max_length=cfg.horizon,
                    sparse_rewards=cfg.sparse_rewards,
                    stochastic_rewards=cfg.stochastic_rewards)
        
        # Get bounds of KL divergence
        max_kl_divergence = get_kl_max(env, pi_b, 50, path=experiment_args["image_path"])
        kl_divergence = np.linspace(0, max_kl_divergence, 10)

        # Init results
        results = []

        # Run for all possible KL divergences
        for kl in kl_divergence:
            print("Running ", kl, " of ", kl_divergence)

            # Set the behavior policy
            experiment_args["p0"] = pi_b

            # Run experiments
            graph_results = run(experiment_args, kl)

            # Save results
            results.append({"behavior_policy":  pi_b,
                            "kl_divergence": kl,
                            "results": graph_results
                            })

        results_all_behaviors.append(results)

    return results_all_behaviors
    

# Set experiment arguments to provided settings
def customize_args(experiment_args, cmd_args):
    # Which models to run, call "SOPE" to only run one
    experiment_args["models"] = cmd_args["models"]

    # Set horizon
    experiment_args["horizon"] = cmd_args["horizon"]

    # Set amount of trials
    experiment_args["num_trials"] = cmd_args["num_trials"]

    # Set values for N
    experiment_args["Nvals"] = [int(N) for N in cmd_args["Nvals"].split(",")]

    # Set n to fixed value or run for all possible n's if set to None
    if cmd_args["fix_n_value"] is not None:
        if cmd_args["fix_n_value"] <=  experiment_args["horizon"]:
            experiment_args["nstep_custom_ns"] = [cmd_args["fix_n_value"]]
    else:
        experiment_args["nstep_custom_ns"] = None

    # Set behavior policies
    experiment_args["behavior_policies"] = [float(pi_b) for pi_b in cmd_args["behavior_policies"].split(",")]

    # Set image path
    experiment_args["image_path"] = cmd_args["image_path"]

    return experiment_args


# Save date to files
def save_results(unweighted_results, weighted_results, path):

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        raise TypeError('Not serializable ', obj)

    # Check if the directory exists
    if not os.path.exists(path):
        # If it doesn't exist, create it
        os.makedirs(path)

    with open(path + "weighted_results.json", "w") as weighted:
            json.dump(weighted_results, weighted, indent = 4, default=default)
    weighted.close()

    with open(path + "unweighted_results.json", "w") as unweighted:
            json.dump(unweighted_results, unweighted, indent=4, default=default)
    unweighted.close()

def plot_results(cmd_args, fixed_n_value=None):
    path = cmd_args['image_path']

    if not os.path.exists(path + "weighted/"):
        os.makedirs(path + "weighted/")

    if not os.path.exists(path + "unweighted/"):
        os.makedirs(path + "unweighted/")

    # When results are for all n, make 3D plot
    if fixed_n_value is None:
        for i in weighted_graph_results:
            neurips_plot_kl_3D(raw_results=i, 
                               filename=path + "weighted/bp=" + str(i[0]["behavior_policy"]).replace(".", "") + "_graph_plot_kl_3D", 
                               weighted=True,  
                               cycler_small=True)
        for j in unweighted_graph_results:    
            neurips_plot_kl_3D(raw_results=j,
                               filename=path + "unweighted/bp=" + str(j[0]["behavior_policy"]).replace(".", "") + "_graph_plot_kl_3D", 
                               weighted=False, 
                               cycler_small=True)
            
    # If n was fixed, make 2D plot
    else:
        for i in weighted_graph_results:
            neurips_plot_kl_2D(raw_results=i, 
                               filename=path + "weighted/bp=" + str(i[0]["behavior_policy"]).replace(".", "") + "_n="+ str(cmd_args['fix_n_value']) + "_graph_plot_kl_2D", 
                               weighted=True,  
                               cycler_small=True, 
                               fixed_n_value=fixed_n_value)
        for j in unweighted_graph_results:
            neurips_plot_kl_2D(raw_results=j,
                               filename=path + "unweighted/bp=" + str(j[0]["behavior_policy"]).replace(".", "") + "_n="+ str(cmd_args['fix_n_value']) + "_graph_plot_kl_2D", 
                               weighted=False, 
                               cycler_small=True, 
                               fixed_n_value=fixed_n_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph experiment')
    # Models to run, call "SOPE" to only run one
    parser.add_argument('--models', type=str, default="n-step")
    # Horizon of the experiment
    parser.add_argument('--horizon', type=int, default=20)
    # Number of trials to run per experiment  
    parser.add_argument('--num_trials', type=int, default=100)
    # Values for N
    parser.add_argument('--Nvals', type=str, default="256,512,1024,2048,4096,8192")
    # To run for a fixed n, set this to the value of n 
    parser.add_argument('--fix_n_value', type=int, default=None)
    # Behavior policies to run
    parser.add_argument('--behavior_policies', type=str, default="0.5,0.8")
    # Path to which new images are saved
    parser.add_argument('--image_path', type=str, default="./experiment_images/")
    # Path to which new data is saved
    parser.add_argument('--save_path', type=str, default="./experiment_results/")
    # If data is already generated, load it instead of generating it
    parser.add_argument('--load_data_path', type=str, default=None)
    cmd_args = dict(vars(parser.parse_args()))

    if cmd_args["load_data_path"] is None:
        # Customize experiment arguments
        unweighted_graph_args = customize_args(unweighted_graph_args, cmd_args)
        weighted_graph_args = customize_args(weighted_graph_args, cmd_args)

        # Run trials
        unweighted_graph_results = run_kl_experiment(unweighted_graph_args)
        weighted_graph_results = run_kl_experiment(weighted_graph_args)

        # Save results
        save_results(unweighted_graph_results, weighted_graph_results, cmd_args["save_path"])
    else:
        # Load data when already there
        unweighted_graph_results = json.load(open(cmd_args["load_data_path"] + "unweighted_results.json"))
        weighted_graph_results = json.load(open(cmd_args["load_data_path"] + "weighted_results.json"))

    # Plot Results
    plot_results(cmd_args, fixed_n_value=cmd_args["fix_n_value"])
    


