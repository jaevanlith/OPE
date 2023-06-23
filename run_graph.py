import numpy as np
import pickle
from matplotlib import pyplot as plt
from datetime import date
from graph_KL_divergence import get_evaluation_policies, plot_kl_bounds

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

def run(experiment_args, p):

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

        # Set evaluation policy for the given kl_divergence
        cfg.eval_policy = p
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

    return results, {'seeds':seeds, 'args': experiment_args, 'eval_policy': cfg.eval_policy}


def run_kl_experiment(experiment_args):
    results_all_behaviors = []

    # Run for all different behavior policies
    for q in experiment_args["behavior_policies"]:
        print("Running behavior policy ", q)
        # Construct environment
        cfg = get_configuration(experiment_args, 0)
        env = Graph(make_pomdp=cfg.is_pomdp,
                    number_of_pomdp_states=cfg.pomdp_horizon,
                    transitions_deterministic=not cfg.stochastic_env,
                    max_length=cfg.horizon,
                    sparse_rewards=cfg.sparse_rewards,
                    stochastic_rewards=cfg.stochastic_rewards)
        
        # Plot bounds of KL divergence
        if not os.path.exists(experiment_args["image_path"] + "KL_bounds_graph_plot.png"):
            plot_kl_bounds(env, 50, path=experiment_args["image_path"])

        # Get evaluation policies corresponding KL's
        eval_policies_with_kl = get_evaluation_policies(env, q, experiment_args['kl_divergence_steps'])
        print("Evaluation policies with kl's: ")
        print(eval_policies_with_kl)

        # Init results
        results = []

        # Run for all possible KL divergences
        for (p, kl) in eval_policies_with_kl:
            print("Running policy: ", p, " with kl: ", kl)

            # Set the behavior policy
            experiment_args["p0"] = q

            # Run experiments
            graph_results = run(experiment_args, p)
            print
            dict_results = {"behavior_policy":  q,
                            "eval_policy": p,
                            "kl_divergence": kl,
                            "results": graph_results
                            }
            print('behavior_policy: ' , dict_results["behavior_policy"] , ' calculated_eval_policy: ' , dict_results["eval_policy"] , ' kl_divergence: ' , dict_results["kl_divergence"])
            # Save results
            results.append(dict_results)
            
        
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

    # Set number of KL divergence steps
    experiment_args['kl_divergence_steps'] = cmd_args['kl_divergence_steps']
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
def save_results(results, cmd_args, path, weighted):

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

    if weighted:
        with open(path + "weighted_results" +cmd_args['behavior_policies'].replace(".", "").replace(",", "_") +".json", "w") as weighted_data:
                json.dump(results, weighted_data, indent = 4, default=default)
        weighted_data.close()
    else:
        with open(path + "unweighted_results"+ cmd_args['behavior_policies'].replace(".", "").replace(",", "_")  +".json", "w") as unweighted_data:
                json.dump(results, unweighted_data, indent=4, default=default)
        unweighted_data.close()

def plot_results(results, weighted, cmd_args, fixed_n_value=None):
    path = cmd_args['image_path']

    if weighted:
        if not os.path.exists(path + "weighted/"):
            os.makedirs(path + "weighted/")

        # When results are for all n, make 3D plot
        if fixed_n_value is None:
            for i in results:
                fixed_n_value = 10
                neurips_plot_kl_3D(raw_results=i, 
                               filename=path + "weighted/bp=" + str(i[0]["behavior_policy"]).replace(".", "") + "_graph_plot_kl_3D", 
                               weighted=True,  
                               cycler_small=True)
                neurips_plot_kl_2D(raw_results=i, 
                               filename=path + "weighted/bp=" + str(i[0]["behavior_policy"]).replace(".", "") + "_n="+ str(fixed_n_value) + "_graph_plot_kl_2D", 
                               weighted=True,  
                               cycler_small=True, 
                               fixed_n_value=fixed_n_value)
        # If n was fixed, make 2D plot
        else:
            for i in weighted_graph_results:
                neurips_plot_kl_2D(raw_results=i, 
                               filename=path + "weighted/bp=" + str(i[0]["behavior_policy"]).replace(".", "") + "_n="+ str(fixed_n_value) + "_graph_plot_kl_2D", 
                               weighted=True,  
                               cycler_small=True, 
                               fixed_n_value=fixed_n_value)
                
    else:
        if not os.path.exists(path + "unweighted/"):
            os.makedirs(path + "unweighted/")

        if fixed_n_value is None:
            fixed_n_value = 10
            for j in results:    
                neurips_plot_kl_3D(raw_results=j,
                                filename=path + "unweighted/bp=" + str(j[0]["behavior_policy"]).replace(".", "") + "_graph_plot_kl_3D", 
                                weighted=False, 
                                cycler_small=True)
                neurips_plot_kl_2D(raw_results=j,
                                filename=path + "unweighted/bp=" + str(j[0]["behavior_policy"]).replace(".", "") + "_n="+ str(fixed_n_value) + "_graph_plot_kl_2D", 
                                weighted=False, 
                                cycler_small=True, 
                                fixed_n_value=fixed_n_value)

        else:
            for j in unweighted_graph_results:
                neurips_plot_kl_2D(raw_results=j,
                               filename=path + "unweighted/bp=" + str(j[0]["behavior_policy"]).replace(".", "") + "_n="+ str(fixed_n_value) + "_graph_plot_kl_2D", 
                               weighted=False, 
                               cycler_small=True, 
                               fixed_n_value=fixed_n_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph experiment')
    # Models to run, call "SOPE" to only run one
    parser.add_argument('--models', type=str, default="SOPE")
    # Horizon of the experiment
    parser.add_argument('--horizon', type=int, default=20)
    # Number of trials to run per experiment  
    parser.add_argument('--num_trials', type=int, default=100)
    # Values for N
    parser.add_argument('--Nvals', type=str, default="256,512,1024,2048,4096,8192")
    # To run for a fixed n, set this to the value of n 
    parser.add_argument('--fix_n_value', type=int, default=None)
    # Number of steps to run for KL divergence
    parser.add_argument('--kl_divergence_steps', type=int, default=10)
    # Behavior policies to run
    parser.add_argument('--behavior_policies', type=str, default="0.1,0.5,0.9")
    # Path to which new images are saved
    parser.add_argument('--image_path', type=str, default="./experiment_images/")
    # Path to which new data is saved
    parser.add_argument('--save_path', type=str, default="./experiment_results/")
    # Runs experiments for weighted or unweighted estimator
    parser.add_argument('--weighted', type=bool, default=True)
    parser.add_argument('--unweighted', type=bool, default=False)
    # If data is already generated, load it instead of generating it
    parser.add_argument('--load_data_path_weighted', type=str, default=None)
    parser.add_argument('--load_data_path_unweighted', type=str, default=None)
    cmd_args = dict(vars(parser.parse_args()))

    if cmd_args["weighted"]:

        if cmd_args["load_data_path_weighted"] is None:
            print("Running weighted graph experiment")
            # Customize experiment arguments
            weighted_graph_args = customize_args(weighted_graph_args, cmd_args)
            # Run trials
            weighted_graph_results = run_kl_experiment(weighted_graph_args)

            #Save results
            save_results(weighted_graph_results, cmd_args, cmd_args["save_path"], True)

        else:
            print("Loading weighted data from: ", cmd_args["load_data_path_weighted"])
            # Load data when already there
            weighted_graph_results = json.load(open(cmd_args["load_data_path_weighted"]))

        # Plot results
        plot_results(weighted_graph_results, True, cmd_args, fixed_n_value=cmd_args['fix_n_value'])


    if cmd_args["unweighted"]:
        if cmd_args["load_data_path_unweighted"] is None:
            print("Running unweighted graph experiment")
            # Customize experiment arguments
            unweighted_graph_args = customize_args(unweighted_graph_args, cmd_args)
            # Run trials
            unweighted_graph_results = run_kl_experiment(unweighted_graph_args)

            #Save results
            save_results(unweighted_graph_results, cmd_args, cmd_args["save_path"], False)
        else:
            print("Loading unweighted data from: ", cmd_args["load_data_path_unweighted"])
            # Load data when already there
            unweighted_graph_results = json.load(open(cmd_args["load_data_path_unweighted"]))
            
    
        #Plot results
        plot_results(unweighted_graph_results, False, cmd_args, fixed_n_value=cmd_args['fix_n_value'])

    

    


