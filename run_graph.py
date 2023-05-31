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
from neurips_plotting import neurips_plot, neurips_plot_kl_2D
import json

def run(experiment_args, kl_target):

    runner = ExperimentRunner()

    seeds  = []
    all_N_vals = []
    for t in range(experiment_args["num_trials"]): # Number of trials, config file for each trial.

        # Set random seed for trial.
        if experiment_args["seeds"] is None: seed = make_seed()
        else: seed = experiment_args["seeds"][t]
        seeds.append(seed)

        # For each trial, generate a single large dataset with the largest Nval.
        # For each Nval, we will compute the estimate for that value on the same dataset.
        max_Nval = max(experiment_args["Nvals"])
        all_N_vals += experiment_args["Nvals"]

        
        # basic configuration with varying number of trajectories
        configuration = {
            "gamma": experiment_args["gamma"],
            "horizon": experiment_args["horizon"],
            "base_policy": experiment_args["p0"],
            "eval_policy": experiment_args["p1"],
            "stochastic_env": True,
            "stochastic_rewards": False,
            "sparse_rewards": False,
            "num_traj": max_Nval,
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
            "weighted": experiment_args["weighted"]
        }

        # store them
        cfg = Config(configuration)


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


if __name__ == '__main__':
    # Run trials.
    behavior_policies = [0.1]
    max_kl_divergence = get_kl_max(behavior_policies[0], 50)
    kl_divergence = 0.4 #np.linspace(0, max_kl_divergence, 10)
    weighted_results = []
    unweighted_results = []
    # Uses KL divergence to calculate the evaluation policy
    count = 0
    for i in behavior_policies:
        
        for k in kl_divergence:
            # Set the behaviour policy
            print("Running ", k, " of ", kl_divergence)

            unweighted_graph_args["p0"] = i
            weighted_graph_args["p0"] = i
            # Change Nvalues to only one measure
            unweighted_graph_args["Nvals"] = [1024]
            weighted_graph_args["Nvals"] = [1024]

            unweighted_graph_results = run(unweighted_graph_args,k)
            weighted_graph_results = run(weighted_graph_args, k)

            # Save results
            weighted_results.append({"behavior_policy":  i,
                            "kl_divergence": k,
                            "results": weighted_graph_results
                            })
            
            unweighted_results.append({"behavior_policy":  i,
                            "kl_divergence": k,
                            "results": unweighted_graph_results, 
                            })
            
            # # Plot 
            # neurips_plot(unweighted_graph_results, "unweighted_graph_plot_ep_" + str(i).replace(".", "") + "bp_"+ str(j).replace(".", ""), cycler_small=True)
            # neurips_plot(weighted_graph_results,5 "weighted_graph_plot"+ str(i).replace(".", "") + "bp_"+ str(j).replace(".", ""), cycler_small=True)

    # Save data in case of wrong plotting
    with open("weighted_results.json", "w") as weighted:
        json.dump(weighted_results, weighted)
    weighted.close()

    with open("unweighted_results.json", "w") as unweighted:
        json.dump(unweighted_results, unweighted)
    unweighted.close()


    # plot total results
    neurips_plot_kl_2D(weighted_results, "weighted_graph_plot_kl", cycler_small=True)
    neurips_plot_kl_2D(unweighted_results, "unweighted_graph_plot_kl", cycler_small=True)




