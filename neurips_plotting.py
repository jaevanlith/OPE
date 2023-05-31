import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt
import json

def extract_nstep(results_dict, args, weighted=True):
    output = []
    if 'nstep_int' in args.keys():
        for i in range(0, args['horizon'], args['nstep_int']):
            key = 'NStep(t=%d)' % i if weighted else 'NStepNoW(t=%d)' % i
            output.append(results_dict[key])
    else:
        for i in range(0, args['horizon'], 1):
            key = 'NStep(t=%d)' % i if weighted else 'NStepNoW(t=%d)' % i
            output.append(results_dict[key])
            
    return np.array(output)


def format_results(raw_results, weighted=True):
    Ns    = np.array([raw_results[1]['args']['Nvals'] for _ in range(raw_results[1]['args']['num_trials'])]).flatten()
    seeds = np.array(raw_results[1]['seeds'])
    
    results = [ extract_nstep(res, raw_results[1]['args'], weighted) for res in raw_results[0] ]
    true_val = raw_results[0][0]["ON POLICY"][0]

    nvals = np.unique(Ns) # sorted via default behavior of np.unique
    estimate_means  = []
    estimate_ses    = []
    error_means     = []
    error_ses       = []
    estimate_vars   = []
    my_mse          = []

    for nval in nvals:
        results_n = np.array([ res for res,n in zip(results,Ns) if nval==n ])
        results_n_means = results_n.mean(axis=0)
        results_n_ses  = results_n.std(axis=0, ddof=1)
        
        estimate_means.append(results_n_means[:,0])
        estimate_ses.append(results_n_ses[:,0]/np.sqrt(nval))
        error_means.append(results_n_means[:,1])
        error_ses.append(results_n_ses[:,1]/np.sqrt(nval))
        estimate_vars.append(results_n_ses[:,0])
        
        my_mse.append(np.mean((results_n[:,:,0] - true_val)**2, axis=0))

    return {
        'seeds'   : np.unique(seeds),
        'nvals'   : list(nvals),
        'true_val': true_val,
        'ests_mn' : np.array(estimate_means),
        'ests_se' : np.array(estimate_ses),
        'errs_mn' : np.array(error_means),
        'errs_se' : np.array(error_ses),
        "ests_var": np.array(estimate_vars),
        'my_mse'  : np.array(my_mse)}


def get_cyclers(small=False):
    lines = ["-","--","-.",":", (0,(3,1,1,1,1,1)), (0, (5,1))]
    linecycler = cycle(lines)

    if small:
        markers = ['.', '1', 3, "x", "3", 4, "|"]
        markercycler = cycle(markers)
    else:
        markers = ['D', 'p', 'X','>', 's', 'o', '*']
        markercycler = cycle(markers)

    colors = ["b", "orange", "g", "r", "purple", "c"]
    colorcycler = cycle(colors)
    
    return linecycler, markercycler, colorcycler

def neurips_plot(raw_results, filename, cycler_small=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    fig.tight_layout(rect=[0, 0.03, 0.8, 0.95])

    R = format_results(raw_results)
    linecycler, markercycler, colorcycler = get_cyclers(small=cycler_small)

    horizon = raw_results[1]['args']['horizon']
    nstep_int = raw_results[1]['args'].get('nstep_int', 1)

    for (errs, errs_se, nval, ests, ests_var) in zip(R['errs_mn'], R['errs_se'], R['nvals'], R['ests_mn'], R['ests_var']):

        color = next(colorcycler)
        linestyle = next(linecycler)
        marker = next(markercycler)
        
        # MSE
        axes[0].plot(np.arange(0, horizon, nstep_int), errs, color=color, linestyle=linestyle, label=nval, marker=marker)
        axes[0].fill_between(np.arange(0, horizon, nstep_int), errs-1.96*errs_se, errs+1.96*errs_se, alpha=0.25)
        axes[0].set_yscale('log')
        axes[0].set_xlabel("n-step value")
        axes[0].set_title("MSE")
        
        # Bias squared
        axes[1].plot(np.arange(0, horizon, nstep_int), abs(ests - R["true_val"])**2, color=color, linestyle=linestyle, marker=marker)
        axes[1].set_xlabel("n-step value")
        axes[1].set_title("Bias Squared Estimate")
        axes[1].set_yscale('log')
        
        # Variance.
        axes[2].plot(np.arange(0, horizon, nstep_int), ests_var, color=color, linestyle=linestyle, marker=marker)
        axes[2].set_xlabel("n-step value")
        axes[2].set_title("Variance Estimate")
        axes[2].set_yscale('log')
        
    fig.legend(title="# of Trajectories", loc="center right", bbox_to_anchor=(0.88, 0.6))
    fig.savefig(filename)


def format_results_kl_2d(raw_results, weighted=True):
    R = []
    kl_divergence = []
    for r in raw_results:
        kl_divergence.append(r["kl_divergence"])
        R.append(format_results(r["results"], weighted))

    estimate_means  = []
    estimate_ses    = []
    error_means     = []
    error_ses       = []
    estimate_vars   = []
    my_mse          = []
    seeds           = []

    for i in range(len(R[0]['nvals'])):
        esm = []
        ess = []
        erm = []
        ers = []
        esv = []
        mmse = []
        seed = []
        for [k,r] in zip(kl_divergence, R):
            esm.append(r['ests_mn'][i])
            ess.append(r['ests_se'][i]) 
            erm.append(r['errs_mn'][i])
            ers.append(r['errs_se'][i])
            esv.append(r['ests_var'][i])
            mmse.append(r['my_mse'][i])
            seed.append(r['seeds'][i])
        estimate_means.append(np.array(esm).flatten())
        estimate_ses.append(np.array(ess).flatten())
        error_means.append(np.array(erm).flatten())
        error_ses.append(np.array(ers).flatten())
        estimate_vars.append(np.array(esv).flatten())
        my_mse.append(np.array(mmse).flatten())
        seeds.append(np.array(seed).flatten())
    
    return {
        'seeds'   : np.unique(seeds),
        'kl_div': np.array(kl_divergence),
        'nvals'   : list(R[0]['nvals']),
        'true_val': R[0]['true_val'],
        'ests_mn' : np.array(estimate_means),
        'ests_se' : np.array(estimate_ses),
        'errs_mn' : np.array(error_means),
        'errs_se' : np.array(error_ses),
        "ests_var": np.array(estimate_vars),
        'my_mse'  : np.array(my_mse)}

def neurips_plot_kl_3D(raw_results, filename, weighted, cycler_small=False):

    clean_results = format_results_kl_2d(raw_results)
    linecycler, markercycler, colorcycler = get_cyclers(small=cycler_small)

    horizon = raw_results[0]['results'][1]['args']['horizon']
    nstep_int = raw_results[0]['results'][1]['args'].get('nstep_int', 1)

    for (errs, errs_se, kl_div, ests, ests_var, nval) in zip( clean_results['errs_mn'], clean_results['errs_se'], clean_results['kl_div'], clean_results['ests_mn'], clean_results['ests_var'], clean_results['nvals']):
        color = next(colorcycler)
        linestyle = next(linecycler)
        marker = next(markercycler)

        # Generate sample data
        n_step_int = np.arange(0, horizon, nstep_int)
        kl_div =  clean_results['kl_div']
        z_axis = [errs,abs(ests - clean_results["true_val"])**2, ests_var]
        z_names = ["MSE", "Bias Squared Estimate", "Variance Estimate"]
        # Create a grid of coordinates
        X, Y = np.meshgrid(n_step_int, kl_div)

        # Reshape the kl_div data to match the grid shape
        Z = [i.reshape(X.shape) for i in z_axis]
        print(Z[0].shape)
        # Create three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

        # Set different viewing angles for each subplot
        viewing_angles = [(30, 45), (60, 120), (90, 180)]

        for i, ax in enumerate(axes):
            ax.plot_surface(X,Y , Z[i], cmap='viridis')
            ax.view_init(elev=viewing_angles[0][0], azim=viewing_angles[0][1])
            ax.set_xlabel('n_step_int')
            ax.set_ylabel('KL divergence')
            ax.set_zscale('log')
            ax.set_zlabel(z_names[i])
            ax.set_title('Graph {}'.format(z_names[i]))

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()
        fig.savefig(filename + "_nval_" + str(nval).replace(".", ""))

    




