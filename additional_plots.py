from neurips_plotting import neurips_plot_behavior_2D, neurips_plot_ns_2D

def make_additional_plots(save_path, image_path):
    # Behavior policies for which we have results
    behavior_policies = [0.1,0.5,0.9]
    # Fixed values for n with step size of 5
    ns = [0,5,10,15,20]

    # Plot different behavior policies in one plot for all specified ns
    for fixed_n in ns:
        # Weighted
        neurips_plot_behavior_2D(behavior_policies, save_path, image_path, weighted=True, fixed_n_value=fixed_n)
        # Unweighted
        neurips_plot_behavior_2D(behavior_policies, save_path, image_path, weighted=False, fixed_n_value=fixed_n)

    # Plot different values for n in one plot for all behavior policies
    for bp in behavior_policies:
        # Weighted
        neurips_plot_ns_2D(ns, bp, save_path, image_path, weighted=True)
        # Unweighted
        neurips_plot_ns_2D(ns, bp, save_path, image_path, weighted=False)

# Plot for Graph
save_path_graph = './experiment_results/experiment_results_graph/'
image_path_graph = './experiment_images/experiment_images_graph/'
make_additional_plots(save_path_graph, image_path_graph)

# # Plot for GridWorld
save_path_grid = './experiment_results/experiment_results_grid/'
image_path_grid = './experiment_images/experiment_images_grid/'
make_additional_plots(save_path_grid, image_path_grid)