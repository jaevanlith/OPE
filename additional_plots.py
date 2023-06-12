from neurips_plotting import neurips_plot_behavior_2D, neurips_plot_ns_2D

save_path = './experiment_results/'
image_path = './experiment_images/'

# Plot different behavior policies in one plot for n=10
behavior_policies = [0.1,0.5,0.9]
fixed_n = 10
# Weighted
neurips_plot_behavior_2D(behavior_policies, save_path, image_path, weighted=True, fixed_n_value=fixed_n)
# Unweighted
neurips_plot_behavior_2D(behavior_policies, save_path, image_path, weighted=False, fixed_n_value=fixed_n)

# Plot different values for n in one plot for bp=0.5
ns = [0,5,10,15,20]
bp = 0.5
# Weighted
neurips_plot_ns_2D(ns, bp, save_path, image_path, weighted=True)
# Unweighted
neurips_plot_ns_2D(ns, bp, save_path, image_path, weighted=False)