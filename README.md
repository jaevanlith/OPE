# State Visitation Correction Robustness in Off-policy Reinforcement Learning

This respository contains code for the following paper:
> Mayank Prashar, Jochem van Lith and Joost Jansen. \
State Visitation Correction in Off-Policy Reinforcement Learning.

The code built upon [SOPE: Spectrum of Off-Policy Estimators](https://github.com/Pearl-UTexas/SOPE).

#### Installation

Please use Python 3.6.13
```
pip install -r requirements.txt
pip install -e .
```

#### Results
To reproduce the results from the paper, see:
- `run_graph.py`
- `run_gridworld.py`

The following promps were used to reproduce the results from the paper:
```
!python run_graph.py --models "SOPE" --Nvals "1024" --behavior_policies "0.1, 0.5, 0.9" --save_path ./experiment_results/ --image_path ./experiment_images/ --weighted True --unweighted True
```

```
!python run_gridworld.py --models "SOPE" --Nvals "1024" --behavior_policies "0.1, 0.5, 0.9" --save_path ./experiment_results/ --image_path ./experiment_images/ --weighted True --unweighted True
```

The results are generated within Google Colab with the following notebook:
- [IDM_project.ipynb](https://colab.research.google.com/drive/1b8ij7ONO6MWGPHnA6tUvjttVW0PLQ5M7?usp=sharing)

#### Plots
To generate the plots from the paper, see:
- `additional_plots.py`


