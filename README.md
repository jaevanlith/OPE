# SOPE: Spectrum of Off-Policy Estimators

This respository contains code for the following [paper](https://arxiv.org/abs/2111.03936):
> C. Yuan, Y. Chandak, S. Giguere, P.S. Thomas, and S. Niekum. 
SOPE: Spectrum of Off-Policy Estimators.
Neural Information Processing Systems (NeurIPS), December 2021. 

The code built upon [Caltech OPE Benchmarking Suite (COBS) Library](https://github.com/clvoloshin/COBS).

#### Installation

Please use Python 3.6+.
```
pip install -r requirements.txt
pip install -e .
```

#### Plots
To reproduce the plots from the paper, see:
- `run_graph.py`
- `run_toymc.py`

#### Bibliography
```
@article{yuan2021sope,
  title={SOPE: Spectrum of Off-Policy Estimators},
  author={Yuan, Christina J and Chandak, Yash and Giguere, Stephen and Thomas, Philip S and Niekum, Scott},
  journal={arXiv preprint arXiv:2111.03936},
  year={2021}
}
```
