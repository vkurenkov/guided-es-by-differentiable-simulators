import random
import numpy as np
import os

from scipy.stats import loguniform, randint
from math import ceil
from coolname import generate_slug

NUM_SEEDS                  = 5
SAMPLE_EPISODES_BUDGET     = 500
HYPERPARAMS_BUDGET_PER_ALG = 30
REAL                       = False

learning_rates    = loguniform(1e-4, 1.0)
num_perturbations = randint(1, 12)
optimizers        = ["adam", "sgd", "fromage"]
stds              = loguniform(1e-3, 1.0)
smooth            = [True, False]
shooting_ns       = randint(0, 40)
seeds             = [8114, 7766, 4891, 9522, 6433]

def sample_hyperparams():
    return {
        "lr": np.round(learning_rates.rvs(), decimals=4),
        "num_per": num_perturbations.rvs(),
        "opt": random.choice(optimizers),
        "std": np.round(stds.rvs(), decimals=4),
        "smooth": random.choice(smooth),
        "shoot_n": shooting_ns.rvs()
    } 

def run_command(params, alg, env, seed, codename):
    n_iters = ceil(SAMPLE_EPISODES_BUDGET / (params["num_per"] * 2))
    cmd = f'python3 train.py --alg=\"{alg}\" --iterations={n_iters} --seed={seed} --env=\"{env}\" --std={params["std"]} --optimizer=\"{params["opt"]}\" --lr={params["lr"]} --num-perturbs={params["num_per"]} '
    if params["smooth"]:
        cmd += " --smooth"
    if REAL:
        cmd += " --real"
    cmd += f" --codename={codename}"
    cmd += f" --shooting-n={params['shoot_n']}"
    os.system(cmd)

# Pendulum, Guided-ES
for _ in range(HYPERPARAMS_BUDGET_PER_ALG):
    params  = sample_hyperparams()
    codename = generate_slug(3)
    for seed in seeds:
        run_command(params, alg="guided-es", env="pendulum", seed=seed, codename=codename)

# Sim, Pendulum, CMA-ES
for _ in range(HYPERPARAMS_BUDGET_PER_ALG):
    params  = sample_hyperparams()
    codename = generate_slug(3)
    for seed in seeds:
        run_command(params, alg="cma-es", env="pendulum", seed=seed, codename=codename)


# Pendulum, Vanilla-ES
for _ in range(HYPERPARAMS_BUDGET_PER_ALG):
    params  = sample_hyperparams()
    codename = generate_slug(3)
    for seed in seeds:
        run_command(params, alg="vanilla-es", env="pendulum", seed=seed, codename=codename)