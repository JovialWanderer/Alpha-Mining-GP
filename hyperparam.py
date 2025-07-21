import yaml
import numpy as np
import pandas as pd
config_file_path = "config.yaml"

try:
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: Configuration file '{config_file_path}' not found.")
    exit()
except yaml.YAMLError as exc:
    print(f"Error parsing YAML file: {exc}")
    exit()

# Accessing parameters
SEED = config['basicfeed']['SEED']
filepath=config['basicfeed']['filepath']
# Example: Accessing evolutionary algorithm parameters
num_individuals = config['evolutionary_algorithm']['population']['num_individuals']
num_elite = config['evolutionary_algorithm']['population']['perc_elite']*num_individuals
rng = np.random.default_rng(seed=SEED)
df=pd.read_csv(filepath)
# Example: Accessing feature flags
is_simulated = config['ablation']['is_simulated_annealing']
is_consecutive_warmstart = config['ablation']['is_consecutive_warmstart']
is_fixed_rate = config['ablation']['fixed_rates']['is_fixed_rate']
fixed_cross_rate = config['ablation']['fixed_rates']['fixed_crossover_rate']
fixed_mutation_rate = config['ablation']['fixed_rates']['fixed_mutation_rate']

print(f"Loaded SEED: {SEED}")
