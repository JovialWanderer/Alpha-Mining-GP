import yaml
import random
import numpy as np

# Path to your YAML configuration file
config_file_path = 'config.yaml' # Or 'hyperparams.yaml'

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
# Example: Accessing seeds
SEED = config['seeds']['SEED']
random.seed(SEED)
np.random.seed(SEED)
global_random_state = random.Random(SEED)
global_np_random_state = np.random.RandomState(SEED)

# Example: Accessing volatility parameters
yz_window = config['volatility']['yz_window_length']
vol_params = config['volatility']['params']

# Example: Accessing evolutionary algorithm parameters
num_individuals = config['evolutionary_algorithm']['population']['num_individuals']
num_elite = config['evolutionary_algorithm']['population']['num_elite'] # Or calculate dynamically: int(0.1 * num_individuals)

# Example: Accessing backtest parameters
init_cash = config['backtest']['initial_cash']

# Example: Accessing feature flags
is_simulated = config['ablation']['is_simulated_annealing']
is_consecutive_warmstart = config['ablation']['is_consecutive_warmstart']
is_fixed_rate = config['ablation']['fixed_rates']['is_fixed_rate']
fixed_cross_rate = config['ablation']['fixed_rates']['fixed_crossover_rate']

# You would then continue to assign all your variables from the 'config' dictionary
# based on the structure defined in the YAML file.

print(f"Loaded SEED: {SEED}")
print(f"Loaded vol_params: {vol_params}")
print(f"Loaded num_individuals: {num_individuals}")
print(f"Loaded is_fixed_rate: {is_fixed_rate}")
print(f"Loaded fixed_crossover_rate: {fixed_cross_rate}")

# Re-initialize dynamic structures if needed, based on loaded config
avg_sharpe_dict = {}
avg_test_res = config['execution']['result_tracking']['avg_test_result_initial'] # Get initial value
dict_low = {}
dict_high = {}
exp_del_p = {}
price_del_p = {}

for d in range(2, config['integration']['num_depth'] + 1):
    avg_sharpe_dict[d] = []
    # avg_test_res is a single value, not per depth in your original code
    dict_low[d] = {}
    dict_high[d] = {}
    exp_del_p[d] = {}
    price_del_p[d] = {}

# Apply conditional logic from your original code
curr_warmstart_percent = config['execution']['warmstart']['warmstart_percent']
if not is_consecutive_warmstart:
    curr_warmstart_percent = 0.0

fixed_cross_rate_actual = config['ablation']['fixed_rates']['fixed_crossover_rate']
fixed_mut_rate_actual = config['ablation']['fixed_rates']['fixed_mutation_rate']
if not is_fixed_rate:
    # These would typically be updated by your adaptive rate logic
    fixed_cross_rate_actual = config['integration']['crossover_rates']['initial_current_cross'] # Example: use initial adaptive rate
    fixed_mut_rate_actual = config['integration']['mutation_rates']['initial_current_mut'] # Example: use initial adaptive rate


print(f"Current warmstart percent after logic: {curr_warmstart_percent}")
print(f"Actual fixed crossover rate after logic: {fixed_cross_rate_actual}")