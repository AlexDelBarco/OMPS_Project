## Import packages
import gurobipy as gb
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

np.random.seed(1)


prices_data = [
    2.02, 0.04, 0.14, 0.73, 17.47, 36.22, 51.59, 74.26,
    74.26, 64.81, 51.03, 42.64, 40.00, 40.22, 42.08, 48.53,
    51.67, 79.67, 92.92, 113.99, 124.52, 104.46, 90.10, 84.10
]

#num_power_balance_scenarios = 3
num_price_scenarios = 20
num_scenarios_wind = 20

#SCENARIOS_BALANCE = [f'"B{k}"' for k in range(1, num_power_balance_scenarios + 1)]
SCENARIOS_PRICE = [f'"P{k}"' for k in range(1, num_price_scenarios + 1)]
SCENARIOS_WIND = [f'"V{k}"' for k in range(1, num_scenarios_wind + 1)]

#num_scenarios_total = num_price_scenarios * num_power_balance_scenarios * num_scenarios_wind
num_scenarios_total = num_price_scenarios * num_scenarios_wind

SCENARIOS_TOT = [k for k in range(1, num_scenarios_total + 1)]

SCENARIOS = np.sort(np.random.choice(np.arange(1, num_scenarios_total + 1), size=num_scenarios_total, replace=False))
#SCENARIOS_OUT = np.setdiff1d(SCENARIOS_TOT, SCENARIOS)

# Generate binary variables representing deficit (0) or excess (1)
# balance_scenarios = {}
# for k in SCENARIOS_BALANCE:
#     for t in range(1, 25):
#         power_balance = np.random.randint(0, 2)
#         balance_scenarios[t, k] = power_balance

## Generate price scenarios using normal distribution
price_scenarios = {}
# plt.figure(figsize=(12, 12))
k_x = 0
for k in SCENARIOS_PRICE:
    k_x = k_x + 1
    plot_values_k = []
    for t in range(1, 25):
        coeff_prices = abs(np.random.normal(1, 0.1))
        price_scenarios[t, k] = coeff_prices * prices_data[t - 1]
        plot_values_k.append(coeff_prices * prices_data[t - 1])
    plt.plot(range(0, 24), plot_values_k, linewidth=1.5, label=f'P{k_x}')


#plt.size = (6, 6)
#plt.xlim(0,24)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Electricity Price (EUR/MWh)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.title('Price scenarios', fontsize=12)
plt.legend(fontsize = 7, loc='upper left')
#plt.savefig('figures/price_scenarios.png', dpi=600, bbox_inches='tight')
plt.show()

### WT Power definition
WIND_FARMS = ['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6']

dfs = {}
wind_power_scenarios = {}
#TODO check 200MW??
p_max_wind = 200 / 6

for k in range(1, 7):
    file_path = f'Assignment_ideas/data/wind {k}.out'

    with open(file_path, 'r') as file:
        file_contents = file.readlines()
        data = [line.strip().split(",") for line in file_contents]
        df = pd.DataFrame(data)
        df = df.iloc[:, 1:]
        df.columns = df.iloc[0]
        df = df.iloc[1:25, :num_scenarios_wind]
        df = df.reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        dfs[f'WT{k}'] = df * p_max_wind

# plt.figure(figsize=(12, 12))
k_x = 0
for k in SCENARIOS_WIND:
    k_x = k_x + 1
    plot_values_k = []
    for t in range(1, 25):
        wind_power_scenarios[t, k] = sum(dfs[wt][k][t - 1] for wt in WIND_FARMS)
        plot_values_k.append(sum(dfs[wt][k][t - 1] for wt in WIND_FARMS))

    plt.plot(range(0,24), plot_values_k, linewidth=1.5, label=f'W{k_x}')

plt.xlim(0,24)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Real windfarm production (MWh)', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.title('Wind farm production scenarios', fontsize=12)
plt.legend(fontsize=7, loc='upper left')
#plt.savefig('figures/wind_scenarios.png', dpi=600, bbox_inches='tight')
plt.show()

##
scenarios_data = {}
for t in range(1, 25):
    for k in range(1, num_scenarios_total + 1):
        k_p = SCENARIOS_PRICE[((k - 1) // num_scenarios_wind) % num_price_scenarios]
        k_w = SCENARIOS_WIND[(k - 1) % num_scenarios_wind]
        scenarios_data[(t, k)] = [price_scenarios[(t, k_p)], wind_power_scenarios[(t, k_w)]]

# for t in range(1, 25):
#     for k in range(1, num_scenarios_total + 1):
#         k_b = SCENARIOS_BALANCE[(k - 1) % num_power_balance_scenarios]
#         k_p = SCENARIOS_PRICE[((k - 1) // num_power_balance_scenarios) % num_price_scenarios]
#         k_w = SCENARIOS_WIND[((k - 1) // (num_power_balance_scenarios * num_price_scenarios)) % num_scenarios_wind]
#         scenarios_data[(t, k)] = [balance_scenarios[(t, k_b)], price_scenarios[(t, k_p)],
#                                   wind_power_scenarios[(t, k_w)]]
# test 2

