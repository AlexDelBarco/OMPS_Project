import pandas as pd
import numpy as np

def generate_scenarios(num_price_scenarios, num_wind_scenarios):
    np.random.seed(1)
    #euro/MWh
    prices_data_DA = [
        51.49, 48.39, 48.92, 49.45, 42.72, 50.84, 82.15, 100.96, 116.60,
        112.20, 108.54, 111.61, 114.02, 127.40, 134.02, 142.18, 147.42,
        155.91, 154.10, 148.30, 138.59, 129.44, 122.89, 112.47
    ]

    prices_data_B = [
        51.49, 108.89, 90.76, 90.76, 42.72, -2.00, 11.89, 40.50, 67.98, 42.00, 42.00,
        46.92, 114.02, 127.40, 134.02, 142.18, 198.65, 402.30, 154.10, 148.30,
        138.59, 129.44, 122.89, 112.47
    ]

    SCENARIOS_PRICE = [f'P{k}' for k in range(1, num_price_scenarios + 1)]
    SCENARIOS_WIND = [f'"V{k}"' for k in range(1, num_wind_scenarios + 1)]

    num_scenarios_total = num_price_scenarios * num_wind_scenarios

    SCENARIOS = np.sort(np.random.choice(np.arange(1, num_scenarios_total + 1), size=num_scenarios_total, replace=False))

    price_scenarios_DA = {}
    price_scenarios_B = {}
    k_x = 0
    for k in SCENARIOS_PRICE:
        k_x += 1
        for t in range(1, 25):
            coeff_prices = abs(np.random.normal(1, 0.1))
            price_scenarios_DA[t, k] = coeff_prices * prices_data_DA[t - 1]
            price_scenarios_B[t, k] = coeff_prices * prices_data_B[t - 1]

    WIND_FARMS = ['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6']

    dfs = {}
    wind_power_scenarios = {}
    p_max_wind = 48 / 6

    for k in range(1, 7):
        file_path = f'data/wind {k}.out'

        with open(file_path, 'r') as file:
            file_contents = file.readlines()
            data = [line.strip().split(",") for line in file_contents]
            df = pd.DataFrame(data)
            df = df.iloc[:, 1:]
            df.columns = df.iloc[0]
            df = df.iloc[1:25, :num_wind_scenarios]
            df = df.reset_index(drop=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            dfs[f'WT{k}'] = df * p_max_wind

    k_x = 0
    for k in SCENARIOS_WIND:
        k_x += 1
        for t in range(1, 25):
            wind_power_scenarios[t, k] = sum(dfs[wt][k][t - 1] for wt in WIND_FARMS)

    scenarios_data_DAprices = {}
    scenarios_data_Bprices = {}
    scenarios_data_WindProd = {}
    for t in range(1, 25):
        for k in range(1, num_scenarios_total + 1):
            k_p = SCENARIOS_PRICE[((k - 1) // num_wind_scenarios) % num_price_scenarios]
            k_w = SCENARIOS_WIND[(k - 1) % num_wind_scenarios]
            scenarios_data_DAprices[(t, k)] = price_scenarios_DA[(t, k_p)]
            scenarios_data_Bprices[(t, k)] = price_scenarios_B[(t, k_p)]
            scenarios_data_WindProd[(t, k)] = wind_power_scenarios[(t, k_w)]

    return scenarios_data_DAprices, scenarios_data_Bprices, scenarios_data_WindProd


scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(10, 10)
