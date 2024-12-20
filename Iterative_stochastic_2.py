import gurobipy as gp
from gurobipy import GRB
from scenario_generation_function import generate_scenarios
import matplotlib.pyplot as plt

results = {
    "soc": {},
    "balancing_power": {},
    "balancing_charge": {},
    "balancing_discharge": {},
    "objective_value": {},
    "B_price": {},
    "Wind_production": {},
    "DA_prices": {},
    "DA_bid": {}
}

#SOC_st = []
#SOC_st.append(0)

n_sc = 20*20
SOC_st = [{}]  # List of dictionaries to store SOC for each scenario at each time step
initial_soc = 10
SOC_st[0] = {k: initial_soc for k in range(1, n_sc + 1)}

for h in range(1,25):
    class InputData:

        def __init__(
                self,
                SCENARIOS: list,
                TIME: list,
                generator_cost: float,
                generator_capacity: float,
                generator_availability: dict[tuple[int, int], float],
                b_price: dict[tuple[int, int], float],
                da_price: list,
                pi: dict[int, float],
                rho_charge: float,
                rho_discharge: float,
                soc_max: float,
                soc_init: float,
                charging_capacity: float,
                generator_DAbid: list,
                SOC: dict[tuple[int, int], float]
        ):
            # List of scenarios
            self.SCENARIOS = SCENARIOS
            # List of time set
            self.TIME = TIME
            # Generators costs (c^G_i)
            self.generator_cost = generator_cost
            # Wind availability in each scenario
            self.generator_availability = generator_availability
            # Generators capacity (P^G_i)
            self.generator_capacity = generator_capacity
            # Market clearing price DA(lambda_DA)
            self.da_price = da_price
            # Market clearing price Balancing Market (lambda_B)
            self.b_price = b_price
            # Scenario probability
            self.pi = pi
            #rho charge
            self.rho_charge = rho_charge
            #rho discharge
            self.rho_discharge = rho_discharge
            # Battery capacity (SOC)
            self.soc_max = soc_max
            # Initial battery soc
            self.soc_init = soc_init
            # power output of battery
            self.charging_capacity = charging_capacity
            # DA power bid
            self.generator_DAbid = generator_DAbid
            # SOC of battery
            self.SOC = SOC


    if __name__ == '__main__':
        num_wind_scenarios = 20
        num_price_scenarios = 20
        SCENARIOS = num_wind_scenarios * num_price_scenarios
        Scenarios = [i for i in range(1, SCENARIOS + 1)]
        Time = [h]

        scescenario_DA_prices = {}
        scenario_B_prices = {}
        scenario_windProd = {}
        scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(num_price_scenarios, num_wind_scenarios)


        scenario_B_prices = {key: value for key, value in scenario_B_prices.items() if key[0] == h}
        scenario_windProd = {key: value for key, value in scenario_windProd.items() if key[0] == h}


        scenario_DA_prices = [
            51.49, 48.39, 48.92, 49.45, 42.72, 50.84, 82.15, 100.96, 116.60,
            112.20, 108.54, 111.61, 114.02, 127.40, 134.02, 142.18, 147.42,
            155.91, 154.10, 148.30, 138.59, 129.44, 122.89, 112.47
        ]


        da_production =  [19.38809586917009, 21.503560001390973, 22.470803694851874, 25.455881096031092, 28.095600215068348,
         24.831544139658043, 25.044261446962928, 24.680626664933506, 26.62202031828845, 28.128963658061036,
         27.559908610050755, 26.540759164792767, 25.18216095126608, 25.658854186304207, 27.9539416745984,
         24.947139937413866, 20.913987154615185, 23.277756558519155, 20.773859557669823, 22.443171461887,
         25.903047336543384, 28.37591189666934, 26.278558704228168, 25.909366451370115]

        #SOC = SOC_st[-1]
        SOC = SOC_st[h-1]

        # Equal probability of each scenario 1/100
        pi = 1 / SCENARIOS

        input_data = InputData(
            SCENARIOS=Scenarios,
            TIME=Time,
            generator_cost=20,
            generator_capacity=48,
            generator_availability=scenario_windProd, # WF production
            generator_DAbid=da_production[h-1], # # Power to DA
            da_price=scenario_DA_prices[h-1], # DA price
            b_price=scenario_B_prices, # Balancing price
            pi=pi,
            rho_charge=0.8332,  # TODO what were the rho values before??
            rho_discharge=0.8332,
            SOC=SOC,
            soc_max=120,
            soc_init=10, # I dont think I need it
            charging_capacity=100
        )
    print('HEEEEEEEREEEEEEEEEE')
    print(f'Wind Prod for {h} {scenario_windProd}')
    #print(f'DA prod for {h} {da_production[h-1]}')
    #print(f'DA prices for {h} {scenario_DA_prices[h-1]}')
    #print(f'B prices for {h} {scenario_B_prices}')
    #print(f'SOC for {h} {SOC}')
    #print('End Input Data')

    class Expando(object):
        '''
            A small class which can have attributes set
        '''
        pass

    class StochasticOfferingStrategy():

        def __init__(self, input_data: InputData):
            self.data = input_data
            self.variables = Expando()
            self.constraints = Expando()
            self.results = Expando()
            self._build_model()

        def _build_variables(self):
            self.variables.soc = {
                (t, k): self.model.addVar(
                    lb=0, ub=1000, name=f'State of charge_{t}_{k}'
                )
                for t in self.data.TIME
                for k in self.data.SCENARIOS

            }

            self.variables.balancing_discharge = {
                (t, k): self.model.addVar(
                    lb=0, ub=self.data.charging_capacity, name=f'Balancing_discharge_{t}_{k}'
                )
                for t in self.data.TIME
                for k in self.data.SCENARIOS

            }

            self.variables.balancing_charge = {
                (t, k): self.model.addVar(
                    lb=0, ub=self.data.charging_capacity, name=f'Balancing_charge_{t}_{k}'
                )
                for t in self.data.TIME
                for k in self.data.SCENARIOS

            }

            self.variables.balancing_power = {
                (t, k): self.model.addVar(
                    lb=0, ub=self.data.generator_capacity, name=f'Balancing power_{t}_{k}'
                )
                for t in self.data.TIME
                for k in self.data.SCENARIOS
            }


        def _build_constraints(self):

            self.constraints.balancing_constraint = {
                (t, k): self.model.addLConstr(
                    self.data.generator_availability[(t,k)] + self.variables.balancing_discharge[(t,k)],
                    GRB.EQUAL,
                    self.data.generator_DAbid + self.variables.balancing_power[(t,k)] + self.variables.balancing_charge[(t,k)],
                    name=f'Balancing power constraint_{t}_{k}'
                )
                for k in self.data.SCENARIOS
                for t in self.data.TIME
            }


            self.constraints.SOC_max = {
                (t, k): self.model.addLConstr(
                    self.variables.soc[(t,k)],
                    GRB.LESS_EQUAL,
                    self.data.soc_max,
                    name=f'Max state of charge constraint_{t}_{k}'
                )
                for k in self.data.SCENARIOS
                for t in self.data.TIME
            }

            self.constraints.SOC_time = {
                (t, k): self.model.addLConstr(
                    self.data.SOC[k] + (self.variables.balancing_charge[(t,k)])* self.data.rho_charge
                    - (self.variables.balancing_discharge[(t,k)]) * (1/self.data.rho_discharge),
                    GRB.EQUAL,
                    self.variables.soc[(t,k)],
                    name=f'SOC time constraint_{t}_{k}'
                )
                for k in self.data.SCENARIOS
                for t in self.data.TIME
            }

        def _build_objective_function(self):
            objective = (
                gp.quicksum(
                    gp.quicksum(
                        (self.data.da_price - self.data.generator_cost) * self.data.generator_DAbid
                        + self.data.pi * (self.data.b_price[(t, k)] - self.data.generator_cost) * self.variables.balancing_power[(t, k)]
                        for k in self.data.SCENARIOS
                    )
                    for t in self.data.TIME
                )
            )
            self.model.setObjective(objective, GRB.MAXIMIZE)

        def _build_model(self):
            self.model = gp.Model(name='Iterative stochastic offering strategy')
            self._build_variables()
            self._build_constraints()
            self._build_objective_function()
            self.model.update()

        def _save_results(self):
            self.results.objective_value = self.model.ObjVal #total revenue
            self.results.balancing_power = { #
                (t, k): self.variables.balancing_power[(t, k)].x for t in self.data.TIME for k in self.data.SCENARIOS
            }
            self.results.balancing_charge = {
                (t, k): self.variables.balancing_charge[(t, k)].x for t in self.data.TIME for k in self.data.SCENARIOS
            }
            self.results.balancing_discharge = {
                (t, k): self.variables.balancing_discharge[(t, k)].x for t in self.data.TIME for k in self.data.SCENARIOS
            }
            self.results.soc = {
                (t, k): self.variables.soc[(t, k)].x for t in self.data.TIME for k in self.data.SCENARIOS
            }

            self.results.soc_aggregated = {t: sum(self.variables.soc[(t, k)].x for k in self.data.SCENARIOS)
                                           for t in self.data.TIME}

        def run(self):
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                self._save_results()
            else:
                self.model.computeIIS()
                self.model.write("model.ilp")
                print(f"optimization of {self.model.ModelName} was not successful")
                for v in self.model.getVars():
                    print(f"Variable {v.varName}: LB={v.lb}, UB={v.ub}")
                for c in self.model.getConstrs():
                    print(f"Constraint {c.ConstrName}: Slack={c.slack}")
                raise RuntimeError(f"optimization of {self.model.ModelName} was not successful")


    model = StochasticOfferingStrategy(input_data)
    model.run()

    results["soc"][h] = model.results.soc
    results["balancing_power"][h] = model.results.balancing_power
    results["balancing_charge"][h] = model.results.balancing_charge
    results["balancing_discharge"][h] = model.results.balancing_discharge
    results["objective_value"][h] = model.results.objective_value
    results["B_price"][h] = scenario_B_prices
    results["Wind_production"][h] = scenario_windProd
    results["DA_prices"][h] = scenario_DA_prices[h - 1]
    results["DA_bid"][h] = da_production[h -1]

    #soc_value = sum(model.results.soc[(h, k)] for k in model.data.SCENARIOS)
    #SOC_st.append(soc_value/SCENARIOS)
    scenario_SOC = {k: model.results.soc[(h, k)] for k in model.data.SCENARIOS}
    SOC_st.append(scenario_SOC)  # Store the dictionary for this hour

    #model.display_results()

# Revenue
print('here')

rev = {
    "B_revenue": {},
    "DA_revenue": {},
    "Total_revenue": {}
}

for h in range(1, 25):
    scenarios = list(results["soc"][h].keys())  # Get the list of scenarios for the hour
    num_scenarios = len(scenarios)
    rev["B_revenue"][h] = {}
    rev["DA_revenue"][h] = {}
    rev["Total_revenue"][h] = {}
    for s in range(1, num_scenarios+1):
        if results["balancing_power"][h][h, s] != 0:
            rev["B_revenue"][h][h, s] = (results["B_price"][h][h, s]- 20) * (results["balancing_power"][h][h, s])
        if results["balancing_power"][h][h, s] == 0:
            rev["B_revenue"][h][h, s] = 0
        rev["DA_revenue"][h] = (results["DA_prices"][h]-20) * (results["DA_bid"][h])
        rev["Total_revenue"][h][h, s] = rev["B_revenue"][h][h, s] + rev["DA_revenue"][h]

#rev['bal_revenue'][]

# Mean results
mean_results = {
    "soc": {},
    "balancing_power": {},
    "balancing_charge": {},
    "balancing_discharge": {},
    "objective_value": {},
    "scenario_DA_prices": {},
    "da_production": {},
    "scenario_B_price_mean": {},
    "B_price": {},
    "Wind_production": {},
    "revenue_B": {},
    "revenue_DA": {},
    "revenue_T": {}
}

for h in range(1, 25):
    scenarios = list(results["soc"][h].keys())  # Get the list of scenarios for the hour
    num_scenarios = len(scenarios)

    # Compute the mean for each metric
    mean_results["soc"][h] = sum(results["soc"][h][k] for k in scenarios) / num_scenarios
    mean_results["balancing_power"][h] = sum(results["balancing_power"][h][k] for k in scenarios) / num_scenarios
    mean_results["balancing_charge"][h] = sum(results["balancing_charge"][h][k] for k in scenarios) / num_scenarios
    mean_results["balancing_discharge"][h] = sum(results["balancing_discharge"][h][k] for k in scenarios) / num_scenarios
    mean_results["objective_value"][h] = results["objective_value"][h]  # Already aggregated per hour
    mean_results["B_price"][h] = sum(results["B_price"][h][k] for k in scenarios) / num_scenarios
    mean_results["Wind_production"][h] = sum(results["Wind_production"][h][k] for k in scenarios) / num_scenarios

    # Add scenario_DA_prices and da_production (no mean calculation needed)
    mean_results["scenario_DA_prices"][h] = scenario_DA_prices[h - 1]
    mean_results["da_production"][h] = da_production[h - 1]

    # Calculate the mean of scenario_B_prices across all scenarios
    mean_results["scenario_B_price_mean"][h] = sum(results["B_price"][h][k] for k in scenarios) / num_scenarios

    # Calculate the mean revenue
    mean_results["revenue_B"][h] = sum(rev["B_revenue"][h][k] for k in scenarios) / num_scenarios
    mean_results["revenue_DA"][h] = rev["DA_revenue"][h]
    mean_results["revenue_T"][h] = sum(rev["Total_revenue"][h][k] for k in scenarios) / num_scenarios

#Results for Min and Max producction 1
scenario_sums = {i: 0.0 for i in range(1, len(results['Wind_production'][1])+1)}  # There are 25 scenarios

# Sum up values for each scenario across all hours
for hour, values in results['Wind_production'].items():
    for (h, scenario), value in values.items():
        scenario_sums[scenario] += value

# Find the scenario with the minimum and maximum total wind production
min_scenario = min(scenario_sums, key=scenario_sums.get)
max_scenario = max(scenario_sums, key=scenario_sums.get)

max_results = {
    "soc": {},
    "balancing_power": {},
    "balancing_charge": {},
    "balancing_discharge": {},
    "objective_value": {},
    "scenario_DA_prices": {},
    "da_production": {},
    "scenario_B_price_mean": {},
    "B_price": {},
    "Wind_production": {},
    "revenue_DA": {},
    "revenue_B": {},
    "revenue_t": {}
}

min_results = {
    "soc": {},
    "balancing_power": {},
    "balancing_charge": {},
    "balancing_discharge": {},
    "objective_value": {},
    "scenario_DA_prices": {},
    "da_production": {},
    "scenario_B_price_mean": {},
    "B_price": {},
    "Wind_production": {},
    "revenue_DA": {},
    "revenue_B": {},
    "revenue_t": {}
}

for h in range(1, 25):
    num_scenarios = len(results['Wind_production'][1])

    # Take value for min and max scenario
    min_results["soc"][h] = results["soc"][h][h,min_scenario]
    max_results["soc"][h] = results["soc"][h][h,max_scenario]
    min_results["balancing_power"][h] = results["balancing_power"][h][h,min_scenario]
    max_results["balancing_power"][h] = results["balancing_power"][h][h,max_scenario]
    min_results["balancing_charge"][h] = results["balancing_charge"][h][h,min_scenario]
    max_results["balancing_charge"][h] = results["balancing_charge"][h][h,max_scenario]
    min_results["balancing_discharge"][h] = results["balancing_discharge"][h][h,min_scenario]
    max_results["balancing_discharge"][h] = results["balancing_discharge"][h][h,max_scenario]
    min_results["objective_value"][h] = results["objective_value"][h]
    max_results["objective_value"][h] = results["objective_value"][h] # Already aggregated per hour
    min_results["B_price"][h] = results["B_price"][h][h,min_scenario]
    max_results["B_price"][h] = results["B_price"][h][h,max_scenario]
    min_results["Wind_production"][h] = results["Wind_production"][h][h,min_scenario]
    max_results["Wind_production"][h] = results["Wind_production"][h][h,max_scenario]
    min_results["revenue_t"][h] = rev["Total_revenue"][h][h, min_scenario]
    max_results["revenue_t"][h] = rev["Total_revenue"][h][h, max_scenario]
    min_results["revenue_DA"][h] = rev["DA_revenue"][h]
    max_results["revenue_DA"][h] = rev["DA_revenue"][h]
    min_results["revenue_B"][h] = rev["B_revenue"][h][h, min_scenario]
    max_results["revenue_B"][h] = rev["B_revenue"][h][h, max_scenario]


    # Add scenario_DA_prices and da_production (no mean calculation needed)
    min_results["scenario_DA_prices"][h] = scenario_DA_prices[h - 1]
    max_results["scenario_DA_prices"][h] = scenario_DA_prices[h - 1]
    min_results["da_production"][h] = da_production[h - 1]
    max_results["da_production"][h] = da_production[h - 1]

    # Calculate the mean of scenario_B_prices across all scenarios
    min_results["scenario_B_price_mean"][h] = results["B_price"][h][h,min_scenario]
    max_results["scenario_B_price_mean"][h] = results["B_price"][h][h,max_scenario]
    min_results["scenario_B_price_mean"][h] = results["B_price"][h][h,min_scenario]
    max_results["scenario_B_price_mean"][h] = results["B_price"][h][h,max_scenario]

rev_mean = sum(mean_results["revenue_T"].values())
rev_min = sum(min_results["revenue_t"].values())
rev_max = sum(max_results["revenue_t"].values())

rev_DA_mean = sum(mean_results["revenue_DA"].values())
rev_DA_min = sum(min_results["revenue_DA"].values())
rev_DA_max = sum(max_results["revenue_DA"].values())

rev_B_mean = sum(mean_results["revenue_B"].values())
rev_B_min = sum(min_results["revenue_B"].values())
rev_B_max = sum(max_results["revenue_B"].values())


print('Plts Mean')

time = [i for i in range(1, 25)]

# Plot SOC
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

axs[0].plot(time, [mean_results['soc'][t] for t in range(1, 25)], label="State of Charge (SOC)", color="blue", marker="o")
axs[0].set_xlabel("Time [hour]")
axs[0].set_ylabel("SOC")
axs[0].set_title(f"Mean SOC for all scenarios")
axs[0].grid(True)
axs[0].legend()

axs[1].bar(time, [mean_results['balancing_charge'][t] for t in range(1, 25)], label=f"Battery charge",
           color="blue", alpha=0.5)
axs[1].bar(time, [mean_results['balancing_discharge'][t] for t in range(1, 25)], label=f"Battery discharge",
           color="orange", alpha=0.5)
axs[1].plot(time, [mean_results['balancing_power'][t] for t in range(1, 25)], label=f"Mean Balancing Bid",
            color="green", marker="x")
axs[1].plot(time, [da_production[t] for t in range(0, 24)], label="DA Bid", color="green", linestyle="--")
axs[1].set_xlabel("Time (t)")
axs[1].set_xticks(time)
axs[1].set_ylabel("Power (MW)")
axs[1].set_title(f"Mean Power and Prices for all scenarios")
axs[1].grid(True)
axs[1].legend()

# Create a secondary y-axis for prices
ax2 = axs[1].twinx()
ax2.plot(time, [mean_results['scenario_B_price_mean'][t] for t in time], label=f"Mean Balancing Price", color="red",
         alpha=0.5)
ax2.plot(time, [mean_results['scenario_DA_prices'][t] for t in time], label="DA Price", color="red", linestyle="--", alpha=0.5)

# the next few lines are necessary for aligning the price and power axis
ylim = max(axs[1].get_ylim())
negative_percentage = ((min(mean_results['balancing_charge'].values()) / ylim))
negative_percentage = (min(axs[1].get_ylim()) / ylim)
ax2.set_ylim((max(ax2.get_ylim()) * negative_percentage), max(ax2.get_ylim()))  # Set y-axis limits for prices
ax2.set_ylabel("Price ($)")

lines1, labels1 = axs[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[1].legend(lines1, labels1, loc='upper left')
ax2.legend(lines2, labels2, loc='upper right')
# axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.savefig('fig_iterative_Mean.jpg',bbox_inches='tight',dpi=300)
plt.tight_layout()
plt.show()

print('Plts Min')

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

axs[0].plot(time, [min_results['soc'][t] for t in range(1, 25)], label="State of Charge (SOC)", color="blue", marker="o")
axs[0].set_xlabel("Time [hour]")
axs[0].set_ylabel("SOC")
axs[0].set_title(f"SOC for lowest production scenario")
axs[0].grid(True)
axs[0].legend()

axs[1].bar(time, [min_results['balancing_charge'][t] for t in range(1, 25)], label=f"Battery charge",
           color="blue", alpha=0.5)
axs[1].bar(time, [min_results['balancing_discharge'][t] for t in range(1, 25)], label=f"Battery discharge",
           color="orange", alpha=0.5)
axs[1].plot(time, [min_results['balancing_power'][t] for t in range(1, 25)], label=f"Balancing Bid",
            color="green", marker="x")
axs[1].plot(time, [da_production[t] for t in range(0, 24)], label="DA Bid", color="green", linestyle="--")
axs[1].set_xlabel("Time (t)")
axs[1].set_xticks(time)
axs[1].set_ylabel("Power (MW)")
axs[1].set_title(f"Power and Prices for lowest production scenario")
axs[1].grid(True)
axs[1].legend()

# Create a secondary y-axis for prices
ax2 = axs[1].twinx()
ax2.plot(time, [mean_results['scenario_B_price_mean'][t] for t in time], label=f"Balancing Price", color="red",
         alpha=0.5)
ax2.plot(time, [mean_results['scenario_DA_prices'][t] for t in time], label="DA Price", color="red", linestyle="--", alpha=0.5)

# the next few lines are necessary for aligning the price and power axis
ylim = max(axs[1].get_ylim())
negative_percentage = ((min(mean_results['balancing_charge'].values()) / ylim))
negative_percentage = (min(axs[1].get_ylim()) / ylim)
ax2.set_ylim((max(ax2.get_ylim()) * negative_percentage), max(ax2.get_ylim()))  # Set y-axis limits for prices
ax2.set_ylabel("Price ($)")

lines1, labels1 = axs[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[1].legend(lines1, labels1, loc='upper left')
ax2.legend(lines2, labels2, loc='upper right')
# axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.savefig('fig_iterative_Min.jpg',bbox_inches='tight',dpi=300)
plt.tight_layout()
plt.show()


print('Plts Max')

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

axs[0].plot(time, [max_results['soc'][t] for t in range(1, 25)], label="State of Charge (SOC)", color="blue", marker="o")
axs[0].set_xlabel("Time [hour]")
axs[0].set_ylabel("SOC")
axs[0].set_title(f"SOC for highest production scenario")
axs[0].grid(True)
axs[0].legend()

axs[1].bar(time, [max_results['balancing_charge'][t] for t in range(1, 25)], label=f"Battery charge",
           color="blue", alpha=0.5)
axs[1].bar(time, [max_results['balancing_discharge'][t] for t in range(1, 25)], label=f"Battery discharge",
           color="orange", alpha=0.5)
axs[1].plot(time, [max_results['balancing_power'][t] for t in range(1, 25)], label=f"Balancing Bid",
            color="green", marker="x")
axs[1].plot(time, [da_production[t] for t in range(0, 24)], label="DA Bid", color="green", linestyle="--")
axs[1].set_xlabel("Time (t)")
axs[1].set_xticks(time)
axs[1].set_ylabel("Power (MW)")
axs[1].set_title(f"Power and Prices for highest production scenario")
axs[1].grid(True)
axs[1].legend()

# Create a secondary y-axis for prices
ax2 = axs[1].twinx()
ax2.plot(time, [mean_results['scenario_B_price_mean'][t] for t in time], label=f"Balancing Price", color="red",
         alpha=0.5)
ax2.plot(time, [mean_results['scenario_DA_prices'][t] for t in time], label="DA Price", color="red", linestyle="--", alpha=0.5)

# the next few lines are necessary for aligning the price and power axis
ylim = max(axs[1].get_ylim())
negative_percentage = ((min(mean_results['balancing_charge'].values()) / ylim))
negative_percentage = (min(axs[1].get_ylim()) / ylim)
ax2.set_ylim((max(ax2.get_ylim()) * negative_percentage), max(ax2.get_ylim()))  # Set y-axis limits for prices
ax2.set_ylabel("Price ($)")

lines1, labels1 = axs[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[1].legend(lines1, labels1, loc='upper left')
ax2.legend(lines2, labels2, loc='upper right')
# axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.savefig('fig_iterative_Max.jpg',bbox_inches='tight',dpi=300)
plt.tight_layout()
plt.show()

print('End')
