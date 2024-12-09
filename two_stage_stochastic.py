import gurobipy as gp
from gurobipy import GRB
from scenario_generation_function import generate_scenarios
import matplotlib.pyplot as plt


class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


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
            pi: float,
            rho_charge: float,
            rho_discharge: float,
            soc_max: float,
            soc_init: float,
            charging_capacity: float
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


class StochasticOfferingStrategy():

    def __init__(self, input_data: InputData):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        self.variables.generator_production = {
            t: self.model.addVar(
                lb=0, ub=self.data.generator_capacity, name=f'DA production_h_{t}'
            )
            for t in self.data.TIME
        }

        self.variables.soc = {
            (t, k): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'State of charge_{t}_{k}'
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
                self.variables.generator_production[t] + self.variables.balancing_power[(t,k)] + self.variables.balancing_charge[(t,k)],
                name=f'Balancing power constraint_{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
        }

        self.constraints.SOC_max = {
            (t, k): self.model.addLConstr(
                self.variables.soc[(t,k)],
                GRB.LESS_EQUAL,
                self.data.soc_max,
                name=f'Max state of charge constraint_{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
        }

        self.constraints.SOC_time = {
            (t, k): self.model.addLConstr(
                self.variables.soc[(t-1,k)] + (self.variables.balancing_charge[(t,k)])* self.data.rho_charge
                - (self.variables.balancing_discharge[(t,k)]) * (1/self.data.rho_discharge),
                GRB.EQUAL,
                self.variables.soc[(t,k)],
                name=f'SOC time constraint_{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
            if t > 1
        }

        self.constraints.SOC_init = {
            (k): self.model.addLConstr(
                self.data.soc_init + ( self.variables.balancing_charge[(1,k)]) * self.data.rho_charge
                - ( self.variables.balancing_discharge[(1,k)]) * (1/self.data.rho_discharge),
                GRB.EQUAL,
                self.variables.soc[(1,k)],
                name=f'SOC initial constraint{1}_{k}'
            )
            for k in self.data.SCENARIOS
        }

        self.constraints.DA_constraint_availability = {
            t: self.model.addLConstr(
                self.variables.generator_production[t],
                GRB.LESS_EQUAL,
                self.data.generator_availability[(t, k)],
                name=f'p_DA constraint_availability_max{t}_{k}'
            )
            for t in Time
            for k in self.data.SCENARIOS
        }


    def _build_objective_function(self):
        objective = (
            gp.quicksum(
                gp.quicksum(
                    (self.data.da_price[(t-1)] - self.data.generator_cost) * self.variables.generator_production[t]
                    + self.data.pi * (self.data.b_price[(t, k)] - self.data.generator_cost) * self.variables.balancing_power[(t, k)]
                    for t in self.data.TIME
                )
                for k in self.data.SCENARIOS
            )
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='Two-stage stochastic offering strategy')
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.generator_production = [
            self.variables.generator_production[t].x for t in self.data.TIME
            ]
        self.results.balancing_power = {
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
                print(f"Constraint {c.ConstrName}")
            raise RuntimeError(f"optimization of {self.model.ModelName} was not successful")

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected day-ahead profit:")
        print(self.results.objective_value)
        print("Optimal DA bid:")
        print(self.results.generator_production)
        print("--------------------------------------------------")
        print("Optimal balancing offer:")
        for k in self.data.SCENARIOS:
            print(f"Scenario {k}: ", end="")
            for t in self.data.TIME:
                print(round(self.results.balancing_power[(t, k)],1), end=", ")
            print()
        print("--------------------------------------------------")
        print("Optimal balancing charging power:")
        for k in self.data.SCENARIOS:
            print(f"Scenario {k}: ", end="")
            for t in self.data.TIME:
                print(round(self.results.balancing_charge[(t, k)],1), end=", ")
            print()
        print("--------------------------------------------------")
        print("Optimal balancing discharging power:")
        for k in self.data.SCENARIOS:
            print(f"Scenario {k}: ", end="")
            for t in self.data.TIME:
                print(round(self.results.balancing_discharge[(t, k)],1), end=", ")
            print()
        print("--------------------------------------------------")

    def get_extreme_scenarios(self, scenario_windProd):
        # Calculate total wind production for each scenario
        total_wind_production = {k: sum(scenario_windProd[(t, k)] for t in range(1, 25)) for k in
                                 range(1, len(self.data.SCENARIOS) + 1)}

        # Find the scenario with the minimum wind production
        min_scenario = min(total_wind_production, key=total_wind_production.get)
        min_wind_production = total_wind_production[min_scenario]

        # Find the scenario with the maximum wind production
        max_scenario = max(total_wind_production, key=total_wind_production.get)
        max_wind_production = total_wind_production[max_scenario]

        # Find the scenario with the median wind production
        sorted_scenarios = sorted(total_wind_production.items(), key=lambda item: item[1])
        median_index = len(sorted_scenarios) // 2
        median_scenario = sorted_scenarios[median_index][0]
        median_wind_production = sorted_scenarios[median_index][1]

        print(f"Scenario with minimum wind production: {min_scenario} (Total: {min_wind_production})")
        print(f"Scenario with maximum wind production: {max_scenario} (Total: {max_wind_production})")
        print(f"Scenario with median wind production: {median_scenario} (Total: {median_wind_production})")

    def plot_results(self, scenario_num):
        """
        Plots the SOC and other results such as charging/discharging power, DA bid, and balancing power.
        """
        # inspected scenario
        scenario = scenario_num
        time = self.data.TIME
        scenarios = self.data.SCENARIOS

        # Extract SOC
        soc = soc = {(t, k): self.results.soc[(t, k)] for t in time for k in scenarios}

        # DA STAGE
        # Extract charging and discharging power
        da_bid = {t: self.results.generator_production[t - 1] for t in time}
        da_price = {t: self.data.da_price[t - 1] for t in time}

        # Extract balancing bid (averaged over scenarios)
        balancing_bid = {(t, k): self.results.balancing_power[(t, k)] for t in time for k in scenarios}
        balancing_charge = {(t, k): self.results.balancing_charge[(t, k)] for t in time for k in scenarios} \
        # multiply each element of balancing chaarge by -1
        balancing_charge = {key: -value for key, value in balancing_charge.items()}
        balancing_discharge = {(t, k): self.results.balancing_discharge[(t, k)] for t in time for k in scenarios}
        balancing_price = {(t, k): self.data.b_price[(t, k)] for t in time for k in scenarios}

        # Plot SOC
        fig, axs = plt.subplots(2, 1, figsize=(12, 12))


        axs[0].plot(time, [soc[t, scenario] for t in time], label="State of Charge (SOC)", color="blue", marker="o")
        axs[0].set_xlabel("Time (t)")
        axs[0].set_ylabel("SOC")
        axs[0].set_title(f"SOC for S{scenario}")
        axs[0].grid(True)
        axs[0].legend()


        axs[1].bar(time, [balancing_charge[t, scenario] for t in time], label=f"Balancing charge S{scenario}",
                   color="blue", alpha=0.5)
        axs[1].bar(time, [balancing_discharge[t, scenario] for t in time], label=f"Balancing discharge S{scenario}",
                   color="orange", alpha=0.5)
        axs[1].plot(time, [balancing_bid[t, scenario] for t in time], label=f"Balancing Bid for S{scenario}",
                    color="green", marker="x")
        axs[1].plot(time, [da_bid[t] for t in time], label="DA Bid", color="green", linestyle="--")
        axs[1].set_xlabel("Time (t)")
        axs[1].set_xticks(time)
        axs[1].set_ylabel("Power (MW)")
        axs[1].set_title(f"Power  and Prices for S{scenario}")
        axs[1].grid(True)
        axs[1].legend()

        # Create a secondary y-axis for prices
        ax2 = axs[1].twinx()
        ax2.plot(time, [balancing_price[t, scenario] for t in time], label=f"Balancing Price S{scenario}", color="red",
                 alpha=0.5)
        ax2.plot(time, [da_price[t] for t in time], label="DA Price", color="red", linestyle="--", alpha=0.5)

        # the next few lines are necessary for aligning the price and power axis
        ylim = max(axs[1].get_ylim())
        negative_percentage = ((min(balancing_charge.values()) /ylim))
        negative_percentage = (min(axs[1].get_ylim()) /ylim)
        ax2.set_ylim((max(ax2.get_ylim())*negative_percentage), max(ax2.get_ylim()))  # Set y-axis limits for prices
        ax2.set_ylabel("Price ($)")


        lines1, labels1 = axs[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[1].legend(lines1, labels1, loc='upper left')
        ax2.legend(lines2, labels2, loc='upper right')
        #axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        #plt.savefig(f'figure/s1_scenario{scenario}.jpg',bbox_inches='tight',dpi=300)
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    num_wind_scenarios = 20
    num_price_scenarios = 20
    SCENARIOS = num_wind_scenarios * num_price_scenarios
    Scenarios = [i for i in range(1, SCENARIOS + 1)]
    Time = [i for i in range(1, 25)]

    scenario_DA_prices = {}
    scenario_B_prices = {}
    scenario_windProd = {}
    scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(num_price_scenarios, num_wind_scenarios)
    scenario_DA_prices = [
        51.49, 48.39, 48.92, 49.45, 42.72, 50.84, 82.15, 100.96, 116.60,
        112.20, 108.54, 111.61, 114.02, 127.40, 134.02, 142.18, 147.42,
        155.91, 154.10, 148.30, 138.59, 129.44, 122.89, 112.47
    ]

    #Equal probability of each scenario
    pi = 1/SCENARIOS

    input_data = InputData(
        SCENARIOS=Scenarios,
        TIME=Time,
        generator_cost=20,
        generator_capacity=48,
        generator_availability=scenario_windProd,
        da_price=scenario_DA_prices,
        b_price=scenario_B_prices,
        pi=pi,
        rho_charge=0.8332,
        rho_discharge=0.8332,
        soc_max=120,
        soc_init=10,
        charging_capacity=100
    )

    model = StochasticOfferingStrategy(input_data)
    model.run()
    model.display_results()

    model.get_extreme_scenarios(scenario_windProd)
    # model.plot_results(scenario_num=19)
    # model.plot_results(scenario_num=1)

    model_PI = StochasticOfferingStrategy(input_data)
    model_PI.run()
    model_PI.display_results()