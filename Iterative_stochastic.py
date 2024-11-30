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
            pi: dict[int, float],
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

    #Save results is not changed yet
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
                print(f"Constraint {c.ConstrName}: Slack={c.slack}")
            raise RuntimeError(f"optimization of {self.model.ModelName} was not successful")


    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected day-ahead profit:")
        print(self.results.objective_value)
        print("Optimal generator offer:")
        print(self.results.generator_production)
        print("--------------------------------------------------")
        print("Optimal balancing offer:")
        #print(self.results.balancing_charge)
        for k in self.data.SCENARIOS:
            print(f"Scenario {k}: ", end="")
            for t in self.data.TIME:
                print(round(self.results.balancing_power[(t, k)],1), end=", ")
            print()
        print("--------------------------------------------------")
        print("Optimal balancing charging power:")
        #print(self.results.balance_charge)
        for k in self.data.SCENARIOS:
            print(f"Scenario {k}: ", end="")
            for t in self.data.TIME:
                print(round(self.results.balancing_charge[(t, k)],1), end=", ")
            print()
        print("--------------------------------------------------")
        print("Optimal balancing discharging power:")
        #print(self.results.balancing_discharge)
        for k in self.data.SCENARIOS:
            print(f"Scenario {k}: ", end="")
            for t in self.data.TIME:
                print(round(self.results.balancing_discharge[(t, k)],1), end=", ")
            print()
        print("--------------------------------------------------")

    def plot_results(self):
        """
        Plots the SOC and other results such as charging/discharging power, DA bid, and balancing power.
        """
        time = self.data.TIME
        scenarios = self.data.SCENARIOS

        # Extract SOC
        soc = {t: sum(self.results.soc[(t, k)] for k in scenarios) / len(scenarios) for t in time}

        # DA STAGE
        # Extract charging and discharging power
        da_bid = {t: self.results.generator_production[t-1] for t in time}
        da_price = {t: self.data.da_price[t-1] for t in time}

        # Extract balancing bid (averaged over scenarios)
        #balancing_bid = {t: sum(self.results.balancing_power[(t, k)] for k in scenarios) / len(scenarios) for t in time}
        balancing_bid = {(t,k): self.results.balancing_power[(t, k)] for t in time for k in scenarios}
        balancing_charge = {(t,k): self.results.balancing_charge[(t, k)] for t in time for k in scenarios}
        balancing_discharge = {(t,k): self.results.balancing_discharge[(t, k)] for t in time for k in scenarios}
        balancing_price = {(t,k): self.data.b_price[(t,k)] for t in time for k in scenarios}


        # Plot SOC
        plt.figure(figsize=(10, 6))
        plt.plot(time, [soc[t] for t in time], label="State of Charge (SOC)", color="blue", marker="o")
        plt.xlabel("Time (t)")
        plt.ylabel("SOC")
        plt.title("State of Charge Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        # inspected scenario
        scenario = 1
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(time, [balancing_bid[t, scenario] for t in time], label=f"Balancing Bid for S{scenario}",
                 color="green", marker="x")
        ax1.plot(time, [balancing_charge[t, scenario] for t in time], label=f"Balancing charge S{scenario}",
                 color="blue", alpha=0.5)
        ax1.plot(time, [balancing_discharge[t, scenario] for t in time], label=f"Balancing discharge S{scenario}",
                 color="orange", alpha=0.5)
        ax1.plot(time, [da_bid[t] for t in time], label="DA Bid", color="green", linestyle="--")
        ax1.set_xlabel("Time (t)")
        ax1.set_xticks(time)
        ax1.set_ylabel("Power (MW)")
        ax1.set_title("Day-Ahead, Charging/Discharging, and Balancing Power")
        ax1.grid(True)

        # Create a secondary y-axis for prices
        ax2 = ax1.twinx()
        ax2.plot(time, [balancing_price[t, scenario] for t in time], label=f"Balancing Price S{scenario}", color="red",
                 alpha=0.5)
        ax2.plot(time, [da_price[t] for t in time], label="DA Price", color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Price ($)")

        # Add legends for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='upper left')
        ax2.legend(lines2, labels2, loc='upper right')

        plt.show()



if __name__ == '__main__':
    testdata = False
    if(testdata):
        generator_availability_values = {
            (1, 1): 33, (1, 2): 30, (1, 3): 27,
            (1, 4): 21, (1, 5): 20,
            (2, 1): 20, (2, 2): 21, (2, 3): 31,
            (2, 4): 17, (2, 5): 10,
            (3, 1): 33, (3, 2): 27, (3, 3): 50,
            (3, 4): 19, (3, 5): 80,
            (4, 1): 21, (4, 2): 36, (4, 3): 50,
            (4, 4): 50, (4, 5): 50,
            (5, 1): 25, (5, 2): 39, (5, 3): 50,
            (5, 4): 50, (5, 5): 50
        }

        b_price = {
            (1, 1): 88.31, (1, 2): 136.25, (1, 3): 85.61, (1, 4): 137.09, (1, 5): 146.05,
            (2, 1): 134.67, (2, 2): 96.76, (2, 3): 139.7, (2, 4): 96.58, (2, 5): 117.66,
            (3, 1): 112.06, (3, 2): 86.55, (3, 3): 79.03, (3, 4): 115.01, (3, 5): 115.76,
            (4, 1): 85.12, (4, 2): 122.11, (4, 3): 83.61, (4, 4): 80.92, (4, 5): 90.75,
            (5, 1): 111.76, (5, 2): 141.14, (5, 3): 135.55, (5, 4): 75.47, (5, 5): 118.62
        }

        da_price = {
            (1, 1): 51.49, (1, 2): 48.39, (1, 3): 48.92, (1, 4): 49.45, (1, 5): 42.72,
            (2, 1): 50.84, (2, 2): 82.15, (2, 3): 100.96, (2, 4): 116.60, (2, 5): 112.20,
            (3, 1): 108.54, (3, 2): 111.61, (3, 3): 114.02, (3, 4): 127.40, (3, 5): 134.02,
            (4, 1): 142.18, (4, 2): 147.42, (4, 3): 155.91, (4, 4): 154.10, (4, 5): 148.30,
            (5, 1): 138.59, (5, 2): 129.44, (5, 3): 122.89, (5, 4): 112.47, (5, 5): 112.47
        }
        da_price = [51.49, 48.39, 82.15, 116.6, 42]

        input_data = InputData(
            #SCENARIOS=[1, 2, 3, 4, 5],
            SCENARIOS=[1],
            TIME=[1, 2, 3, 4, 5],
            generator_cost=20,
            generator_capacity= 48,
            generator_availability=generator_availability_values,
            da_price= da_price,
            b_price=b_price,
            pi=0.2,
            rho_charge=0.9,
            rho_discharge=0.9,
            soc_max = 120,
            soc_init=10,
            charging_capacity= 40
        )
    else:
        num_wind_scenarios = 1
        num_price_scenarios = 1
        SCENARIOS = num_wind_scenarios * num_price_scenarios
        Scenarios = [i for i in range(1, SCENARIOS + 1)]
        Time = [i for i in range(1, 25)]

        scescenario_DA_prices = {}
        scenario_B_prices = {}
        scenario_windProd = {}
        scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(num_price_scenarios, num_wind_scenarios)
        scenario_DA_prices = [
            51.49, 48.39, 48.92, 49.45, 42.72, 50.84, 82.15, 100.96, 116.60,
            112.20, 108.54, 111.61, 114.02, 127.40, 134.02, 142.18, 147.42,
            155.91, 154.10, 148.30, 138.59, 129.44, 122.89, 112.47
        ]

        #Equal probability of each scenario 1/100
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
            rho_charge=0.9,  # TODO what were the rho values before??
            rho_discharge=0.9,
            soc_max=120,
            soc_init=10,
            charging_capacity=100
        )

    model = StochasticOfferingStrategy(input_data)
    model.run()
    model.display_results()
    model.plot_results()


    # model_PI = StochasticOfferingStrategy(input_data)
    # model_PI.run()
    # model_PI.display_results()