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

    def __init__(self, input_data: InputData, epsilon):
        self.data = input_data
        self.epsilon = epsilon
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

        self.big_M = 10000
        self.variables.binary_balance = {
            k: self.model.addVar(
                vtype=GRB.INTEGER, name='Binary var for balance constraint'
            ) for k in self.data.SCENARIOS
        }

        self.variables.binary_da = {
            k: self.model.addVar(
                vtype=GRB.INTEGER, name='Binary var for da constraint'
            ) for k in self.data.SCENARIOS
        }

    def _build_constraints(self):

        self.constraints.balancing_chance_constraint1 = {
            (t, k): self.model.addLConstr(
                self.data.generator_availability[(t,k)] + self.variables.balancing_discharge[(t,k)]
                -  self.variables.generator_production[t] - self.variables.balancing_power[(t,k)] - self.variables.balancing_charge[(t,k)],
                GRB.LESS_EQUAL,
                self.big_M * (1 - self.variables.binary_balance[k]),
                name=f'Balancing power chance constraint1_{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
        }

        self.constraints.balancing_chance_constraint2 = {
            (t, k): self.model.addLConstr(
                self.data.generator_availability[(t,k)] + self.variables.balancing_discharge[(t,k)]
                -  self.variables.generator_production[t] - self.variables.balancing_power[(t,k)] - self.variables.balancing_charge[(t,k)],
                GRB.GREATER_EQUAL,
                -(self.big_M * (1 - self.variables.binary_balance[k])),
                name=f'Balancing power chance constraint2_{t}_{k}'
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

        self.constraints.DA_chance_constraint_availability = {
            (t, k): self.model.addLConstr(
                self.variables.generator_production[t] - self.data.generator_availability[(t, k)],
                GRB.LESS_EQUAL,
                self.big_M * (1 - self.variables.binary_da[k]),
                name=f'p_DA chance_constraint_availability_max{t}_{k}'
            )
            for t in Time
            for k in self.data.SCENARIOS
        }

        #constraints for binary variables
        self.constraints.binary_balance = self.model.addLConstr(
            gp.quicksum(
                self.variables.binary_balance[k] for k in self.data.SCENARIOS
            ) / len(self.data.SCENARIOS),
            GRB.GREATER_EQUAL,
            1 - self.epsilon,
            name='Binary balance constraint',
        )

        self.constraints.binary_da = self.model.addLConstr(
            gp.quicksum(
                self.variables.binary_da[k] for k in self.data.SCENARIOS
            ) / len(self.data.SCENARIOS),
            GRB.GREATER_EQUAL,
            1 - self.epsilon,
            name='Binary DA constraint',
        )

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
        print("Optimal generator offer:")
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

def plot_results(epsilon_values, revenues):
    plt.figure(figsize=(8, 5))
    plt.plot(epsilon_values, revenues, marker='o', label="Revenue")
    plt.xlabel('Epsilon')
    plt.ylabel('Revenue')
    plt.title('Epsilon vs Revenue')
    plt.grid(True)
    plt.legend()

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
        rho_charge=0.8332,
        rho_discharge=0.8332,
        soc_max=120,
        soc_init=10,
        charging_capacity=100
    )

    epsilon_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    revenues = []

    for epsilon in epsilon_values:
        model = StochasticOfferingStrategy(input_data, epsilon=epsilon)
        model.run()
        revenues.append(model.results.objective_value)
    plot_results(epsilon_values, revenues)
    print(revenues)