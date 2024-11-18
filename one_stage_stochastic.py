import gurobipy as gp
from gurobipy import GRB


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
            generator_capacity: dict[str, dict[str, float] | dict[str, float] | dict[str, float] | dict[str, float] | dict[str, float]],
            price: dict[str, dict[str, float] | dict[str, float] | dict[str, float] | dict[str, float] | dict[str, float]],
            pi: dict[str, float],
            rho_charge: float,
            rho_discharge: float,
            soc_max: float,
            soc_init: float
    ):
        # List of scenarios
        self.SCENARIOS = SCENARIOS
        # List of time set
        self.TIME = TIME
        # Generators costs (c^G_i)
        self.generator_cost = generator_cost
        # Generators capacity (P^G_i)
        self.generator_capacity = generator_capacity
        # Market clearing price (lambda)
        self.price = price
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


class StochasticOfferingStrategy():

    def __init__(self, input_data: InputData):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        self.variables.generator_production = {
            (k, t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Electricity production'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.variables.charging_power = {
            (k, t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Charging power'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.variables.discharging_power = {
            (k, t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='Disharging power'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.variables.soc = {
            (k, t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name='State of charge'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

    def _build_constraints(self):
        self.constraints.total_power_constraint = {
            (k, t): self.model.addLConstr(
                self.data.generator_capacity[k][t] - self.variables.charging_power[k][t] + self.variables.dischaging_power[k][t],
                GRB.EQUAL,
                self.variables.generator_production[k][t],
                name=f'Max production constraint_{k}_{t}'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.constraints.max_production_constraints = {
            (k, t): self.model.addLConstr(
                self.variables.charging_power[k][t],
                GRB.LESS_EQUAL,
                self.data.generator_capacity[k][t],
                name=f'Max production constraint_{k}_{t}'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.constraints.SOC_max = {
            (k, t): self.model.addLConstr(
                self.variables.soc[k][t],
                GRB.LESS_EQUAL,
                self.data.soc_max,
                name=f'Max state of charge constraint_{k}_{t}'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.constraints.SOC_time = {
            (k, t): self.model.addLConstr(
                self.variables.soc[k][t-1] + self.variables.charging_power[k][t] * self.data.rho_charge - self.variables.discharging_power[k][t] * (1/self.data.rho_discharge),
                GRB.EQUAL,
                self.variables.soc[k][t],
                name=f'SOC time constraint_{k}_{t}'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }

        self.constraints.SOC_init = {
            (k, t): self.model.addLConstr(
                self.data.soc_init + self.variables.charging_power[k][t] * self.data.rho_charge - self.variables.discharging_power[k][t] * (1/self.data.rho_discharge),
                GRB.EQUAL,
                self.variables.soc[k][1],
                name=f'SOC initial constraint{k}_{t}'
            )
            for k in self.data.SCENARIOS
            for t in self.data.TIME
        }


    def _build_objective_function(self):
        objective = (
            gp.quicksum(
                gp.quicksum(
                    self.variables.generator_production[k][t] * self.data.pi[k] * (
                            self.data.price[k][t] - self.data.generator_cost)
                    for t in self.data.TIME
                )
                for k in self.data.SCENARIOS
            )
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='One-stage stochastic offering strategy')
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.generator_production = [
            self.variables.generator_production[k][t].x for k in self.data.SCENARIOS for t in self.data.TIME
            ]
        self.results.charging_power = [
            self.variables.charging_power[k][t].x for k in self.data.SCENARIOS for t in self.data.TIME
        ]
        self.results.discharging_power = [
            self.variables.discharging_power[k][t].x for k in self.data.SCENARIOS for t in self.data.TIME
        ]

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"optimization of {model.ModelName} was not successful")

    def display_results(self):
        print()
        print("-------------------   RESULTS  -------------------")
        print("Expected day-ahead profit:")
        print(self.results.objective_value)
        print("Optimal generator offer:")
        print(self.results.generator_production)
        print("--------------------------------------------------")
        print("Optimal charing power:")
        print(self.results.charging_power)
        print("--------------------------------------------------")
        print("Optimal discharging power:")
        print(self.results.discharging_power)
        print("--------------------------------------------------")

if __name__ == '__main__':
    generator_capacity_values = {
        'S1': {'T1': 88.31, 'T2': 136.25, 'T3': 85.61, 'T4': 137.09, 'T5': 146.05},
        'S2': {'T1': 134.67, 'T2': 96.76, 'T3': 139.7, 'T4': 96.58, 'T5': 117.66},
        'S3': {'T1': 112.06, 'T2': 86.55, 'T3': 79.03, 'T4': 115.01, 'T5': 115.76},
        'S4': {'T1': 85.12, 'T2': 122.11, 'T3': 83.61, 'T4': 80.92, 'T5': 90.75},
        'S5': {'T1': 111.76, 'T2': 141.14, 'T3': 135.55, 'T4': 75.47, 'T5': 118.62}
    }

    price_values = {
        'S1': {'T1': 88.31, 'T2': 136.25, 'T3': 85.61, 'T4': 137.09, 'T5': 146.05},
        'S2': {'T1': 134.67, 'T2': 96.76, 'T3': 139.7, 'T4': 96.58, 'T5': 117.66},
        'S3': {'T1': 112.06, 'T2': 86.55, 'T3': 79.03, 'T4': 115.01, 'T5': 115.76},
        'S4': {'T1': 85.12, 'T2': 122.11, 'T3': 83.61, 'T4': 80.92, 'T5': 90.75},
        'S5': {'T1': 111.76, 'T2': 141.14, 'T3': 135.55, 'T4': 75.47, 'T5': 118.62}
    }

    input_data = InputData(
        SCENARIOS=['S1', 'S2', 'S3', 'S4', 'S5'],
        TIME=['T1', 'T2', 'T3', 'T4', 'T5'],
        generator_cost=15,
        generator_capacity=generator_capacity_values,
        price=price_values,
        pi={'S1': 0.25, 'S2': 0.25, 'S3': 0.25, 'S4': 0.25},
        rho_charge=1.5,
        rho_discharge=1.5,
        soc_max = 500,
        soc_init=50
    )
    model = StochasticOfferingStrategy(input_data)
    model.run()
    model.display_results()
    model_PI = StochasticOfferingStrategy(input_data)
    model_PI.run()
    model_PI.display_results()