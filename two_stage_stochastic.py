import gurobipy as gp
from gurobipy import GRB
import scenario_generation as scenario_gen


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
            generator_capacity: dict[int, dict[int, float] | dict[int, float] | dict[int, float] | dict[int, float] | dict[int, float]],
            B_price: dict[int, dict[int, float] | dict[int, float] | dict[int, float] | dict[int, float] | dict[int, float]],
            DA_price: float,
            pi: dict[int, float],
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
        # Market clearing price DA(lambda_DA)
        self.DA_price = DA_price
        # Market clearing price Balancing Market (lambda_B)
        self.B_price = B_price
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

    # Implementing only one Electricity production per Hour covers the case of not having perfect information
    def _build_variables(self):
        self.variables.generator_production = {
            (t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'Electricity production_h_{t}'
            )
            for t in self.data.TIME
        }

        self.variables.charging_power = {
            (t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'Charging power_h_{t}'
            )
            for t in self.data.TIME
        }

        self.variables.discharging_power = {
            (t): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'Discharging power_h_{t}'
            )
            for t in self.data.TIME
        }

        self.variables.soc = {
            (t, k): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'State of charge_{t}_{k}'
            )
            for k in self.data.SCENARIOS for t in self.data.TIME

        }

        self.variables.balancing_discharge = {
            (t, k): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'Balancing_discharge_{t}_{k}'
            )
            for k in self.data.SCENARIOS for t in self.data.TIME

        }

        self.variables.balancing_charge = {
            (t, k): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'Balancing_charge_{t}_{k}'
            ) for k in self.data.SCENARIOS for t in self.data.TIME

        }

        self.variables.balancing_power = {
            (t, k): self.model.addVar(
                lb=0, ub=GRB.INFINITY, name=f'Balancing power_{t}_{k}'
            )for k in self.data.SCENARIOS for t in self.data.TIME
        }

    def _build_constraints(self):


        self.constraints.balancing_power = {
            (t,k): self.model.addLConstr(
                self.data.generator_capacity[t][k] - self.variables.generator_production[t] - self.variables.balancing_charge[(t,k)] + self.variables.balancing_discharge[(t,k)],
                GRB.EQUAL,
                self.variables.balancing_power[(t,k)]
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
        }
        self.constraints.total_power_constraint = {
            (t, k): self.model.addLConstr(
                self.data.generator_capacity[t][k] - self.variables.charging_power[(t)] + self.variables.discharging_power[(t)],
                GRB.EQUAL,
                self.variables.generator_production[(t)],
                name=f'Max production constraint_{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
        }

        self.constraints.max_production_constraints = {
            (t, k): self.model.addLConstr(
                self.variables.charging_power[(t)] + self.variables.balancing_charge[(t,k)],
                GRB.LESS_EQUAL,
                self.data.generator_capacity[t][k],
                name=f'Max production constraint_{t}_{k}'
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
            ) for t in self.data.TIME
            for k in self.data.SCENARIOS
        }

        self.constraints.SOC_time = {
            (t, k): self.model.addLConstr(
                self.variables.soc[(k,t-1)] + (self.variables.charging_power[(t)] + self.variables.balancing_charge[(t,k)])* self.data.rho_charge - (self.variables.discharging_power[(t)]+self.variables.balancing_discharge[(t,k)]) * (1/self.data.rho_discharge),
                GRB.EQUAL,
                self.variables.soc[(t,k)],
                name=f'SOC time constraint_{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
            if t > 1
        }

        self.constraints.SOC_init = {
            (t, k): self.model.addLConstr(
                self.data.soc_init + (self.variables.charging_power[(t)] + self.variables.balancing_charge[(t,k)]) * self.data.rho_charge - (self.variables.discharging_power[(t)]+self.variables.balancing_discharge[(t,k)]) * (1/self.data.rho_discharge),
                GRB.EQUAL,
                self.variables.soc[(k,1)],
                name=f'SOC initial constraint{t}_{k}'
            )
            for t in self.data.TIME
            for k in self.data.SCENARIOS
        }


    def _build_objective_function(self):
        objective = (
            gp.quicksum(
                gp.quicksum(
                    (self.data.DA_price-self.data.generator_cost)*self.variables.generator_production[t]+ self.data.pi[k]*(self.data.B_price[t][k]-self.data.generator_cost)*self.variables.balancing_power[(t,k)]
                        for k in self.data.SCENARIOS
                )
                for t in self.data.TIME
            )
        )
        self.model.setObjective(objective, GRB.MAXIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='Two-stage stochastic offering strategy')
        self._build_variables()
        self._build_constraints()
        self._build_objective_function()
        #self.model.update()

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



    testdata = False
    if(testdata):

        generator_capacity_values = {
            1: {1: 88.31, 2: 136.25, 3: 85.61, 4: 137.09, 5: 146.05},
            2: {1: 134.67, 2: 96.76, 3: 139.7, 4: 96.58, 5: 117.66},
            3: {1: 112.06, 2: 86.55, 3: 79.03, 4: 115.01, 5: 115.76},
            4: {1: 85.12, 2: 122.11, 3: 83.61, 4: 80.92, 5: 90.75},
            5: {1: 111.76, 2: 141.14, 3: 135.55, 4: 75.47, 5: 118.62}
        }

        price_values =  {
            1: {1: 88.31, 2: 136.25, 3: 85.61, 4: 137.09, 5: 146.05},
            2: {1: 134.67, 2: 96.76, 3: 139.7, 4: 96.58, 5: 117.66},
            3: {1: 112.06, 2: 86.55, 3: 79.03, 4: 115.01, 5: 115.76},
            4: {1: 85.12, 2: 122.11, 3: 83.61, 4: 80.92, 5: 90.75},
            5: {1: 111.76, 2: 141.14, 3: 135.55, 4: 75.47, 5: 118.62}
        }

        input_data = InputData(
            SCENARIOS=[1, 2, 3, 4, 5],
            TIME=[1, 2, 3, 4, 5],
            generator_cost=15,
            generator_capacity=generator_capacity_values,
            DA_price= 10,
            B_price=price_values,
            pi={1: 0.2, 2: 0.2, 3: 0.25, 4: 0.25, 5: 0.25},
            rho_charge=1.5,
            rho_discharge=1.5,
            soc_max = 500,
            soc_init=50
        )
    else:
        scenario_data = scenario_gen.createScenarios()

        max_i = max(key[0] for key in scenario_data.keys())
        max_j = max(key[1] for key in scenario_data.keys())
        print(max_i, max_j)
        scenarios = [j for j in range(1, max_j+1)]
        time = [i for i in range(1, max_i+1)]
        print(time)

    #model = StochasticOfferingStrategy(input_data)
    #model.run()
    #model.display_results()
    #model_PI = StochasticOfferingStrategy(input_data)
    #model_PI.run()
    #model_PI.display_results()