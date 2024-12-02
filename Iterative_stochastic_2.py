import gurobipy as gp
from gurobipy import GRB
from scenario_generation_function import generate_scenarios
import matplotlib.pyplot as plt


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
        num_wind_scenarios = 2
        num_price_scenarios = 3
        SCENARIOS = num_wind_scenarios * num_price_scenarios
        Scenarios = [i for i in range(1, SCENARIOS + 1)]
        Time = [1]

        scescenario_DA_prices = {}
        scenario_B_prices = {}
        scenario_windProd = {}
        scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(num_price_scenarios,
                                                                                      num_wind_scenarios)
        scenario_DA_prices = [
            51.49, 48.39, 48.92, 49.45, 42.72, 50.84, 82.15, 100.96, 116.60,
            112.20, 108.54, 111.61, 114.02, 127.40, 134.02, 142.18, 147.42,
            155.91, 154.10, 148.30, 138.59, 129.44, 122.89, 112.47
        ]
        da_production = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.549490884357137, 32.76784850051348, 31.064958809504038,
                         31.332530016698215, 30.88555999023277, 28.76005112309651, 28.9701378870548, 0.0,
                         32.196188026852454, 0.0, 48.0, 0.0, 0.0, 33.55400332783193, 0.0, 0.0, 0.0]
        SOC = [0 for i in range(1, 25)]

        # Equal probability of each scenario 1/100
        pi = 1 / SCENARIOS

        input_data = InputData(
            SCENARIOS=Scenarios,
            TIME=Time,
            generator_cost=20,
            generator_capacity=48,
            generator_availability=scenario_windProd,
            generator_DAbid=da_production[h],
            da_price=scenario_DA_prices[h],
            b_price=scenario_B_prices,
            pi=pi,
            rho_charge=0.9,  # TODO what were the rho values before??
            rho_discharge=0.9,
            SOC=SOC,
            soc_max=120,
            soc_init=10,
            charging_capacity=100
        )


    print('End Input Data')

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
                    self.variables.soc[(t-1,k)] + (self.variables.balancing_charge[(t,k)])* self.data.rho_charge
                    - (self.variables.balancing_discharge[(t,k)]) * (1/self.data.rho_discharge),
                    GRB.EQUAL,
                    self.variables.soc[(t,k)],
                    name=f'SOC time constraint_{t}_{k}'
                )
                for k in self.data.SCENARIOS
                for t in self.data.TIME
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
            self.results.objective_value = self.model.ObjVal
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



    model = StochasticOfferingStrategy(input_data)
    model.run()
    #model.display_results()

print('End')