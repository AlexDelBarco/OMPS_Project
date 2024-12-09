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
}

SOC_st = []
SOC_st.append(10)

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
        num_wind_scenarios = 1
        num_price_scenarios = 5
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

        SOC = SOC_st[-1]


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
            rho_charge=0.9,  # TODO what were the rho values before??
            rho_discharge=0.9,
            SOC=SOC,
            soc_max=120,
            soc_init=10, # I dont think I need it
            charging_capacity=100
        )
    print('HEEEEEEEREEEEEEEEEE')
    print(f'Wind Prod for {h} {scenario_windProd}')
    print(f'DA prod for {h} {da_production[h-1]}')
    print(f'DA prices for {h} {scenario_DA_prices[h-1]}')
    print(f'B prices for {h} {scenario_B_prices}')
    print(f'SOC for {h} {SOC}')
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
                    self.data.SOC + (self.variables.balancing_charge[(t,k)])* self.data.rho_charge
                    - (self.variables.balancing_discharge[(t,k)]) * (1/self.data.rho_discharge),
                    GRB.EQUAL,
                    self.variables.soc[(t,k)],
                    name=f'SOC time constraint_{t}_{k}'
                )
                for k in self.data.SCENARIOS
                for t in self.data.TIME
            }

            #self.constraints.SOC_init = {
            #    (k): self.model.addLConstr(
            #        self.data.soc_init + ( self.variables.balancing_charge[(1,k)]) * self.data.rho_charge
            #        - ( self.variables.balancing_discharge[(1,k)]) * (1/self.data.rho_discharge),
            #        GRB.EQUAL,
            #        self.variables.soc[(1,k)],
            #        name=f'SOC initial constraint{1}_{k}'
            #    )
            #    for k in self.data.SCENARIOS
            #}

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

    soc_value = sum(model.results.soc[(h, k)] for k in model.data.SCENARIOS)
    SOC_st.append(soc_value/SCENARIOS)
    #model.display_results()

print('End')