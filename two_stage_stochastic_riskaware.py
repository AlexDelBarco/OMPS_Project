import gurobipy as gp
from gurobipy import GRB
import numpy as np
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

    def __init__(self, input_data: InputData, risk_averse: bool = False, alpha: float = 0, beta: float = 0):
        self.data = input_data
        self.risk_averse = risk_averse
        self.alpha = alpha
        self.beta = beta
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

        if self.risk_averse:
            # self.variables.zeta = {
            #     t : self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'zeta_{t}')
            #     for t in self.data.TIME
            # }
            self.variables.zeta = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='zeta')

            # self.variables.eta = {
            #     (t, k) : self.model.addVar(lb=0, name=f'eta_{t}_{k}')
            #     for t in self.data.TIME
            #     for k in self.data.SCENARIOS
            # }
            self.variables.eta = {
                (k) : self.model.addVar(lb=0, name=f'eta_{k}')
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
            t: self.model.addLConstr(self.variables.generator_production[t], GRB.LESS_EQUAL,
                self.data.generator_availability[(t, k)], name=f'p_DA constraint_availability_max{t}_{k}') for t in Time
            for k in self.data.SCENARIOS}

        if self.risk_averse:
            self.constraints.eta_constraint = {
                (t,k):self.model.addLConstr(
                    self.variables.zeta-self.variables.eta[(k)]
                    -((self.data.b_price[(t, k)] - self.data.generator_cost)*self.variables.balancing_power[(t,k)])
                    ,GRB.LESS_EQUAL
                    ,0
                )
                for t in self.data.TIME
                for k in self.data.SCENARIOS
            }

    def _build_objective_function(self):

        if self.risk_averse:

            objective = (
                    (1 - self.beta) * (
                gp.quicksum(
                    gp.quicksum(
                        (self.data.da_price[t,k] - self.data.generator_cost) * self.variables.generator_production[t]
                        + self.data.pi * (self.data.b_price[t, k] - self.data.generator_cost) *
                        self.variables.balancing_power[t, k]
                        for t in self.data.TIME
                    )
                    for k in self.data.SCENARIOS
                )
            )
                    + self.beta * (
                            self.variables.zeta - (1 / (1 - self.alpha)) * gp.quicksum(
                        self.data.pi * self.variables.eta[k] for k in self.data.SCENARIOS
                        )
                    )
            )



            #objective = DA_Profit + (1-self.beta) * (BA_Profit) + self.beta * CVaR
            self.model.setObjective(objective, GRB.MAXIMIZE)

        else:
            objective = (gp.quicksum(gp.quicksum(
                (self.data.da_price[(t - 1)] - self.data.generator_cost) * self.variables.generator_production[
                    t] + self.data.pi * (self.data.b_price[(t, k)] - self.data.generator_cost) *
                self.variables.balancing_power[(t, k)] for t in self.data.TIME) for k in self.data.SCENARIOS))
            self.model.setObjective(objective, GRB.MAXIMIZE)

    def _build_model(self):
        self.model = gp.Model(name='Two-stage stochastic offering strategy - Risk Aware')
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

        if self.risk_averse:
            self.results.zeta ={
                t : self.variables.zeta.x for t in self.data.TIME
            }
            self.results.cvar = {
                t: self.variables.zeta.x - 1 / (1 - self.alpha) * np.sum(self.data.pi * self.variables.eta[(k)].x for k in self.data.SCENARIOS)
                for t in self.data.TIME
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
        if self.risk_averse:
            print()
            print("Risk-Aware Optimization")
            print("Alpha: "+ str(self.alpha))
            print("Beta: "+ str(self.beta))
            print("--------------------------------------------------")
            for t in self.data.TIME:
                 print(f"CVaR at Time {t}: "+str(self.results.cvar[t]), end="")
                 print(f"Zeta VaR: "+str(self.results.zeta[t]))
                 print()







def run_riskaware_optimization(input_data, alpha_values, beta_values):
    results = []
    for alpha in alpha_values:
        for beta in beta_values:
            model = StochasticOfferingStrategy(input_data, risk_averse=True, alpha=alpha, beta=beta)
            model.run()
            results.append({
                'alpha': alpha,
                'beta': beta,
                'objective_value': model.results.objective_value,
                #'generator_production': model.results.generator_production,
                #'balancing_power': model.results.balancing_power,
                #'balancing_charge': model.results.balancing_charge,
                #'balancing_discharge': model.results.balancing_discharge,
                #'soc': model.results.soc,
                #'zeta': model.results.zeta if model.risk_averse else None,
                #'cvar': model.results.cvar if model.risk_averse else None
            })
    return results


def plot_results(results):
    """
    Plots the SOC and other results such as charging/discharging power, DA bid, and balancing power.
    """

    # Extract beta and objective values
    alpha = results[0]['alpha']
    beta_values = [entry['beta'] for entry in results]
    objective_values = [entry['objective_value'] for entry in results]

    scaled_values = [value / 10**6 for value in objective_values]
    normalized_values = [value / objective_values[0] for value in objective_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, scaled_values, marker='o', label=rf'Expected Profit at $\alpha$ = {alpha}')
    plt.title(r'Expected Profit($\beta$)', fontsize=14)
    plt.xlabel(r'$\beta$ [-]', fontsize=12)
    plt.ylabel('Expected Profit [mio €]', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, normalized_values, marker='o', label='Expected Profit')
    plt.title(r'Normalised Expected Profit($\beta$)', fontsize=14)
    plt.xlabel(r'$\beta$ [-]', fontsize=12)
    plt.ylabel('Expected Profit [-]', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
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
            SCENARIOS=[1,2],
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
        num_wind_scenarios = 20
        num_price_scenarios = 20
        SCENARIOS = num_wind_scenarios * num_price_scenarios
        Scenarios = [i for i in range(1, SCENARIOS + 1)]
        Time = [i for i in range(1, 25)]

        scenario_DA_prices = {}
        scenario_B_prices = {}
        scenario_windProd = {}
        scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(num_price_scenarios, num_wind_scenarios)

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
            pi = pi,
            rho_charge=0.8332,  # see source pdf table 3
            rho_discharge=0.8332,
            soc_max=120,
            soc_init=10,
            charging_capacity=100
        )


    # Compute expected profits for a given alpha for beta 0-0.95:
    alpha_values = [0.95]
    beta_values = np.arange(0, 0.85, 0.05)
    #beta_values = [0.1]
    #results = [{'alpha': 0.95, 'beta': 0.0, 'objective_value': 20792224.527455278}, {'alpha': 0.95, 'beta': 0.05, 'objective_value': 19752613.301082462}, {'alpha': 0.95, 'beta': 0.1, 'objective_value': 18713002.074709874}, {'alpha': 0.95, 'beta': 0.15000000000000002, 'objective_value': 17673390.848336924}, {'alpha': 0.95, 'beta': 0.2, 'objective_value': 16633779.62196417}, {'alpha': 0.95, 'beta': 0.25, 'objective_value': 15594168.395591494}, {'alpha': 0.95, 'beta': 0.30000000000000004, 'objective_value': 14554557.169218712}, {'alpha': 0.95, 'beta': 0.35000000000000003, 'objective_value': 13514945.942846004}, {'alpha': 0.95, 'beta': 0.4, 'objective_value': 12475334.716473123}, {'alpha': 0.95, 'beta': 0.45, 'objective_value': 11435723.49010023}, {'alpha': 0.95, 'beta': 0.5, 'objective_value': 10396112.263727639}, {'alpha': 0.95, 'beta': 0.55, 'objective_value': 9356501.037354937}, {'alpha': 0.95, 'beta': 0.6000000000000001, 'objective_value': 8316889.810982084}, {'alpha': 0.95, 'beta': 0.65, 'objective_value': 7277278.584609356}, {'alpha': 0.95, 'beta': 0.7000000000000001, 'objective_value': 6237667.35823656}, {'alpha': 0.95, 'beta': 0.75, 'objective_value': 5198056.131863819}, {'alpha': 0.95, 'beta': 0.8, 'objective_value': 4158444.905491042}]



    results = run_riskaware_optimization(input_data, alpha_values, beta_values)
    #plot_results(results)
    print(results)