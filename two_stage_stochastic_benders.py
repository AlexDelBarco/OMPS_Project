import random
from random import randint

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scenario_generation_function import generate_scenarios
import matplotlib.pyplot as plt

# This file approaches to initialize the benders decomposition algorithm, for the sake of simplicity it is only done for a single hour

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
        self.variables_Main = Expando()
        self.variables_sub = Expando()
        self.constraints_Main = Expando()
        self.constraints_sub = Expando()
        self.results = Expando()


    def _build_variables(self, model):
        if model.ModelName == "Main-Problem":
            self.variables_Main.generator_production = {
                t: model.addVar(
                    lb=0, ub=self.data.generator_capacity, name=f'DA production_h_{t}'
                )
                for t in self.data.TIME
            }
            self.variables_Main.gamma = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gamma")


        if model.ModelName == "Sub-Problem":
            # p_DA is fixed and p_B should be solved for the subproblem and then handed over back to thr main problem to determine the next fixed p_DA
            #p_B
            self.variables_sub.p_B = {
                t: model.addVar(
                    lb=0, ub=self.data.generator_capacity, name=f'Balancing power_h_{t}'
                )
                for t in self.data.TIME
            }
            # p_DA
            self.variables_sub.p_DA = {
                t: model.addVar(
                    lb=0, ub=self.data.generator_capacity, name=f'DA power_h_{t}'
                )
                for t in self.data.TIME
            }
            #p_dis
            self.variables_sub.p_dis = {
                t: model.addVar(
                    lb=0, ub=self.data.charging_capacity, name=f'Balancing discharge_h_{t}'
                )
                for t in self.data.TIME
            }
            #p_cha
            self.variables_sub.p_cha = {
                t: model.addVar(
                    lb=0, ub=self.data.charging_capacity, name=f'Balancing charge_h{t}'
                )
                for t in self.data.TIME
            }
            #SOC
            self.variables_sub.soc = {
                t: model.addVar(
                    lb=0, ub=self.data.soc_max,name=f'SOC h_{t}'
                )
                for t in self.data.TIME
            }


    def _build_constraints(self, model, k, p_DA_fixed, new_gamma):
        if(model.ModelName == "Main-Problem"):
            self.constraints_Main.gamma = model.addLConstr(
                self.variables_Main.gamma,
                GRB.LESS_EQUAL,
                new_gamma,      #TODO what is new gamma?
                name=f'Gamma contraint_{k}')

        if model.ModelName == "Sub-Problem":
            self.constraints_sub.balancing_constraint = {
                t: model.addLConstr(
                    self.data.generator_availability[(t, k)] + self.variables_sub.p_dis[t],
                    GRB.EQUAL,
                    self.variables_sub.p_DA[t] + self.variables_sub.p_B[t] + self.variables_sub.p_cha[t],
                    name=f'Balancing power constraint_{t}_{k}'
                )
                for t in self.data.TIME
            }

            self.constraints_sub.DA_constraint = {
                t: model.addLConstr( self.variables_sub.p_DA[t],
                    GRB.EQUAL,
                    p_DA_fixed[t],
                    name=f'Fixed p_DA constraint_{t}_{k}')
                for t in self.data.TIME
            }

            self.constraints_sub.soc_init_constraint = model.addLConstr(
                self.data.soc_init + self.variables_sub.p_cha[1] * self.data.rho_charge - self.variables_sub.p_dis[1] * (1 / self.data.rho_discharge),
                                                                        GRB.EQUAL,
                                                                        self.variables_sub.soc[1],
                                                                        name='SOC initial constraint')

            self.constraints_sub.soc_time_constraint = {
                t: model.addLConstr(
                    self.variables_sub.soc[t - 1] + self.variables_sub.p_cha[t] * self.data.rho_charge - self.variables_sub.p_dis[t] * (
                                1 / self.data.rho_discharge),
                    GRB.EQUAL,
                    self.variables_sub.soc[t],
                    name=f'SOC time constraint_h_{t}_{k}'
                )
                for t in self.data.TIME
                if t > 1
            }

    def _build_objective_function(self, model, k):
        if model.ModelName == "Main-Problem":
            DA_Rev = gp.quicksum(
                (self.data.da_price[t-1] - self.data.generator_capacity) * self.variables_Main.generator_production[t] + self.variables_Main.gamma
                for t in self.data.TIME
            )
            self.main_model.setObjective(DA_Rev, GRB.MAXIMIZE)

        if model.ModelName == "Sub-Problem":
            subObjectivefunction = gp.quicksum(
                self.data.pi * (self.data.b_price[(t, k)] - self.data.generator_cost) * self.variables_sub.p_B[t]
                for t in self.data.TIME
            )
            model.setObjective(subObjectivefunction, GRB.MAXIMIZE)

    #Is used per scenario==iteration for creating the values for a bender's cut
    def _SubproblemOptimization(self, p_DA_fixed, k):
        optimalBalance = 0
        dual = 0
        self.submodel = gp.Model(name='Sub-Problem')
        self._build_variables(self.submodel)
        self._build_constraints(self.submodel, k, p_DA_fixed, 0)
        self._build_objective_function(self.submodel, k)
        self.submodel.optimize()

        if self.submodel.status == GRB.OPTIMAL:
            submodel_solution_p_B = np.zeros(len(self.data.TIME))
            submodel_solution_DA_constraint = np.zeros(len(self.data.TIME))
            submodel_soc = np.zeros(len(self.data.TIME))
            submodel_p_Cha = np.zeros(len(self.data.TIME))
            submodel_p_dis = np.zeros(len(self.data.TIME))
            for t in range(len(self.data.TIME)):
                submodel_solution_p_B[t] = self.variables_sub.p_B[t+1].x
                submodel_solution_DA_constraint[t] = self.constraints_sub.DA_constraint[t+1].pi
                submodel_soc[t] = self.variables_sub.soc[t+1].x
                submodel_p_Cha[t] = self.variables_sub.p_cha[t+1].x
                submodel_p_dis[t] = self.variables_sub.p_dis[t+1].x



            return submodel_solution_p_B, submodel_solution_DA_constraint
        else:
            if self.submodel.Status == GRB.INFEASIBLE:
                self.submodel.computeIIS()

                # Write IIS to a file
                self.submodel.write("model.ilp")

                print("Subproblem - Irreducible Inconsistent Subsystem (IIS):")

                for constr in self.submodel.getConstrs():
                    if constr.IISConstr:
                        print(f"Infeasible constraint: {constr.ConstrName}")

            elif self.submodel.Status == GRB.UNBOUNDED:
                print("Subproblem - Unbounded Subsystem (UNBOUNDED):")

            return False, False

    # start of the decompositioning algorithm, for now approahed for a fixed t
    def _iterate_Bender(self):
        new_gamma = -100
        #initialising Master problem
        self.main_model = gp.Model(name='Main-Problem')
        # p_DA and y_gamma
        self._build_variables(self.main_model)
        #setting the first y_gamma constraint to y_gamma>=-INFINITY and p_DA < Pnorm
        self._build_constraints(self.main_model, 0, 0, new_gamma)

        I = 0
        epsilon = 20
        #first guess for p_DA that will be fixed in the subproblem
        #p_DA_fixed = float(random.randint(0,self.data.generator_capacity))
        p_DA_fixed = {t:20.0 for t in self.data.TIME}
        benderscuts = 0

        #Master problems Objective function max((lambda_DA-c)*p_DA+y_gamma)
        self._build_objective_function(self.main_model, 0)

        #loop that (should) solve the subproblem, alters y_gamma accordingly and adds the benders cuts
        for k in self.data.SCENARIOS:

            #solution of subproblem optimal p_B at given fixed p_DA and the corresponding dual variable of the conflicting constraint
            optimalBalance, dual = self._SubproblemOptimization(p_DA_fixed, k)

            #Checks whether subproblem was Succesfull
            if type(dual) != type(False):
                #each iteration adds a new gamma-constraint (=Bender's cut) based on the subproblem's optimal Balance and the respective dual variable
                self.constraints_Main.gamma_benders = {
                    t: self.main_model.addLConstr(
                        self.data.pi * (self.data.b_price[(t,k)]-self.data.generator_cost) * optimalBalance[t-1] + dual[t-1] * (self.variables_Main.generator_production[t] - p_DA_fixed[t]),
                        GRB.GREATER_EQUAL,
                        self.variables_Main.gamma,
                        name=f'Gamma contraint for benders_{t}_{k}'
                    )
                    for t in self.data.TIME
                }
                print(I)
                print(new_gamma)
                self._build_constraints(self.main_model, 0, 0, new_gamma)

                self.main_model.update()
                self.main_model.optimize()

                if self.main_model.status == GRB.OPTIMAL:
                    new_gamma = self.variables_Main.gamma.x
                    #calculating the lower and upper bound and check for convergence
                    Lower_Bound = self.main_model.objVal
                    optimal_p_Da = {t:self.variables_Main.generator_production[t].x for t in self.data.TIME}
                    Upper_Bound = sum(
                        (self.data.da_price[t-1] - self.data.generator_cost) * optimal_p_Da[t] + self.data.pi * (self.data.b_price[(t,k)]-self.data.generator_cost)*optimalBalance[t-1]
                        for t in self.data.TIME
                    )

                    if abs(Upper_Bound - Lower_Bound) <= epsilon:
                        break
                    else:
                        I = I + 1
                        # self.main_model.reset()
                        p_DA_fixed = optimal_p_Da
                else:
                    if self.main_model.Status == GRB.INFEASIBLE:
                        self.main_model.computeIIS()

                        # Write IIS to a file
                        self.main_model.write("model.ilp")

                        print("Main-Problem - Irreducible Inconsistent Subsystem (IIS):")
                        for constr in self.main_model.getConstrs():
                            if constr.IISConstr:
                                print(f"Infeasible constraint: {constr.ConstrName}")
                    elif self.main_model.Status == GRB.UNBOUNDED:
                        print("Main-Problem - Unbounded Subsystem (UNBOUNDED):")
                    elif self.main_model.Status == GRB.SUBOPTIMAL:
                        print("Found a solution but not the best")
                    elif self.main_model.Status == GRB.INF_OR_UNBD:
                        print("Ich versteh die welt nicht mehr")
                        for constr in self.main_model.getConstrs():
                            if abs(constr.RHS) > 1e6:  # Example threshold for large RHS
                                print(f"Constraint {constr.ConstrName} has a large RHS: {constr.RHS}")


                print("Jössas")
            else:
                print("Subproblem causes the trouble")

        self.main_model.update()


    def run(self):
        self._iterate_Bender()


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
            SCENARIOS=[1, 2, 3, 4, 5],
            #SCENARIOS=[1],
            TIME=[1],
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
        num_wind_scenarios = 2
        num_price_scenarios = 2
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
            rho_charge=0.9,  # TODO what were the rho values before??
            rho_discharge=0.9,
            soc_max=120,
            soc_init=10,
            charging_capacity=100
        )
    print(input_data.da_price)
    model = StochasticOfferingStrategy(input_data)
    model.run()
    #model.display_results()
    #model.plot_results()


    # model_PI = StochasticOfferingStrategy(input_data)
    # model_PI.run()
    # model_PI.display_results()