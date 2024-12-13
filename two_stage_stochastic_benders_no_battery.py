import gurobipy as gb
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from scenario_generation_function import generate_scenarios

# Set global variables
num_wind_scenarios = 20
num_price_scenarios = 20
s = num_wind_scenarios * num_price_scenarios
SCENARIOS = [i for i in range(1, s + 1)]
TIME = [i for i in range(1, 25)]

scenario_DA_prices = {}
scenario_B_prices = {}
scenario_windProd = {}
generator_capacity = 48
scenario_DA_prices, scenario_B_prices, scenario_windProd = generate_scenarios(num_price_scenarios, num_wind_scenarios)
scenario_DA_prices = [
    51.49, 48.39, 48.92, 49.45, 42.72, 50.84, 82.15, 100.96, 116.60,
    112.20, 108.54, 111.61, 114.02, 127.40, 134.02, 142.18, 147.42,
    155.91, 154.10, 148.30, 138.59, 129.44, 122.89, 112.47
]

# Equal probability of each scenario 1/100
pi = 1 / s


class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class benders_subproblem:  # Class representing the subproblems for each scenario

    def __init__(self, master, scenario, generator_cost, generator_capacity, generator_availability, p_DA_fixed, b_price, da_price, pi):  # initialize class
        self.data = expando()  # define data attributes
        self.variables = expando()  # define variable attributes
        self.constraints = expando()  # define constraints attributes
        self.master = master  # define master problem to which subproblem is attached
        self._init_data(scenario, generator_cost, generator_capacity, generator_availability, p_DA_fixed, b_price, da_price, pi)  # initialize data
        self._build_model()  # build gurobi model

    def _init_data(self, scenario, generator_cost, generator_capacity, generator_availability, p_DA_fixed, b_price, da_price, pi):  # initialize data
        self.data.scenario = scenario  # add scenario #
        self.data.generator_cost = generator_cost,
        self.data.generator_capacity = generator_capacity,
        self.data.generator_availability = generator_availability,
        self.data.p_DA_fixed = p_DA_fixed,
        self.data.b_price = b_price,
        self.data.da_price = da_price,
        self.data.pi = pi

    def _build_model(self):  # build gurobi model

        self.model = gb.Model(name='subproblem')  # build model
        self._build_variables()  # add variables
        self._build_objective()  # add objective
        self._build_constraints()  # add constraints
        self.model.update()

    def _build_variables(self):  # build variables

        # index shortcut
        m = self.model

        # p_B
        self.variables.p_B = {
            (t, s): m.addVar(
                lb=0, ub=generator_capacity, name=f'Balancing power_h_{t}_{s}'
            )
            for t in TIME
            for s in SCENARIOS
        }

        # p_DA
        self.variables.p_DA = {
            t: m.addVar(
                lb=0, ub=generator_capacity, name=f'DA power_h_{t}'
            )
            for t in TIME
        }

        m.update()  # update model

    def _build_objective(self):  # define objective function

        # index shortcut
        m = self.model
        k = self.data.scenario

        subObjectivefunction = gb.quicksum(
            self.data.pi * (self.data.b_price[0][(t, k)] - self.data.generator_cost[0]) * self.variables.p_B[(t, k)]
            for t in TIME
        )

        m.setObjective(subObjectivefunction, GRB.MAXIMIZE)

        m.update()

    def _build_constraints(self):

        # index shortcut
        m = self.model
        k = self.data.scenario

        self.constraints.balancing_constraint = {
            t: m.addLConstr(
                self.data.generator_availability[0][(t, k)],
                GRB.EQUAL,
                self.variables.p_DA[t] + self.variables.p_B[(t, k)],
                name=f'Balancing power constraint_{t}_{k}'
            )
            for t in TIME
        }

        self.constraints.DA_constraint = {
            t: m.addLConstr(
                self.variables.p_DA[t],
                GRB.EQUAL,
                self.data.p_DA_fixed[0][t],
                name=f'Fixed p_DA constraint_{t}'
            )
            for t in TIME
        }

        self.constraints.DA_constraint_availability = {
            t: m.addLConstr(
                self.variables.p_DA[t],
                GRB.LESS_EQUAL,
                self.data.generator_availability[0][(t, k)],
                name=f'p_DA constraint_availability_max{t}_{k}'
            )
            for t in TIME
        }

        self.constraints.B_constraint_availability = {
            t: m.addLConstr(
                self.variables.p_B[(t, k)],
                GRB.LESS_EQUAL,
                self.data.generator_availability[0][(t, k)],
                name=f'p_B constraint_availability_max{t}_{k}'
            )
            for t in TIME
        }

        m.update()

    def _update_complicating_variables(self):  # function that updates the value of the complicating variables in the right-hand-side

        # index shortcut
        m = self.model

        for t in TIME:
            self.constraints.DA_constraint[t].rhs = self.master.variables.generator_production[t].x

        m.update()


#  define class of master problem taking as inputs epsilon: convergence criteria parameter, and max_iters: maximum number of terations

class benders_master:  # class of master problem

    def __init__(self, epsilon, max_iters):  # initialize class
        self.data = expando()  # build data attributes
        self.variables = expando()  # build variable attributes
        self.constraints = expando()  # build sontraint attributes
        self._init_data(epsilon, max_iters)  # initialize data
        self._build_model()  # build gurobi model

    def _init_data(self, epsilon, max_iters):  # initialize data
        self.data.epsilon = epsilon  # add value of convergence criteria
        self.data.max_iters = max_iters  # add max number of iterations
        self.data.iteration = 1  # initialize value of iteration count
        self.data.upper_bounds = {}  # initialize list of upper-bound values
        self.data.lower_bounds = {}  # initialize list of lower-bound values
        self.data.da_dual = {}  # initialize list of sensitivities values
        self.data.da_fixed_values = {}  # initialize list of complicating variables values
        self.data.gamma_values = {}  # initialize list of gamma values
        self.data.subproblem_objectives = {}  # initialize list of subproblems objective values
        self.data.master_objectives = {}  # initialize list of master problem objective values
        self.data.da_results = {}  # initialize list of da power bid results
        self.data.balancing_results = {}  # initialize list of balancing power bid results

    def _build_model(self):  # build gurobi model
        self.model = gb.Model(name='master')  # build model
        self._build_variables()  # add variables
        self._build_objective()  # add objective
        self._build_constraints()  # add constraints
        self.model.update()

    def _build_variables(self):  # build variables

        # index shortcut
        m = self.model

        self.variables.generator_production = {
            t: m.addVar(
                lb=0, ub=generator_capacity, name=f'DA production_h_{t}'
            )
            for t in TIME
        }

        self.variables.gamma = m.addVar(lb=-1000, ub=1000, name="gamma")

        m.update()

    def _build_objective(self):  # build objective

        # index shortcut
        m = self.model

        # Set the objective function for the master problem
        master_objective = gb.quicksum(
            (scenario_DA_prices[t-1] - generator_capacity) * self.variables.generator_production[t]
            + self.variables.gamma
            for t in TIME
        )
        m.setObjective(master_objective, GRB.MAXIMIZE)

        m.update()

    def _build_constraints(self):  # build constraints

        # index shortcut
        m = self.model

        self.constraints.generator_availability_max = {
            (t, s): m.addLConstr(
                self.variables.generator_production[t],
                GRB.LESS_EQUAL,
                scenario_windProd[(t, s)],
                name=f'Generator_production_availability_max_{t}_{s}'
            )
            for t in TIME
            for s in SCENARIOS
        }

        # initialize master problem cuts (empty)
        self.constraints.master_cuts = {}

        m.update()

    def _build_subproblems(self):  # function that builds subproblems

        self.subproblem = {s: benders_subproblem(
            self, scenario=s,
            generator_cost=20,
            generator_capacity=generator_capacity,
            generator_availability=scenario_windProd,
            p_DA_fixed={t: self.variables.generator_production[t].x for t in TIME},
            b_price=scenario_B_prices,
            da_price=scenario_DA_prices,
            pi=pi)
            for s in SCENARIOS}

    def _update_master_cut(self):  # fucntion that adds cuts to master problem
        # index shortcut
        m = self.model

        self.constraints.master_cuts[self.data.iteration] = m.addLConstr(
            self.variables.gamma,
            gb.GRB.LESS_EQUAL,
            gb.quicksum(pi * (
                        self.data.subproblem_objectives[self.data.iteration - 1][s] + gb.quicksum(
                    self.data.da_dual[self.data.iteration - 1][t, s] * (
                                self.variables.generator_production[t] -
                                self.data.da_fixed_values[self.data.iteration - 1][t]) for t in
                    TIME)) for s in SCENARIOS),
            name='new cut at iteration {0}'.format(self.data.iteration))

        m.update()

    def _save_master_data(self):  # function that saves results of master problem optimization at each iteration (complicating variables, objective value, lower bound value)

        # index shortcut
        m = self.model
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError("Master problem has not been solved optimally. Cannot retrieve variable values.")

        # save complicating variables value
        self.data.da_fixed_values[self.data.iteration] = {t: self.variables.generator_production[t].x for t in TIME}

        # save gamma value
        self.data.gamma_values[self.data.iteration] = self.variables.gamma.x

            # save lower bound value
        self.data.lower_bounds[self.data.iteration] = m.ObjVal

        # save master problem objective value
        self.data.master_objectives[self.data.iteration] = m.ObjVal - self.variables.gamma.x

        m.update()

    def _save_subproblems_data(self):  # function that saves results of subproblems optimization at each iteration (sensitivities, objective value, upper bound value)

        # index shortcut
        m = self.model
        if self.subproblem[s].model.Status == GRB.INFEASIBLE:
            print(f"Subproblem {s} status: {self.subproblem[s].model.Status}")
            self.subproblem[s].model.computeIIS()
            self.subproblem[s].model.write("infeasible_model.ilp")

        # save sensitivities (for each complicating variables in each subproblem)
        self.data.da_dual[self.data.iteration] = {
            (t, s): self.subproblem[s].constraints.DA_constraint[t].Pi for t in TIME for s in SCENARIOS}

        self.data.da_results[self.data.iteration] = {
            (t, s): self.subproblem[s].variables.p_DA[t].x for t in TIME for s in SCENARIOS}

        self.data.balancing_results[self.data.iteration] = {
            (t, s): self.subproblem[s].variables.p_B[(t, s)].x for t in TIME for s in SCENARIOS}

        # save subproblems objective values
        self.data.subproblem_objectives[self.data.iteration] = {s: self.subproblem[s].model.ObjVal for s in SCENARIOS}

        # save upper bound value
        self.data.upper_bounds[self.data.iteration] = self.data.master_objectives[self.data.iteration] + sum(
            pi * self.subproblem[s].model.ObjVal for s in SCENARIOS)

        m.update()

    def _do_benders_step(self):  # function that does one benders step

        # index shortcut
        m = self.model

        self.data.iteration += 1  # go to next iteration
        self._update_master_cut()  # add cut
        m.optimize()  # optimize master problem
        self._save_master_data()  # save master problem optimization results
        for s in SCENARIOS:
            self.subproblem[s]._update_complicating_variables()  # update value of complicating constraints in subproblems
            self.subproblem[s].model.optimize()  # solve subproblems
        self._save_subproblems_data()  # save subproblems optimization results

    def _benders_iterate(self):  # function that solves iteratively the benders algorithm

        # index shortcut
        m = self.model

        # initial iteration:
        m.optimize()  # solve master problem (1st iteration)
        self._save_master_data()  # save results of master problem and lower bound
        self._build_subproblems()  # build subproblems (1st iteration)
        for s in SCENARIOS:
            self.subproblem[s].model.optimize()  # solve subproblems
        self._save_subproblems_data()  # save results of subproblems and upper bound

        # do benders steps until convergence
        while (
                (abs(self.data.upper_bounds[self.data.iteration] - self.data.lower_bounds[
                    self.data.iteration]) > self.data.epsilon and
                 self.data.iteration < self.data.max_iters)):
            self._do_benders_step()

    def plot_results(self, scenario_num, iteration):
        """
        Plots results of DA bid and balancing power.
        """
        # inspected scenario
        scenario = scenario_num
        time = TIME
        scenarios = SCENARIOS

        # DA STAGE
        da_bid = {t: self.data.da_results[iteration][(t, k)] for t in time for k in scenarios}
        da_price = {t: scenario_DA_prices[t - 1] for t in time}

        # Extract balancing bid (averaged over scenarios)
        balancing_bid = {(t, k): self.data.balancing_results[iteration][(t, k)] for t in time for k in scenarios}
        balancing_price = {(t, k): scenario_B_prices[(t, k)] for t in time for k in scenarios}

        fig, axs1 = plt.subplots(figsize=(9, 6))
        axs1.plot(time, [balancing_bid[t, scenario] for t in time], label=f"Balancing Bid for S{scenario}",
                    color="green", marker="x")
        axs1.plot(time, [da_bid[t] for t in time], label="DA Bid", color="green", linestyle="--")
        axs1.set_xlabel("Time (t)")
        axs1.set_xticks(time)
        axs1.set_ylabel("Power (MW)")
        axs1.set_title(f"Power  and Prices for S{scenario}")
        axs1.grid(True)
        axs1.legend()

        # Create a secondary y-axis for prices
        ax2 = plt.twinx()
        ax2.plot(time, [balancing_price[t, scenario] for t in time], label=f"Balancing Price S{scenario}", color="red",
                 alpha=0.5)
        ax2.plot(time, [da_price[t] for t in time], label="DA Price", color="red", linestyle="--", alpha=0.5)

        ax2.set_ylabel("Price ($)")


        lines1, labels1 = axs1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs1.legend(lines1, labels1, loc='upper left')
        ax2.legend(lines2, labels2, loc='upper right')
        #plt.savefig(f'figure/s1_scenario{scenario}.jpg',bbox_inches='tight',dpi=300)
        #plt.savefig(f's1_scenario{scenario}.jpg', bbox_inches='tight', dpi=300)
        plt.tight_layout()
        plt.show()

# solve and print results
DA_model = benders_master(epsilon=0.1, max_iters=1000)
DA_model._benders_iterate()
DA_model.plot_results(13, DA_model.data.iteration)
DA_model.plot_results(19, DA_model.data.iteration)
print('Optimal cost ', DA_model.data.upper_bounds[DA_model.data.iteration])  # print optimal cost (last upper-bound)