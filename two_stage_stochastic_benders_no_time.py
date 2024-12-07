#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:29:44 2023

@author: lesiamitridati
"""

import gurobipy as gb
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import timeit

# Set plot parameters
sb.set_style('ticks')
size_pp = 15
font = {'family': 'times new roman',
        'color': 'black',
        'weight': 'normal',
        'size': size_pp,
        }

num_wind_scenarios = 2
num_price_scenarios = 2
s = num_wind_scenarios * num_price_scenarios
SCENARIOS = [i for i in range(1, s + 1)]

generator_capacity = 48
scenario_B_prices = {1: 88.31, 2: 136.25, 3: 85.61, 4: 137.09, 5: 146.05}
scenario_windProd = {1: 33, 2: 30, 3: 27, 4: 21, 5: 20}
scenario_DA_prices = 51.49  # Single DA price value
pi = 1 / s


class expando(object):
    """
    A small class which can have attributes set
    """
    pass


class benders_subproblem:
    def __init__(self, master, scenario, generator_cost, generator_capacity, generator_availability, p_DA_fixed, b_price, da_price, pi):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.master = master
        self._init_data(scenario, generator_cost, generator_capacity, generator_availability, p_DA_fixed, b_price, da_price, pi)
        self._build_model()

    def _init_data(self, scenario, generator_cost, generator_capacity, generator_availability, p_DA_fixed, b_price, da_price, pi):
        self.data.scenario = scenario
        self.data.generator_cost = generator_cost
        self.data.generator_capacity = generator_capacity
        self.data.generator_availability = generator_availability
        self.data.p_DA_fixed = p_DA_fixed
        self.data.b_price = b_price
        self.data.da_price = da_price
        self.data.pi = pi

    def _build_model(self):
        self.model = gb.Model(name='subproblem')
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        self.variables.p_B = m.addVar(lb=0, ub=self.data.generator_capacity, name='Balancing_power')
        self.variables.p_DA = m.addVar(lb=0, ub=self.data.generator_capacity, name='DA_power')
        m.update()

    def _build_objective(self):
        m = self.model
        subObjectivefunction = self.data.pi * (self.data.b_price - self.data.generator_cost) * self.variables.p_B
        m.setObjective(subObjectivefunction, GRB.MAXIMIZE)
        m.update()

    def _build_constraints(self):
        m = self.model
        self.constraints.balancing_constraint = m.addConstr(
            self.data.generator_availability == self.variables.p_DA + self.variables.p_B,
            name='Balancing_constraint'
        )
        #these constraints can clash together. if p_DA is set to 48, which is allowed considering the generator capacity max
        # and it must be done because of this da_constraint. then the balancing constraint does not hold anymore, since
        # p_da is already higher than the generator availability. therefore, we must add a constraint that p_da is also not more than the generator availability
        self.constraints.DA_constraint = m.addConstr(
            self.variables.p_DA == self.data.p_DA_fixed,
            name='DA_constraint'
        )
        self.constraints.DA_constraint_availability = m.addConstr(
            self.variables.p_DA <= self.data.generator_availability,
            name='DA_constraint_availability'
        )
        m.update()

    def _update_complicating_variables(self):
        self.constraints.DA_constraint.rhs = self.master.variables.generator_production.x
        self.model.update()


class benders_master:
    def __init__(self, epsilon, max_iters):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._init_data(epsilon, max_iters)
        self._build_model()

    def _init_data(self, epsilon, max_iters):
        self.data.epsilon = epsilon
        self.data.max_iters = max_iters
        self.data.iteration = 1
        self.data.upper_bounds = {}
        self.data.lower_bounds = {}
        self.data.da_dual = {}
        self.data.da_fixed_values = {}
        self.data.gamma_values = {}
        self.data.subproblem_objectives = {}
        self.data.master_objectives = {}

    def _build_model(self):
        self.model = gb.Model(name='master')
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        m = self.model
        self.variables.generator_production = m.addVar(lb=0, ub=generator_capacity, name='DA_production')
        self.variables.gamma = m.addVar(lb=-1000, ub=10000, name='gamma')
        m.update()

    def _build_objective(self):
        m = self.model
        master_objective = (scenario_DA_prices - generator_capacity) * self.variables.generator_production + self.variables.gamma
        m.setObjective(master_objective, GRB.MAXIMIZE)
        m.update()

    def _build_constraints(self):
        self.constraints.master_cuts = {}
        self.constraints.generator_availability_max = {
            s: self.model.addLConstr(
                self.variables.generator_production,
                GRB.LESS_EQUAL,
                scenario_windProd[s],
                name=f'Generator_production_availability_max_{s}'
            )
            for s in SCENARIOS
        }
        self.model.update()

    def _build_subproblems(self):
        self.subproblem = {
            s: benders_subproblem(
                self,
                scenario=s,
                generator_cost=20,
                generator_capacity=generator_capacity,
                generator_availability=scenario_windProd[s],
                p_DA_fixed=self.variables.generator_production.x,
                b_price=scenario_B_prices[s],
                da_price=scenario_DA_prices,
                pi=pi
            )
            for s in SCENARIOS
        }

    def _update_master_cut(self):
        self.constraints.master_cuts[self.data.iteration] = self.model.addConstr(
            self.variables.gamma >= sum(
                pi * (
                    self.data.subproblem_objectives[self.data.iteration - 1][s] +
                    self.data.da_dual[self.data.iteration - 1][s] *
                    (self.variables.generator_production - self.data.da_fixed_values[self.data.iteration - 1])
                )
                for s in SCENARIOS
            ),
            name=f'cut_{self.data.iteration}'
        )
        self.model.update()

    def _save_master_data(self):
        if self.model.Status != GRB.OPTIMAL:
            raise RuntimeError("Master problem has not been solved optimally.")
        self.data.da_fixed_values[self.data.iteration] = self.variables.generator_production.x
        self.data.gamma_values[self.data.iteration] = self.variables.gamma.x
        self.data.lower_bounds[self.data.iteration] = self.model.ObjVal
        self.data.master_objectives[self.data.iteration] = self.model.ObjVal - self.variables.gamma.x
        self.model.update()

    def _save_subproblems_data(self):
        if self.subproblem[s].model.Status == GRB.INFEASIBLE:
            print(f"Subproblem {s} status: {self.subproblem[s].model.Status}")
            self.subproblem[s].model.computeIIS()
            self.subproblem[s].model.write("infeasible_model.ilp")
        self.data.da_dual[self.data.iteration] = {
            s: self.subproblem[s].constraints.DA_constraint.Pi for s in SCENARIOS
        }
        self.data.subproblem_objectives[self.data.iteration] = {
            s: self.subproblem[s].model.ObjVal for s in SCENARIOS
        }
        self.data.upper_bounds[self.data.iteration] = self.data.master_objectives[self.data.iteration] + sum(
            pi * self.subproblem[s].model.ObjVal for s in SCENARIOS
        )

    def _do_benders_step(self):
        self.data.iteration += 1
        self._update_master_cut()
        self.model.optimize()
        self._save_master_data()
        for s in SCENARIOS:
            self.subproblem[s]._update_complicating_variables()
            self.subproblem[s].model.optimize()
            if self.subproblem[s].model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Subproblem {s} is not solved optimally.")
        self._save_subproblems_data()

    def _benders_iterate(self):
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


# Solve and print results
start = timeit.timeit()

DA_model = benders_master(epsilon=0.1, max_iters=100)
DA_model._benders_iterate()

end = timeit.timeit()

print('Solving time:', end - start)
print('Optimal cost:', DA_model.data.upper_bounds[DA_model.data.iteration])
