# Optimization for Modern Power Systems Project - Group 12
## Data preparation
Write a bit about scenario generation

input data used everywhere: 

20 wind and 20 price scenarios

input_data = InputData(

            SCENARIOS=Scenarios,
            TIME=Time,
            generator_cost=20,
            generator_capacity=48,
            generator_availability=scenario_windProd,
            da_price=scenario_DA_prices,
            b_price=scenario_B_prices,
            pi = pi,
            rho_charge=0.8332, 
            rho_discharge=0.8332,
            soc_max=120,
            soc_init=10,
            charging_capacity=100
        )

## Two-Stage Stochastic Optimization
Write about the general two-stage model

### Chance Constrained

### Risk Aware

### EVPI

### Benders Decomposition

## Iterative Stochastic Problem

