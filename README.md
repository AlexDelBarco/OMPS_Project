# Optimization for Modern Power Systems Project - Group 12
The file ***scenario_generation_function.py*** contains the code that prepared the data and generated the scenarios. 
Since the same seed was used for the generation of data, the different classes that use this code will all use the same data, which will make it possible to compare results accurately later.

As input, the same values are used for all different models. The input can be seen below as an illustration.
However, as mentioned in the report, for the chance constrained problem, 4 wind and price scenarios were used instead of 20 as in the other problems.

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

***two_stage_stochastic.py*** contains the code for the baseline two-stage stochastic optimization problem.  

***two_stage_stochastic_EVPI.py*** contains the code for the two-stage problem but including the Expected Value of Perfect Information  

***two_stage_stochastic_chance_constrained.py*** contains the code for the chance-constrained problem. 

***two_stage_stochastic_riskaware.py*** contains the code for the risk-aware version of the two-stage stochastic problem.  

***two_stage_stochastic_benders_no_battery.py*** contains the code for the benders decomposition, as mentioned, without the use of a battery.  

***iterative_stochastic.py*** contains the code for the iterative approach of the problem.


