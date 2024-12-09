\ Model Two-stage stochastic offering strategy
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  33.194 DA_production_h_1 + 77.55 DA_production_h_2
   + 104.118 DA_production_h_3 + 134.582 DA_production_h_4
   + 108.172 DA_production_h_5
Subject To
 DA_power_constraint_1_1: DA_production_h_1 - Balancing_discharge_1
   + Balancing_charge_1 <= 33
 DA_power_constraint_1_2: DA_production_h_1 - Balancing_discharge_1
   + Balancing_charge_1 <= 30
 DA_power_constraint_1_3: DA_production_h_1 - Balancing_discharge_1
   + Balancing_charge_1 <= 27
 DA_power_constraint_1_4: DA_production_h_1 - Balancing_discharge_1
   + Balancing_charge_1 <= 21
 DA_power_constraint_1_5: DA_production_h_1 - Balancing_discharge_1
   + Balancing_charge_1 <= 20
 DA_power_constraint_2_1: DA_production_h_2 - Balancing_discharge_2
   + Balancing_charge_2 <= 20
 DA_power_constraint_2_2: DA_production_h_2 - Balancing_discharge_2
   + Balancing_charge_2 <= 21
 DA_power_constraint_2_3: DA_production_h_2 - Balancing_discharge_2
   + Balancing_charge_2 <= 31
 DA_power_constraint_2_4: DA_production_h_2 - Balancing_discharge_2
   + Balancing_charge_2 <= 17
 DA_power_constraint_2_5: DA_production_h_2 - Balancing_discharge_2
   + Balancing_charge_2 <= 10
 DA_power_constraint_3_1: DA_production_h_3 - Balancing_discharge_3
   + Balancing_charge_3 <= 33
 DA_power_constraint_3_2: DA_production_h_3 - Balancing_discharge_3
   + Balancing_charge_3 <= 27
 DA_power_constraint_3_3: DA_production_h_3 - Balancing_discharge_3
   + Balancing_charge_3 <= 50
 DA_power_constraint_3_4: DA_production_h_3 - Balancing_discharge_3
   + Balancing_charge_3 <= 19
 DA_power_constraint_3_5: DA_production_h_3 - Balancing_discharge_3
   + Balancing_charge_3 <= 80
 DA_power_constraint_4_1: DA_production_h_4 - Balancing_discharge_4
   + Balancing_charge_4 <= 21
 DA_power_constraint_4_2: DA_production_h_4 - Balancing_discharge_4
   + Balancing_charge_4 <= 36
 DA_power_constraint_4_3: DA_production_h_4 - Balancing_discharge_4
   + Balancing_charge_4 <= 50
 DA_power_constraint_4_4: DA_production_h_4 - Balancing_discharge_4
   + Balancing_charge_4 <= 50
 DA_power_constraint_4_5: DA_production_h_4 - Balancing_discharge_4
   + Balancing_charge_4 <= 50
 DA_power_constraint_5_1: DA_production_h_5 - Balancing_discharge_5
   + Balancing_charge_5 <= 25
 DA_power_constraint_5_2: DA_production_h_5 - Balancing_discharge_5
   + Balancing_charge_5 <= 39
 DA_power_constraint_5_3: DA_production_h_5 - Balancing_discharge_5
   + Balancing_charge_5 <= 50
 DA_power_constraint_5_4: DA_production_h_5 - Balancing_discharge_5
   + Balancing_charge_5 <= 50
 DA_power_constraint_5_5: DA_production_h_5 - Balancing_discharge_5
   + Balancing_charge_5 <= 50
 Max_DA_production_constraint_1_1: DA_production_h_1 + Balancing_charge_1
   <= 33
 Max_DA_production_constraint_1_2: DA_production_h_1 + Balancing_charge_1
   <= 30
 Max_DA_production_constraint_1_3: DA_production_h_1 + Balancing_charge_1
   <= 27
 Max_DA_production_constraint_1_4: DA_production_h_1 + Balancing_charge_1
   <= 21
 Max_DA_production_constraint_1_5: DA_production_h_1 + Balancing_charge_1
   <= 20
 Max_DA_production_constraint_2_1: DA_production_h_2 + Balancing_charge_2
   <= 20
 Max_DA_production_constraint_2_2: DA_production_h_2 + Balancing_charge_2
   <= 21
 Max_DA_production_constraint_2_3: DA_production_h_2 + Balancing_charge_2
   <= 31
 Max_DA_production_constraint_2_4: DA_production_h_2 + Balancing_charge_2
   <= 17
 Max_DA_production_constraint_2_5: DA_production_h_2 + Balancing_charge_2
   <= 10
 Max_DA_production_constraint_3_1: DA_production_h_3 + Balancing_charge_3
   <= 33
 Max_DA_production_constraint_3_2: DA_production_h_3 + Balancing_charge_3
   <= 27
 Max_DA_production_constraint_3_3: DA_production_h_3 + Balancing_charge_3
   <= 50
 Max_DA_production_constraint_3_4: DA_production_h_3 + Balancing_charge_3
   <= 19
 Max_DA_production_constraint_3_5: DA_production_h_3 + Balancing_charge_3
   <= 80
 Max_DA_production_constraint_4_1: DA_production_h_4 + Balancing_charge_4
   <= 21
 Max_DA_production_constraint_4_2: DA_production_h_4 + Balancing_charge_4
   <= 36
 Max_DA_production_constraint_4_3: DA_production_h_4 + Balancing_charge_4
   <= 50
 Max_DA_production_constraint_4_4: DA_production_h_4 + Balancing_charge_4
   <= 50
 Max_DA_production_constraint_4_5: DA_production_h_4 + Balancing_charge_4
   <= 50
 Max_DA_production_constraint_5_1: DA_production_h_5 + Balancing_charge_5
   <= 25
 Max_DA_production_constraint_5_2: DA_production_h_5 + Balancing_charge_5
   <= 39
 Max_DA_production_constraint_5_3: DA_production_h_5 + Balancing_charge_5
   <= 50
 Max_DA_production_constraint_5_4: DA_production_h_5 + Balancing_charge_5
   <= 50
 Max_DA_production_constraint_5_5: DA_production_h_5 + Balancing_charge_5
   <= 50
 Max_production_constraint_1_1: Balancing_charge_1 <= 33
 Max_production_constraint_1_2: Balancing_charge_1 <= 30
 Max_production_constraint_1_3: Balancing_charge_1 <= 27
 Max_production_constraint_1_4: Balancing_charge_1 <= 21
 Max_production_constraint_1_5: Balancing_charge_1 <= 20
 Max_production_constraint_2_1: Balancing_charge_2 <= 20
 Max_production_constraint_2_2: Balancing_charge_2 <= 21
 Max_production_constraint_2_3: Balancing_charge_2 <= 31
 Max_production_constraint_2_4: Balancing_charge_2 <= 17
 Max_production_constraint_2_5: Balancing_charge_2 <= 10
 Max_production_constraint_3_1: Balancing_charge_3 <= 33
 Max_production_constraint_3_2: Balancing_charge_3 <= 27
 Max_production_constraint_3_3: Balancing_charge_3 <= 50
 Max_production_constraint_3_4: Balancing_charge_3 <= 19
 Max_production_constraint_3_5: Balancing_charge_3 <= 80
 Max_production_constraint_4_1: Balancing_charge_4 <= 21
 Max_production_constraint_4_2: Balancing_charge_4 <= 36
 Max_production_constraint_4_3: Balancing_charge_4 <= 50
 Max_production_constraint_4_4: Balancing_charge_4 <= 50
 Max_production_constraint_4_5: Balancing_charge_4 <= 50
 Max_production_constraint_5_1: Balancing_charge_5 <= 25
 Max_production_constraint_5_2: Balancing_charge_5 <= 39
 Max_production_constraint_5_3: Balancing_charge_5 <= 50
 Max_production_constraint_5_4: Balancing_charge_5 <= 50
 Max_production_constraint_5_5: Balancing_charge_5 <= 50
 Max_state_of_charge_constraint_1: State_of_charge_1 <= 120
 Max_state_of_charge_constraint_2: State_of_charge_2 <= 120
 Max_state_of_charge_constraint_3: State_of_charge_3 <= 120
 Max_state_of_charge_constraint_4: State_of_charge_4 <= 120
 Max_state_of_charge_constraint_5: State_of_charge_5 <= 120
 SOC_time_constraint_2: State_of_charge_1 - State_of_charge_2
   - 1.111111111111111 Balancing_discharge_2 + 0.9 Balancing_charge_2 = 0
 SOC_time_constraint_3: State_of_charge_2 - State_of_charge_3
   - 1.111111111111111 Balancing_discharge_3 + 0.9 Balancing_charge_3 = 0
 SOC_time_constraint_4: State_of_charge_3 - State_of_charge_4
   - 1.111111111111111 Balancing_discharge_4 + 0.9 Balancing_charge_4 = 0
 SOC_time_constraint_5: State_of_charge_4 - State_of_charge_5
   - 1.111111111111111 Balancing_discharge_5 + 0.9 Balancing_charge_5 = 0
 SOC_initial_constraint1: - State_of_charge_1
   - 1.111111111111111 Balancing_discharge_1 + 0.9 Balancing_charge_1
   = -10
 Discharge_initial_constraint1: Balancing_discharge_1 = 0
Bounds
 DA_production_h_1 <= 48
 DA_production_h_2 <= 48
 DA_production_h_3 <= 48
 DA_production_h_4 <= 48
 DA_production_h_5 <= 48
 Balancing_discharge_1 <= 40
 Balancing_discharge_2 <= 40
 Balancing_discharge_3 <= 40
 Balancing_discharge_4 <= 40
 Balancing_discharge_5 <= 40
 Balancing_charge_1 <= 40
 Balancing_charge_2 <= 40
 Balancing_charge_3 <= 40
 Balancing_charge_4 <= 40
 Balancing_charge_5 <= 40
End
