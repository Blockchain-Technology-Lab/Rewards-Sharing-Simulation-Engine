# Metrics

There are several metrics, or model reporters, that can be tracked during a simulation. Each of them is associated with 
an id for convenience. We provide details for all of them below:

1. **Pool count**: the number of active pools in the system.
2. **Total pledge**: the total pledged stake of the system (sum of all pools' pledges).
3. **Mean pledge**: the average value of stake that is pledged in pools.
4. **Median pledge**: the median value of stake that is pledged in pools.
5. **Average pools per operator**: the average number of pools that an operator controls.
6. **Max pools per operator**: the maximum number of pools that an operator controls.
7. **Median pools per operator**: the median number of pools that an operator controls.
8. **Average saturation rate**: the average saturation rate (stake / saturation threshold) across all active pools.
9. **Nakamoto coefficient**: the minimum number of entities that collectively control more than 50% of the system's 
     active stake through their pools.
10. **Statistical distance**: the [statistical distance](https://en.wikipedia.org/wiki/Statistical_distance) of the 
    distributions of the stake that agents controlled at the beginning of the simulation vs on this round.
11. **Min-aggregate pledge**: the minimum aggregate pledge of pools that collectively control more than 50% of the 
    system's active stake. Note that the calculation of this metric is slow because of the complexity of the problem.
12. **Pledge rate**: the fraction of active stake that is used as pledge (total pledge / total active stake).
13. **Pool homogeneity factor**: a metric that describes how homogeneous the pools of the system are (the highest 
    possible value is 1, which is given when all pools have the same size).
14. **Iterations**: the number of iterations that the simulation has gone through.
15. **Mean stake rank**: the average rank of pool operators regarding their initial stake.
16. **Mean cost rank**: the average rank of pool operators regarding their initial cost.
17. **Median stake rank**: the median rank of pool operators regarding their initial stake.
18. **Median cost rank**: the median rank of pool operators regarding their initial cost.
19. **Number of pool splitters**: the number of stakeholders that operate two or more pools.
20. **Cost efficient stakeholders**: the number of agents for whom it is possible to make profit by operating a pool.
21. **StakePairs**: a mapping of each pool id to its stake and profit margin.
22. **Gini-id**: a variation of the gini coefficient, where we consider each agent as an “id” and each pool as a “coin”. 
    Then the gini-id is the gini coefficient considering each party with the coins they have.  In case of each agent 
    operating one pool this coefficient is 0.
23. **Gini-id stake**: like the gini-id above, but considering the total stake each agent controls through their pools 
    instead of the number of pools they operate.
24. **Mean margin**: the average profit margin across all active pools.
25. **Median margin**: the median profit margin across all active pools.
26. **Stake per agent**: a list where each element corresponds to the total stake controlled through an agent's pool for 
    some agent
27. **Stake per agent id**: a mapping of each agent id to the total stake that said agent controls through their pools.
28. **Total delegated stake**: the total stake delegated to active pools (including pledged stake).
29. **Total agent stake**: the total stake held by agents.
30. **Operator count**: the number of stakeholders that operate pools.

Refer to the [Configuration](configuration.md) page for details on specifying which metrics will be used during a 
simulation.