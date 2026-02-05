## dependency
tensorflow >=2.0.0
tensorlayer >=2.0.0
numpy == 1.24.2

## reward shaping step
step1: The number of instances is initialized based on the arrival rate lambd and server processing capability mu set for each microservice.
step2: A reward is obtained when an instance is successfully deployed at each step, and a penalty is obtained when an instance fails to be deployed.
step3: After all instances are deployed, calculate the delay and the accumulated reward for each step, and then assign the reward to the last step.

## Quantization Experimentation
### Model robustness: 2 tables + 2 figures
Adjust the random seed to 0, 1, or 1037 (lr = 0.005).
1. Change the number of edge nodes to 4, 6, 8, or 10.
2. Change the number of microservice types to 4, 6, 8, or 10.
### Latency Performance 3 fig
1. Number of requests
2. Edge nodes
3. Number of microservice types
### Resource Utilization (load balancing) 3 fig
1. CPU
2. Memory
3. Load
Adjust weight values for quantitative experiments
## Ablation Study 3 * 2 fig
1. Learning rate (lr)
2. Soft update factor mu
3. Exploration factor var *
4. LSTM *
5. Hidden layers *
## Comparison Algorithms
mb_ddpg, rsdql, ffd, GMDA
## Remarks
Both mb_ddpg.py and rsdql.py are implementations based on the core ideas of the respective papers, adapted for use in this model. There are some differences due to the need to fit the model established in this paper.
To verify the effectiveness, the hyperparameters are kept largely consistent.


