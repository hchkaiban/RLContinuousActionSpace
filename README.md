# RLContinuousActionSpace
Reinforcement Learning in with continuous states and actions spaces. DDPG: Double Deep Policy Gradient and A3C: Asynchronous Actor-Critic Agents

1° DDPG:
Based on Continuous Control with Deep Reinforcement Learning: https://arxiv.org/pdf/1509.02971.pdf and 
great blog from https://github.com/yanpanlau/DDPG-Keras-Torcs.
This approach combines the advantages of the DDQN algorithm (experience replay and target networks) with the actor-critic structure enabling to output continuous actions.
- OU.py: exploration is done by Ornstein-Uhlenbeck process wich has the convenient mean reverting property. 
- Models.py: Neural networks for actor, critic and target networks

Actor Model    |  Critic Model
:-------------------------:|:-------------------------:
<img src="https://github.com/hchkaiban/RLContinuousActionSpace/blob/master/RL_DDPG/KerasModels/DDPG_Actor_model.png" alt=" " width="300" height="300">  |  <img src="https://github.com/hchkaiban/RLContinuousActionSpace/blob/master/RL_DDPG/KerasModels/DDPG_Critic_model.png" alt=" " width="300" height="300">

- main.py: configure, train, test, display, store, load
- ReplayBuffer.py: Memory buffer for experience replay of stored s,a,r,s' tuples
- TempConfig.py: utils

- Env.py: The algorithm was first validated on the pendulum-v0 game of gym open-ai and then applied on a customized Environement EnvPlant.py, simulating a temperature model.
