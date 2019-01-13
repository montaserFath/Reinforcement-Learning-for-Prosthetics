# NeurIPS2018-Challenge-RL-for-Prosthetics

In this repo, we are trying to apply **Reinforcement Learning (RL)** to enable prosthetics to calibrate with **differences between humans and differences between walking environments**. using [OpenSim](https://opensim.stanford.edu/) to simulate prosthetic.

This project is apart from [NeurIPS 2018: AI for Prosthetics Challenge](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge)

[![AI for prosthetics](https://s3-eu-west-1.amazonaws.com/kidzinski/nips-challenge/images/ai-prosthetics.jpg)](https://github.com/stanfordnmbl/osim-rl)
# Publications and Awards

- Best Graduation project at [University of khartoum](uofk.edu). Khartoum, Sudan

- Poster at NeurIPS 2018 [Black in AI workshop](https://blackinai.github.io/workshop/2018/programs/), Montreal, Canada

- Poster at [Deep Learning Indaba](http://www.deeplearningindaba.com) 2018, South Africa
## Objectives
- **Benchmarking RL algorithms:** Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971), Trust Region Policy Optimization [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) and Proximal Policy Optimization [PPO](https://arxiv.org/abs/1707.06347) algorithms.

- **Reduce training time** using Imitation Learning algorithm Dataset Aggregation algorithm [DAgger](http://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

- **Modificat DAgger algorithm** to balance between exploration and exploiting.

## OpenSim Enviroment
[OpenSim](https://opensim.stanford.edu/) models one human leg and prosthetic in another leg.

### Observations
the [observations](http://osim-rl.stanford.edu/docs/nips2018/observation/) can be divided into five components:

- **Body parts:** the agent observes its position, velocity, acceleration, rotation, rotational velocity, and rotational acceleration.

- **Joints:** the agent observes its position, velocity and acceleration.

- **Muscles:** the agent observes its activation, fiber force, fiber length and fiber velocity.

- **Forces:** describes the forces acting on body parts.

- **Center of mass:** the agent observes the position, velocity, and acceleration.

### Actions

- Muscles activation, lenght and velocity

- Joints angels.

- Tendons.

### Reward

**<img src="https://latex.codecogs.com/gif.latex?R_{t}=9-(3-V_{t})^2" />**


Where the <img src="https://latex.codecogs.com/gif.latex?V_{t}"/> is the horizontal velocity vector of the pelvi which is function of all state variables.

The termination condition for the episode is filling 300 steps or the height of the pelvis falling below 0.6 meters
## Algorithms and Hyperparameters

- **[DDPG](https://arxiv.org/abs/1509.02971)** is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.DDPG is based on the deterministic policy gradient (DPG) algorithm. it combines the actor-critic approach with insights from the recent success of Deep Q Network (DQN).

- **[PPO](https://arxiv.org/abs/1707.06347)** is a policy optimization method that use multiple epochs of stochastic gradient ascent to perform each policy update.

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)** is a model free, on-policy optimization method that effective for optimizing large nonlinear policies such as neural networks.

## Results
![Results](https://github.com/montaserFath/NeurIPS2018-Challenge-RL-for-Prosthetics/blob/master/pro_mean.png)

## Demo
- **Random Actions**

![Random](https://github.com/montaserFath/NeurIPS2018-Challenge-RL-for-Prosthetics/blob/master/Demos/Random_prothetics.gif)

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)**

![TRPO](https://github.com/montaserFath/NeurIPS2018-Challenge-RL-for-Prosthetics/blob/master/Demos/TRPO_prothetics.gif)

- **[PPO](https://arxiv.org/abs/1707.06347)**

![PPO](https://github.com/montaserFath/NeurIPS2018-Challenge-RL-for-Prosthetics/blob/master/Demos/PPO_prothetics.gif)

- **[DDPG](https://arxiv.org/abs/1509.02971)**

![DDPG](https://github.com/montaserFath/NeurIPS2018-Challenge-RL-for-Prosthetics/blob/master/Demos/DDPG_prothetics.gif)
## Discussion

- OpenSim [ProstheticsEnv](http://osim-rl.stanford.edu) is a very **complex environment**, it contains more than 158 continuous state variables and 19 continuous action variables.

- RL algorithms take a **long time** to build a complex policy which has the ability to compute all state variables and select action variables which will maximize the reward.

- **[DDPG](https://arxiv.org/abs/1509.02971) algorithm achieves good** reward because it designed for high dimensions continuous space environments and it uses the replay buffer.

- **[PPO](https://arxiv.org/abs/1707.06347) the least training time** comparing to [DDPG](https://arxiv.org/abs/1509.02971) and [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) because [PPO](https://arxiv.org/abs/1707.06347) uses gradient algorithm approximation instance of the conjugate gradient algorithm.

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) algorithm achieved the maximum Reward** because it takes time to reach the “trusted” region so it slower than [DDPG](https://arxiv.org/abs/1509.02971) and [PPO](https://arxiv.org/abs/1707.06347) .


## Limitations

- The prosthetic model **can not walk for large distances**.

- Each experiment **runs for one time**, So we are planing to Repeat each experiment number of times with different random seeds and take the average and variance.

- We used **same hyperparameters** for all algorithm to make benchmarking between algorithms, we need to select the best hyperparameters for each algorithm and environment.

- We benchmarcked three RL algorithms only and from **one library**([ChainerRL](https://github.com/chainer/chainerrl)). So we are planing to use different implementations.

- We transfer learning between **similar agents**.


## Getting Started

### ChainerRL libary
- [ChainerRL](https://github.com/chainer/chainerrl) is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using Chainer, a flexible deep learning framework.

- [ChainerRL](https://github.com/chainer/chainerrl)  contains DQN, DDPG, TRPO, PPO, etc Reinforcment Learning algorithms.

### Environment
To model physics and biomechanics we use [OpenSim](https://github.com/opensim-org/opensim-core) - a biomechanical physics environment for musculoskeletal simulations.

### Installing
Install OpenSim Envirnment 
```
conda create -n opensim-rl -c kidzik opensim python=3.6.1
source activate opensim-rl
```
Install ChainerRL libary
```
pip install chainerrl
```


## References

[1] T. Garikayi, D. van den Heever and S. Matope, (2016), Robotic prosthetic challenges for clinical applications, IEEE International Conference on Control and Robotics Engineering (ICCRE), Singapore, 2016, pp. 1-5. doi: 10.1109/ICCRE.2016.7476146

[2] Joshi, Girish \& Chowdhary, Girish. (2018). Cross-Domain Transfer in Reinforcement Learning using Target Apprentice.

[3] Lillicrap, Timothy \& J. Hunt, Jonathan \& Pritzel, Alexander \& Heess, Nicolas \& Erez, Tom \& Tassa, Yuval \& Silver, David \& Wierstra, Daan. (2015). Continuous control with deep reinforcement learning. CoRR.

[4] Attia, Alexandre \& Dayan, Sharone. (2018). Global overview of Imitation Learning.

[5] Cheng, Qiao \& Wang, Xiangke \& Shen, Lincheng. (2017). An Autonomous Inter-task Mapping Learning Method via Artificial Neural Network for Transfer Learning. 10.1109/ROBIO.2017.8324510.

[7] J.J. Zhu, DAgger algorithm implementation, (2017), GitHub repository, https://github.com/jj-zhu/jadagger.
