# NeurIPS2018-Challenge-RL-for-Prosthetics

In this competition, you are tasked with developing a controller to enable a physiologically-based human model with a prosthetic leg to walk and run. You are provided with a human musculoskeletal model, a physics-based simulation environment where you can synthesize physically and physiologically accurate motion, and datasets of normal gait kinematics. You are scored based on how well your agent adapts to requested velocity vector changing in real time.

[![AI for prosthetics](https://s3-eu-west-1.amazonaws.com/kidzinski/nips-challenge/images/ai-prosthetics.jpg)](https://github.com/stanfordnmbl/osim-rl)

## Getting Started
### ChainerRL libary
[ChainerRL] (https://github.com/chainer/chainerrl) is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using Chainer, a flexible deep learning framework.

ChainerRL contains DQN, DDPG, TRPO, PPO, etc Reinforcment Learning algorithms.

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

