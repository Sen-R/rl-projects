# LunarLander checkpoint

This is a checkpoint for a MLP DQN agent with `[32, 32]` hidden layers trained over 30,000 time steps.

Training parameters:

* Gamma: 0.99
* Epsilon schedule: drop from 1.0 to 0.1 over first 2,000 steps
* Replay buffer size: 10,000
* Target network soft-update parameter: 0.01
