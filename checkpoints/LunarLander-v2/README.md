# LunarLander checkpoint

This is a checkpoint for a MLP DQN agent with `[128, 128]` hidden layers trained over 30,000 time steps.

Training parameters:

* Gamma: 0.99
* Epsilon schedule: drop from 1.0 to 0.1 over first 5,000 steps
* Replay buffer size: 1,000,000
* Target network soft-update parameter: 0.01
