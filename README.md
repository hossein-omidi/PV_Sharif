# Custom RL

Gymnasium-compatible ODE control environment framework with pluggable plants, reward functions, and RK4 integration.

This version uses a custom ODE-based CartPole environment with a **discrete action space** for **Deep Q-Network (DQN)** training.

## CartPole Action Space

```python
Discrete(2)