````markdown id="xv0vci"
# Custom RL

Gymnasium-compatible ODE control environment framework with pluggable plants, reward functions, and RK4 integration.

This version uses a custom ODE-based CartPole environment with a **discrete action space** for **Deep Q-Network (DQN)** training.

## CartPole Action Space

```python
Discrete(2)
````

```text
0 -> push cart left
1 -> push cart right
```

## Installation

```bash
pip install -e .
```

## Environment Usage

```python
import gymnasium as gym
from custom_rl import register_envs

register_envs()

env = gym.make("CustomODECartPole-v0", reward_id="sparse")
obs, info = env.reset(seed=42)

print(env.observation_space)
print(env.action_space)
```

Available rewards:

```text
dense
sparse
```

Recommended default for DQN CartPole:

```text
sparse
```

## Workflow

### 1. Verify Environment

```bash
python scripts/check_env.py --reward sparse
```

### 2. Train DQN

```bash
python scripts/train_dqn.py \
  --seeds 0 1 2 \
  --reward sparse \
  --total-timesteps 100000 \
  --log-dir logs/dqn_cartpole \
  --save-dir models/dqn_cartpole
```

For a quick test run:

```bash
python scripts/train_dqn.py \
  --seeds 0 \
  --reward sparse \
  --total-timesteps 10000
```

Common training options:

```text
--seeds
--reward dense|sparse
--total-timesteps
--log-dir
--save-dir
--learning-rate
--buffer-size
--learning-starts
--batch-size
--gamma
--train-freq
--gradient-steps
--target-update-interval
--exploration-initial-eps
--exploration-final-eps
--exploration-fraction
--net-arch
--eval-freq
--n-eval-episodes
```

### 3. Evaluate Policy

```bash
python scripts/eval_policy.py \
  --model-dir models/dqn_cartpole \
  --seeds 0 1 2 \
  --n-episodes 10 \
  --out-dir eval_trajectories \
  --reward sparse
```

### 4. Plot Results

```bash
python scripts/plot_results.py \
  --log-dir logs/dqn_cartpole \
  --traj-dir eval_trajectories \
  --out-dir plots \
  --seeds 0 1 2
```

## File Structure

```text
custom_rl/envs/         Generic ODE control environment and Gymnasium registration
custom_rl/plants/       ODE plant dynamics, including discrete-action CartPole
custom_rl/rewards/      Pluggable reward functions
custom_rl/integration/  RK4 integrator
scripts/check_env.py    Environment verification
scripts/train_dqn.py    DQN training
scripts/eval_policy.py  Policy evaluation and trajectory export
scripts/plot_results.py Training and evaluation plots
```

## Main Output Paths

```text
logs/dqn_cartpole
models/dqn_cartpole
eval_trajectories
plots
```

## Complete Command Sequence

```bash
python scripts/check_env.py --reward sparse

python scripts/train_dqn.py \
  --seeds 0 1 2 \
  --reward sparse \
  --total-timesteps 100000 \
  --log-dir logs/dqn_cartpole \
  --save-dir models/dqn_cartpole

python scripts/eval_policy.py \
  --model-dir models/dqn_cartpole \
  --seeds 0 1 2 \
  --n-episodes 10 \
  --out-dir eval_trajectories \
  --reward sparse

python scripts/plot_results.py \
  --log-dir logs/dqn_cartpole \
  --traj-dir eval_trajectories \
  --out-dir plots \
  --seeds 0 1 2
```

```
```
