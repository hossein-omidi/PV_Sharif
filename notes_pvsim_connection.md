Goal:
We do not modify pvsim directly. We treat it as a black box simulator.

Current status:
- Repository cloned successfully.
- Virtual environment created.
- Dependencies installed.
- Existing scripts run without error.
- Current RL test is based on CartPole.
- Next step is to wrap pvsim as a Gymnasium environment.

Planned connection:
RL Agent
→ action: change panel tilt / azimuth
→ Gymnasium wrapper
→ pvsim simulation
→ output power
→ reward
→ next state

Current implementation:
- `PVSimEnv` is an initial scaffold and does not call the real pvsim yet.
- The temporary reward is based on a simple angle-error approximation.
- The next step is to replace the temporary reward calculation with actual pvsim output power.
- `PVSimEnv` should be registered in `registration.py` and created using `gym.make("PVSimEnv-v0")`.