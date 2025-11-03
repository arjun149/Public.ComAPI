import pytest
import numpy as np
from src.gym_env import MarketEnv

def test_env_step_reset():
    # Create environment with 2 assets and dummy data
    prices = np.random.rand(100, 2)  # 100 weeks, 2 assets
    env = MarketEnv(price_history=prices, long_short=True)
    
    obs = env.reset()
    assert obs.shape[0] > 0, "Observation should not be empty"

    # Take a random action
    action = np.random.uniform(-1, 1, size=2)
    obs, reward, done, info = env.step(action)
    
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done should be a boolean"
    assert isinstance(info, dict), "Info should be a dict"
